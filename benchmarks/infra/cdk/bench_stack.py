"""Per-branch benchmark infrastructure: SQS queue, ASG, scaling, IAM.

Creates an isolated pipeline per git branch:
  SQS (work queue) → ASG (spot workers from AMI) → S3 (results)

Workers poll SQS for benchmark permutation IDs, run them via the harness CLI,
upload results to S3, and terminate when the queue is empty. The ASG scales
from 0 to max_workers based on queue depth.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import Duration, Tags
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sqs as sqs
from constructs import Construct

PROJECT_TAG = "muninn-benchmarks"
USER_DATA_TEMPLATE = Path(__file__).parent.parent / "worker_user_data.sh"


class BenchStack(cdk.Stack):
    """Per-branch benchmark stack: SQS + ASG + scaling."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        branch: str,
        ami_id: str,
        s3_bucket: str,
        s3_region: str,
        max_workers: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("project", PROJECT_TAG)
        Tags.of(self).add("branch", branch)

        # ── SQS: work queue + dead letter queue ───────────────────
        dlq = sqs.Queue(
            self,
            "DLQ",
            retention_period=Duration.days(14),
        )

        queue = sqs.Queue(
            self,
            "Queue",
            visibility_timeout=Duration.hours(2),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=dlq,
            ),
        )

        # ── IAM: worker role ──────────────────────────────────────
        role = iam.Role(
            self,
            "WorkerRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
        )

        # S3 access for results + heartbeat
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
                resources=[
                    f"arn:aws:s3:::{s3_bucket}",
                    f"arn:aws:s3:::{s3_bucket}/*",
                ],
            )
        )

        # SQS consume
        queue.grant_consume_messages(role)

        # SSM for remote access
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"))

        # CloudWatch Logs
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["arn:aws:logs:*:*:log-group:/muninn/*"],
            )
        )

        # Instance profile
        iam.CfnInstanceProfile(
            self,
            "InstanceProfile",
            roles=[role.role_name],
        )

        # ── Security group ────────────────────────────────────────
        # Use default VPC
        vpc = ec2.Vpc.from_lookup(self, "VPC", is_default=True)

        sg = ec2.SecurityGroup(
            self,
            "SG",
            vpc=vpc,
            description="Muninn benchmark workers - SSH access",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(22), "SSH")

        # ── Worker script → S3 (not user-data) ──────────────────
        # The AMI has a systemd service (muninn-worker.service) that runs on
        # every boot and downloads s3://{bucket}/scripts/worker.sh. This
        # bypasses cloud-init entirely — no first-boot vs subsequent-boot issues.
        #
        # CDK uploads the rendered worker script to S3 via a custom resource.
        worker_script = USER_DATA_TEMPLATE.read_text(encoding="utf-8")
        worker_script = worker_script.replace("__S3_BUCKET__", s3_bucket)
        worker_script = worker_script.replace("__S3_REGION__", s3_region)
        worker_script = worker_script.replace("__SQS_QUEUE_URL__", queue.queue_url)
        worker_script = worker_script.replace("__REPO_URL__", "https://github.com/neozenith/sqlite-muninn.git")
        worker_script = worker_script.replace("__BRANCH__", branch)

        # Store for upload by runner.py deploy step
        self._worker_script = worker_script
        self._worker_s3_key = "scripts/worker.sh"

        # No user-data needed — the systemd service handles boot execution
        user_data = ec2.UserData.for_linux()

        # ── Launch template ───────────────────────────────────────
        lt = ec2.LaunchTemplate(
            self,
            "LT",
            machine_image=ec2.MachineImage.generic_linux({self.region: ami_id}),
            instance_type=ec2.InstanceType("t3.xlarge"),
            user_data=user_data,
            security_group=sg,
            role=role,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(
                        40,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        delete_on_termination=True,
                    ),
                ),
            ],
        )

        # ── ASG: spot workers with on-demand fallback ─────────────
        asg = autoscaling.AutoScalingGroup(
            self,
            "ASG",
            vpc=vpc,
            min_capacity=0,
            max_capacity=max_workers,
            desired_capacity=0,
            mixed_instances_policy=autoscaling.MixedInstancesPolicy(
                launch_template=lt,
                instances_distribution=autoscaling.InstancesDistribution(
                    on_demand_base_capacity=0,
                    on_demand_percentage_above_base_capacity=0,
                    spot_allocation_strategy=autoscaling.SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED,
                ),
                launch_template_overrides=[
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("t3.xlarge")),
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("m5.xlarge")),
                    autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType("c5.xlarge")),
                ],
            ),
            cooldown=Duration.minutes(5),
            new_instances_protected_from_scale_in=False,
            group_metrics=[autoscaling.GroupMetrics.all()],
        )

        # ── Scaling: backlog-per-instance (AWS recommended for SQS) ─
        #
        # The naive approach (scale on visible messages) fails because
        # messages go invisible when pulled by a worker. The alarm sees
        # 0 visible and scales in, killing workers mid-benchmark.
        #
        # Fix: use a math expression that considers BOTH visible AND
        # in-flight messages divided by running capacity. This gives
        # a "backlog per instance" metric that stays > 0 while work
        # is being processed.
        #
        # Scale out: backlog_per_instance >= 1 → need more workers
        # Scale in:  backlog_per_instance == 0 for 10 min → scale to zero

        visible = queue.metric_approximate_number_of_messages_visible(
            period=Duration.minutes(1),
            statistic="Maximum",
        )
        not_visible = queue.metric_approximate_number_of_messages_not_visible(
            period=Duration.minutes(1),
            statistic="Maximum",
        )

        total_messages = cloudwatch.MathExpression(
            expression="visible + inflight",
            using_metrics={"visible": visible, "inflight": not_visible},
            period=Duration.minutes(1),
        )

        # Scale out: any messages in queue → add capacity
        cloudwatch.Alarm(
            self,
            "ScaleOutAlarm",
            metric=total_messages,
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # Scale in: zero messages (visible + in-flight) for 10 minutes → safe to remove
        cloudwatch.Alarm(
            self,
            "ScaleInAlarm",
            metric=total_messages,
            threshold=0,
            evaluation_periods=10,  # 10 x 1-minute periods = 10 min of zero
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        )

        asg.scale_on_metric(
            "ScaleOnBacklog",
            metric=total_messages,
            scaling_steps=[
                autoscaling.ScalingInterval(change=-1, upper=0),  # 0 messages → scale in
                autoscaling.ScalingInterval(change=+1, lower=1, upper=5),  # 1-5 → +1
                autoscaling.ScalingInterval(change=+2, lower=5),  # 5+ → +2
            ],
            adjustment_type=autoscaling.AdjustmentType.CHANGE_IN_CAPACITY,
            evaluation_periods=10,  # require 10 consecutive periods before scaling in
        )

        # ── Outputs ───────────────────────────────────────────────
        cdk.CfnOutput(self, "QueueUrl", value=queue.queue_url)
        cdk.CfnOutput(self, "QueueArn", value=queue.queue_arn)
        cdk.CfnOutput(self, "DlqUrl", value=dlq.queue_url)
        cdk.CfnOutput(self, "AsgName", value=asg.auto_scaling_group_name)
        cdk.CfnOutput(self, "Branch", value=branch)
