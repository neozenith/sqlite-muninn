"""AMI cleanup stack: Lambda + EventBridge weekly schedule.

Prunes AMIs tagged with project=muninn-benchmarks that are older than 7 days.
Also deletes the backing EBS snapshots to avoid cost leaks.

The Lambda can be invoked manually:
    aws lambda invoke --function-name MuninnAmiCleanup /dev/stdout
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import Duration, Tags
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from constructs import Construct

PROJECT_TAG = "muninn-benchmarks"
LAMBDA_DIR = Path(__file__).parent / "lambda_fn"


class CleanupStack(cdk.Stack):
    """AMI cleanup: weekly prune of stale benchmark AMIs."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("project", PROJECT_TAG)

        # ── Lambda function ───────────────────────────────────────
        cleanup_fn = lambda_.Function(
            self,
            "AmiCleanup",
            function_name="MuninnAmiCleanup",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="ami_cleanup.handler",
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            timeout=Duration.minutes(5),
            environment={
                "MAX_AGE_DAYS": "7",
                "PROJECT_TAG": PROJECT_TAG,
            },
        )

        # EC2 permissions: describe + deregister AMIs, delete snapshots
        cleanup_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "ec2:DescribeImages",
                    "ec2:DeregisterImage",
                    "ec2:DescribeSnapshots",
                    "ec2:DeleteSnapshot",
                ],
                resources=["*"],
            )
        )

        # ── EventBridge: weekly schedule (Sunday 00:00 UTC) ───────
        rule = events.Rule(
            self,
            "WeeklyPrune",
            schedule=events.Schedule.cron(
                minute="0",
                hour="0",
                week_day="SUN",
            ),
        )
        rule.add_target(targets.LambdaFunction(cleanup_fn))

        # ── Outputs ───────────────────────────────────────────────
        cdk.CfnOutput(
            self,
            "LambdaName",
            value=cleanup_fn.function_name,
            description="Invoke manually: aws lambda invoke --function-name MuninnAmiCleanup /dev/stdout",
        )
