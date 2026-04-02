"""EC2 benchmark runner — launch, monitor, and manage remote benchmark instances.

Single script for the full lifecycle: setup AWS resources, launch a spot (or
on-demand) instance, monitor it via S3 heartbeat every 15 seconds, collect
results, and tear down.

The instance self-selects benchmarks via `manifest --commands --missing --limit N`
so it always picks the next incomplete work. If interrupted (spot reclamation),
the benchmark stays "missing" and the next run retries it automatically.

Usage:
    uv run benchmarks/infra/runner.py setup              # create AWS resources
    uv run benchmarks/infra/runner.py prime              # cold start + create AMI
    uv run benchmarks/infra/runner.py run                # single instance, 1 benchmark
    uv run benchmarks/infra/runner.py submit             # enqueue to SQS for parallel
    uv run benchmarks/infra/runner.py submit --limit 10  # enqueue up to 10
    uv run benchmarks/infra/runner.py status             # check instance + heartbeat
    uv run benchmarks/infra/runner.py teardown           # terminate instance
    uv run benchmarks/infra/runner.py teardown --all     # terminate + delete all resources
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "benchmarks" / "infra" / "config.yml"
USER_DATA_PATH = PROJECT_ROOT / "benchmarks" / "infra" / "user_data.sh"
STATE_DIR = PROJECT_ROOT / "tmp" / "ec2_runs"
STATE_FILE = STATE_DIR / "current.json"
KEY_DIR = STATE_DIR / "keys"

TAG_KEY = "project"
TAG_VALUE = "muninn-benchmarks"

# ── Config ────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load config from YAML file, with env var overrides."""
    if not CONFIG_PATH.exists():
        log.error("Config not found: %s", CONFIG_PATH)
        log.error("Copy config.yml.example to config.yml and fill in your values.")
        sys.exit(1)

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Env var overrides
    env_map = {
        ("aws", "ec2_region"): "BENCH_EC2_REGION",
        ("aws", "s3_bucket"): "BENCH_S3_BUCKET",
        ("aws", "s3_region"): "BENCH_S3_REGION",
        ("aws", "instance_type"): "BENCH_INSTANCE_TYPE",
        ("aws", "use_spot"): "BENCH_USE_SPOT",
        ("aws", "ami_id"): "BENCH_AMI_ID",
        ("aws", "key_name"): "BENCH_KEY_NAME",
        ("aws", "security_group_id"): "BENCH_SECURITY_GROUP_ID",
        ("aws", "instance_profile_name"): "BENCH_INSTANCE_PROFILE_NAME",
        ("repo", "url"): "BENCH_REPO_URL",
        ("repo", "branch"): "BENCH_BRANCH",
        ("benchmark", "category"): "BENCH_CATEGORY",
        ("benchmark", "limit"): "BENCH_LIMIT",
    }

    for (section, key), env_var in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            if key == "use_spot":
                val = val.lower() in ("true", "1", "yes")
            elif key == "limit":
                val = int(val)
            cfg.setdefault(section, {})[key] = val

    # Validate required fields
    required = [
        ("aws", "s3_bucket"),
        ("aws", "ami_id"),
        ("aws", "ec2_region"),
        ("repo", "url"),
        ("repo", "branch"),
    ]
    for section, key in required:
        if not cfg.get(section, {}).get(key):
            log.error("Missing required config: %s.%s", section, key)
            sys.exit(1)

    return cfg


def _aws(cfg: dict) -> dict:
    """Shorthand for cfg['aws']."""
    return cfg["aws"]


# ── State management ──────────────────────────────────────────────


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ── User data rendering ──────────────────────────────────────────


def _render_user_data(cfg: dict) -> str:
    """Read user_data.sh template and substitute config values."""
    template = USER_DATA_PATH.read_text(encoding="utf-8")
    replacements = {
        "__S3_BUCKET__": cfg["aws"]["s3_bucket"],
        "__S3_REGION__": cfg["aws"]["s3_region"],
        "__REPO_URL__": cfg["repo"]["url"],
        "__BRANCH__": cfg["repo"]["branch"],
        "__BENCH_CATEGORY__": cfg.get("benchmark", {}).get("category") or "",
        "__BENCH_LIMIT__": str(cfg.get("benchmark", {}).get("limit", 1)),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


# ── Heartbeat ─────────────────────────────────────────────────────


def _get_heartbeat(s3, cfg: dict, instance_id: str) -> dict | None:
    """Fetch the latest heartbeat from S3."""
    try:
        resp = s3.get_object(
            Bucket=_aws(cfg)["s3_bucket"],
            Key=f"heartbeat/{instance_id}.json",
        )
        return json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _heartbeat_age(hb: dict | None) -> float:
    """Return heartbeat age in seconds, or inf if no heartbeat."""
    if hb is None:
        return float("inf")
    ts = datetime.fromisoformat(hb["timestamp"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - ts).total_seconds()


# ── Commands ──────────────────────────────────────────────────────


def cmd_setup(args: argparse.Namespace) -> None:
    """Create AWS resources: IAM role, instance profile, key pair, security group."""
    cfg = load_config()
    aws_cfg = _aws(cfg)
    ec2_region = aws_cfg["ec2_region"]
    s3_bucket = aws_cfg["s3_bucket"]

    iam = boto3.client("iam")
    ec2 = boto3.client("ec2", region_name=ec2_region)

    role_name = "muninn-bench-ec2-role"
    profile_name = "muninn-bench-ec2-profile"
    key_name = f"muninn-bench-key-{ec2_region}"
    sg_name = "muninn-bench-sg"

    # IAM Role
    log.info("Creating IAM role: %s", role_name)
    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }],
            }),
            Tags=[{"Key": TAG_KEY, "Value": TAG_VALUE}],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            log.info("  Role already exists")
        else:
            raise

    # Policies
    s3_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
            "Resource": [f"arn:aws:s3:::{s3_bucket}", f"arn:aws:s3:::{s3_bucket}/*"],
        }],
    })
    iam.put_role_policy(RoleName=role_name, PolicyName="s3-access", PolicyDocument=s3_policy)

    try:
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        )
    except ClientError:
        pass

    # Instance profile
    log.info("Creating instance profile: %s", profile_name)
    try:
        iam.create_instance_profile(
            InstanceProfileName=profile_name,
            Tags=[{"Key": TAG_KEY, "Value": TAG_VALUE}],
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    try:
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
    except ClientError as e:
        if "LimitExceeded" not in str(e) and "already" not in str(e).lower():
            raise

    # Key pair
    log.info("Creating key pair: %s", key_name)
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key_path = KEY_DIR / f"{key_name}.pem"
    try:
        resp = ec2.create_key_pair(KeyName=key_name)
        key_path.write_text(resp["KeyMaterial"], encoding="utf-8")
        key_path.chmod(0o600)
        log.info("  Private key saved to %s", key_path)
    except ClientError as e:
        if "InvalidKeyPair.Duplicate" in str(e):
            log.info("  Key pair already exists")
        else:
            raise

    # Security group
    log.info("Creating security group: %s", sg_name)
    try:
        resp = ec2.create_security_group(
            GroupName=sg_name,
            Description="Muninn benchmark runner - SSH access",
            TagSpecifications=[{
                "ResourceType": "security-group",
                "Tags": [{"Key": TAG_KEY, "Value": TAG_VALUE}],
            }],
        )
        sg_id = resp["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id, IpProtocol="tcp", FromPort=22, ToPort=22, CidrIp="0.0.0.0/0",
        )
        log.info("  Security group created: %s", sg_id)
    except ClientError as e:
        if "InvalidGroup.Duplicate" in str(e):
            desc = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = desc["SecurityGroups"][0]["GroupId"]
            log.info("  Security group already exists: %s", sg_id)
        else:
            raise

    # Wait for IAM propagation
    log.info("Waiting for IAM propagation (10s)...")
    time.sleep(10)

    # Update config
    log.info("Updating config.yml with created resource IDs...")
    cfg["aws"]["key_name"] = key_name
    cfg["aws"]["security_group_id"] = sg_id
    cfg["aws"]["instance_profile_name"] = profile_name
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    log.info("Setup complete. Run 'uv run benchmarks/infra/runner.py run' to launch.")


def _sanitize_branch(branch: str) -> str:
    """Sanitize branch name for AWS resource names (alphanumeric + hyphens)."""
    return re.sub(r"[^a-zA-Z0-9]", "-", branch).strip("-")[:64]


def _get_git_short_hash() -> str:
    """Get the current git short hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout.strip() or "unknown"


def cmd_prime(args: argparse.Namespace) -> None:
    """Launch an on-demand instance, run cold start, create AMI for warm launches.

    The instance runs the full bootstrap (deps, clone, build, uv sync) then shuts down.
    Once stopped, an AMI is created from its EBS volume. Future workers launched from
    this AMI skip the cold start entirely.
    """
    cfg = load_config()
    aws_cfg = _aws(cfg)
    branch = cfg["repo"]["branch"]
    safe_branch = _sanitize_branch(branch)
    commit_hash = _get_git_short_hash()

    ec2 = boto3.client("ec2", region_name=aws_cfg["ec2_region"])
    s3 = boto3.client("s3", region_name=aws_cfg["s3_region"])

    # Check for existing tracked instance
    state = _load_state()
    if state.get("instance_id"):
        log.error("Instance %s already tracked. Run 'teardown' first.", state["instance_id"])
        sys.exit(1)

    # Render user-data — set limit=0 so no benchmarks run (prime only)
    prime_cfg = json.loads(json.dumps(cfg))  # deep copy
    prime_cfg.setdefault("benchmark", {})["limit"] = "0"
    user_data = _render_user_data(prime_cfg)

    log.info("Priming AMI for branch '%s' (commit %s)...", branch, commit_hash)
    log.info("Launching on-demand %s (cold start, no benchmarks)...", aws_cfg["instance_type"])

    resp = ec2.run_instances(
        ImageId=aws_cfg["ami_id"],
        InstanceType=aws_cfg["instance_type"],
        KeyName=aws_cfg["key_name"],
        SecurityGroupIds=[aws_cfg["security_group_id"]],
        IamInstanceProfile={"Name": aws_cfg["instance_profile_name"]},
        InstanceInitiatedShutdownBehavior="stop",
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"muninn-prime-{safe_branch}"},
                {"Key": TAG_KEY, "Value": TAG_VALUE},
                {"Key": "branch", "Value": branch},
            ],
        }],
        MinCount=1,
        MaxCount=1,
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    launch_time = datetime.now(timezone.utc)

    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "none")

    key_path = KEY_DIR / f"{aws_cfg['key_name']}.pem"
    _save_state({
        "instance_id": instance_id,
        "instance_type": aws_cfg["instance_type"],
        "mode": "prime",
        "region": aws_cfg["ec2_region"],
        "public_ip": public_ip,
        "launched_at": launch_time.isoformat(),
        "key_file": str(key_path),
    })

    log.info("Instance %s running at %s", instance_id, public_ip)
    log.info("SSH: ssh -i %s ubuntu@%s", key_path, public_ip)
    log.info("")

    # ── Monitor until instance stops (cold start complete) ────────
    hb_cfg = cfg.get("heartbeat", {})
    poll_interval = hb_cfg.get("interval", 15)
    stale_threshold = hb_cfg.get("stale_threshold", 60)
    hung_threshold = hb_cfg.get("hung_threshold", 180)

    consecutive_stale = 0
    last_phase = None

    log.info("Monitoring cold start (waiting for instance to stop after bootstrap)...")

    while True:
        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%H:%M:%S")

        try:
            desc = ec2.describe_instances(InstanceIds=[instance_id])
            ec2_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception:
            ec2_state = "unknown"

        hb = _get_heartbeat(s3, cfg, instance_id)
        age = _heartbeat_age(hb)

        if ec2_state == "stopped":
            log.info("[%s] Instance stopped. Cold start complete.", ts_str)
            break

        if ec2_state in ("terminated", "shutting-down"):
            log.error("[%s] Instance %s unexpectedly. Aborting prime.", ts_str, ec2_state)
            _clear_state()
            sys.exit(1)

        if hb:
            phase = hb.get("phase", "?")
            if phase != last_phase:
                if last_phase is not None:
                    log.info("[%s] PHASE: %s -> %s", ts_str, last_phase, phase)
                last_phase = phase

            if age > stale_threshold:
                consecutive_stale += 1
                log.warning("[%s] STALE (%.0fs) phase=%s [%d]", ts_str, age, phase, consecutive_stale)
            else:
                consecutive_stale = 0
                log.info("[%s] OK (%.0fs ago) phase=%s", ts_str, age, phase)
        else:
            consecutive_stale += 1
            if consecutive_stale > 4:
                log.warning("[%s] NO heartbeat (%d consecutive)", ts_str, consecutive_stale)

        if consecutive_stale * poll_interval >= hung_threshold:
            log.error("[%s] HUNG. Terminating %s.", ts_str, instance_id)
            ec2.terminate_instances(InstanceIds=[instance_id])
            _clear_state()
            sys.exit(1)

        time.sleep(poll_interval)

    # ── Create AMI from the stopped instance ──────────────────────
    ami_name = f"muninn-bench-{safe_branch}-{commit_hash}-{launch_time.strftime('%Y%m%d')}"
    log.info("Creating AMI: %s", ami_name)

    ami_resp = ec2.create_image(
        InstanceId=instance_id,
        Name=ami_name,
        Description=f"Primed benchmark runner for branch {branch} at {commit_hash}",
        NoReboot=True,
        TagSpecifications=[{
            "ResourceType": "image",
            "Tags": [
                {"Key": TAG_KEY, "Value": TAG_VALUE},
                {"Key": "branch", "Value": branch},
                {"Key": "commit", "Value": commit_hash},
            ],
        }],
    )

    ami_id = ami_resp["ImageId"]
    log.info("AMI %s creating (this takes 5-15 minutes)...", ami_id)

    ec2.get_waiter("image_available").wait(
        ImageIds=[ami_id],
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},  # up to 30 minutes
    )
    log.info("AMI %s ready.", ami_id)

    # Terminate the prime instance (AMI captured the volume)
    log.info("Terminating prime instance %s...", instance_id)
    ec2.terminate_instances(InstanceIds=[instance_id])
    _clear_state()

    # Update config with the new AMI
    cfg["aws"]["ami_id"] = ami_id
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    elapsed = (datetime.now(timezone.utc) - launch_time).total_seconds()
    log.info("")
    log.info("Prime complete in %.0fs.", elapsed)
    log.info("AMI: %s", ami_id)
    log.info("Branch: %s (commit %s)", branch, commit_hash)
    log.info("config.yml updated with new ami_id.")
    log.info("")
    log.info("Next steps:")
    log.info("  uv run benchmarks/infra/runner.py run       # single instance from AMI")
    log.info("  uv run benchmarks/infra/runner.py submit    # enqueue to SQS for parallel workers")



    """Launch an instance, run benchmarks, monitor heartbeat, collect results."""
    cfg = load_config()
    aws_cfg = _aws(cfg)

    # Check for existing instance
    state = _load_state()
    if state.get("instance_id"):
        log.error("Instance %s already tracked. Run 'teardown' first.", state["instance_id"])
        sys.exit(1)

    ec2 = boto3.client("ec2", region_name=aws_cfg["ec2_region"])
    s3 = boto3.client("s3", region_name=aws_cfg["s3_region"])

    user_data = _render_user_data(cfg)
    use_spot = aws_cfg.get("use_spot", True)

    # For spot: terminate on shutdown (EBS destroyed — clean, no orphans)
    # For on-demand: stop on shutdown (EBS preserved — warm restart)
    shutdown_behavior = "terminate" if use_spot else "stop"

    run_params = {
        "ImageId": aws_cfg["ami_id"],
        "InstanceType": aws_cfg["instance_type"],
        "KeyName": aws_cfg["key_name"],
        "SecurityGroupIds": [aws_cfg["security_group_id"]],
        "IamInstanceProfile": {"Name": aws_cfg["instance_profile_name"]},
        "InstanceInitiatedShutdownBehavior": shutdown_behavior,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        "UserData": user_data,
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "muninn-benchmark"},
                {"Key": TAG_KEY, "Value": TAG_VALUE},
            ],
        }],
        "MinCount": 1,
        "MaxCount": 1,
    }

    if use_spot:
        run_params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    mode = "spot" if use_spot else "on-demand"
    log.info("Launching %s %s instance in %s...", mode, aws_cfg["instance_type"], aws_cfg["ec2_region"])

    try:
        resp = ec2.run_instances(**run_params)
    except ClientError as e:
        if use_spot and "InsufficientInstanceCapacity" in str(e):
            log.warning("Spot capacity unavailable. Falling back to on-demand.")
            run_params.pop("InstanceMarketOptions")
            run_params["InstanceInitiatedShutdownBehavior"] = "stop"
            mode = "on-demand (fallback)"
            resp = ec2.run_instances(**run_params)
        else:
            raise

    instance = resp["Instances"][0]
    instance_id = instance["InstanceId"]
    launch_time = datetime.now(timezone.utc)

    log.info("Instance %s launching (%s)...", instance_id, mode)

    # Wait for running
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "none")

    key_path = KEY_DIR / f"{aws_cfg['key_name']}.pem"

    state = {
        "instance_id": instance_id,
        "instance_type": aws_cfg["instance_type"],
        "mode": mode,
        "region": aws_cfg["ec2_region"],
        "public_ip": public_ip,
        "launched_at": launch_time.isoformat(),
        "key_file": str(key_path),
    }
    _save_state(state)

    log.info("Instance %s running at %s", instance_id, public_ip)
    log.info("SSH: ssh -i %s ubuntu@%s", key_path, public_ip)
    log.info("Logs: tail -f /var/log/muninn/benchmark.log")
    log.info("")

    # ── Monitor heartbeat ─────────────────────────────────────────
    hb_cfg = cfg.get("heartbeat", {})
    poll_interval = hb_cfg.get("interval", 15)
    stale_threshold = hb_cfg.get("stale_threshold", 60)
    hung_threshold = hb_cfg.get("hung_threshold", 180)

    timing = {
        "launch_at": launch_time.isoformat(),
        "mode": mode,
        "instance_id": instance_id,
        "instance_type": aws_cfg["instance_type"],
        "first_heartbeat_at": None,
        "phases": [],
        "outcome": None,
        "stopped_at": None,
    }

    consecutive_stale = 0
    last_phase = None
    first_heartbeat = False

    log.info("Monitoring heartbeat every %ds (stale >%ds, hung >%ds)", poll_interval, stale_threshold, hung_threshold)

    while True:
        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%H:%M:%S")

        # Check instance state
        try:
            desc = ec2.describe_instances(InstanceIds=[instance_id])
            ec2_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception:
            ec2_state = "unknown"

        # Check heartbeat
        hb = _get_heartbeat(s3, cfg, instance_id)
        age = _heartbeat_age(hb)

        # Instance terminated/stopped → job done
        if ec2_state in ("stopped", "terminated", "shutting-down"):
            timing["stopped_at"] = now.isoformat()
            if hb is None or (hb and hb.get("phase") == "stopped"):
                timing["outcome"] = "completed"
                log.info("[%s] Instance %s — job completed.", ts_str, ec2_state)
            else:
                timing["outcome"] = "interrupted" if use_spot else "stopped"
                log.warning("[%s] Instance %s — %s", ts_str, ec2_state, timing["outcome"])
            break

        # Heartbeat present
        if hb:
            phase = hb.get("phase", "?")

            if not first_heartbeat:
                first_heartbeat = True
                timing["first_heartbeat_at"] = now.isoformat()
                log.info("[%s] First heartbeat received (phase=%s)", ts_str, phase)

            if phase != last_phase:
                timing["phases"].append({"phase": phase, "at": now.isoformat()})
                if last_phase is not None:
                    log.info("[%s] PHASE: %s → %s", ts_str, last_phase, phase)
                last_phase = phase

            if age > stale_threshold:
                consecutive_stale += 1
                log.warning("[%s] STALE heartbeat (%.0fs old, %d consecutive) phase=%s", ts_str, age, consecutive_stale, phase)
            else:
                if consecutive_stale > 0:
                    log.info("[%s] Heartbeat recovered after %d stale polls", ts_str, consecutive_stale)
                consecutive_stale = 0
                log.info("[%s] OK (%.0fs ago) ec2=%s phase=%s", ts_str, age, ec2_state, phase)
        else:
            consecutive_stale += 1
            log.warning("[%s] NO heartbeat (ec2=%s, %d consecutive)", ts_str, ec2_state, consecutive_stale)

        # Auto-terminate if hung
        if consecutive_stale * poll_interval >= hung_threshold:
            log.error(
                "[%s] HUNG: no heartbeat for %ds. Terminating %s.",
                ts_str, consecutive_stale * poll_interval, instance_id,
            )
            ec2.terminate_instances(InstanceIds=[instance_id])
            timing["outcome"] = "hung_terminated"
            timing["stopped_at"] = now.isoformat()
            break

        time.sleep(poll_interval)

    # ── Save timing log ───────────────────────────────────────────
    _clear_state()
    timing_path = STATE_DIR / f"{launch_time.strftime('%Y%m%d_%H%M%S')}.json"
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log.info("Timing log: %s", timing_path)

    # ── Show phase log if available ───────────────────────────────
    _show_latest_phase_log(s3, cfg)

    # Exit code reflects outcome
    if timing["outcome"] == "completed":
        log.info("Success.")
    elif timing["outcome"] == "interrupted":
        log.warning("Spot instance interrupted. Re-run to retry the benchmark.")
        sys.exit(2)
    else:
        log.error("Instance hung or failed.")
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Launch an instance, run benchmarks, monitor heartbeat, collect results."""
    cfg = load_config()
    aws_cfg = _aws(cfg)

    state = _load_state()
    if state.get("instance_id"):
        log.error("Instance %s already tracked. Run 'teardown' first.", state["instance_id"])
        sys.exit(1)

    ec2 = boto3.client("ec2", region_name=aws_cfg["ec2_region"])
    s3 = boto3.client("s3", region_name=aws_cfg["s3_region"])

    user_data = _render_user_data(cfg)
    use_spot = aws_cfg.get("use_spot", True)
    shutdown_behavior = "terminate" if use_spot else "stop"

    run_params = {
        "ImageId": aws_cfg["ami_id"],
        "InstanceType": aws_cfg["instance_type"],
        "KeyName": aws_cfg["key_name"],
        "SecurityGroupIds": [aws_cfg["security_group_id"]],
        "IamInstanceProfile": {"Name": aws_cfg["instance_profile_name"]},
        "InstanceInitiatedShutdownBehavior": shutdown_behavior,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 20, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        "UserData": user_data,
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "muninn-benchmark"},
                {"Key": TAG_KEY, "Value": TAG_VALUE},
            ],
        }],
        "MinCount": 1,
        "MaxCount": 1,
    }

    if use_spot:
        run_params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    mode = "spot" if use_spot else "on-demand"
    log.info("Launching %s %s instance in %s...", mode, aws_cfg["instance_type"], aws_cfg["ec2_region"])

    try:
        resp = ec2.run_instances(**run_params)
    except ClientError as e:
        if use_spot and "InsufficientInstanceCapacity" in str(e):
            log.warning("Spot capacity unavailable. Falling back to on-demand.")
            run_params.pop("InstanceMarketOptions")
            run_params["InstanceInitiatedShutdownBehavior"] = "stop"
            mode = "on-demand (fallback)"
            resp = ec2.run_instances(**run_params)
        else:
            raise

    instance = resp["Instances"][0]
    instance_id = instance["InstanceId"]
    launch_time = datetime.now(timezone.utc)

    log.info("Instance %s launching (%s)...", instance_id, mode)

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "none")

    key_path = KEY_DIR / f"{aws_cfg['key_name']}.pem"

    state = {
        "instance_id": instance_id,
        "instance_type": aws_cfg["instance_type"],
        "mode": mode,
        "region": aws_cfg["ec2_region"],
        "public_ip": public_ip,
        "launched_at": launch_time.isoformat(),
        "key_file": str(key_path),
    }
    _save_state(state)

    log.info("Instance %s running at %s", instance_id, public_ip)
    log.info("SSH: ssh -i %s ubuntu@%s", key_path, public_ip)
    log.info("")

    # ── Monitor heartbeat ─────────────────────────────────────────
    hb_cfg = cfg.get("heartbeat", {})
    poll_interval = hb_cfg.get("interval", 15)
    stale_threshold = hb_cfg.get("stale_threshold", 60)
    hung_threshold = hb_cfg.get("hung_threshold", 180)

    timing = {
        "launch_at": launch_time.isoformat(),
        "mode": mode,
        "instance_id": instance_id,
        "instance_type": aws_cfg["instance_type"],
        "first_heartbeat_at": None,
        "phases": [],
        "outcome": None,
        "stopped_at": None,
    }

    consecutive_stale = 0
    last_phase = None
    first_heartbeat = False

    log.info("Monitoring heartbeat every %ds (stale >%ds, hung >%ds)", poll_interval, stale_threshold, hung_threshold)

    while True:
        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%H:%M:%S")

        try:
            desc = ec2.describe_instances(InstanceIds=[instance_id])
            ec2_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception:
            ec2_state = "unknown"

        hb = _get_heartbeat(s3, cfg, instance_id)
        age = _heartbeat_age(hb)

        if ec2_state in ("stopped", "terminated", "shutting-down"):
            timing["stopped_at"] = now.isoformat()
            if hb is None or (hb and hb.get("phase") == "stopped"):
                timing["outcome"] = "completed"
                log.info("[%s] Instance %s — job completed.", ts_str, ec2_state)
            else:
                timing["outcome"] = "interrupted" if use_spot else "stopped"
                log.warning("[%s] Instance %s — %s", ts_str, ec2_state, timing["outcome"])
            break

        if hb:
            phase = hb.get("phase", "?")

            if not first_heartbeat:
                first_heartbeat = True
                timing["first_heartbeat_at"] = now.isoformat()
                log.info("[%s] First heartbeat received (phase=%s)", ts_str, phase)

            if phase != last_phase:
                timing["phases"].append({"phase": phase, "at": now.isoformat()})
                if last_phase is not None:
                    log.info("[%s] PHASE: %s -> %s", ts_str, last_phase, phase)
                last_phase = phase

            if age > stale_threshold:
                consecutive_stale += 1
                log.warning("[%s] STALE heartbeat (%.0fs old, %d consecutive) phase=%s", ts_str, age, consecutive_stale, phase)
            else:
                if consecutive_stale > 0:
                    log.info("[%s] Heartbeat recovered after %d stale polls", ts_str, consecutive_stale)
                consecutive_stale = 0
                log.info("[%s] OK (%.0fs ago) ec2=%s phase=%s", ts_str, age, ec2_state, phase)
        else:
            consecutive_stale += 1
            log.warning("[%s] NO heartbeat (ec2=%s, %d consecutive)", ts_str, ec2_state, consecutive_stale)

        if consecutive_stale * poll_interval >= hung_threshold:
            log.error(
                "[%s] HUNG: no heartbeat for %ds. Terminating %s.",
                ts_str, consecutive_stale * poll_interval, instance_id,
            )
            ec2.terminate_instances(InstanceIds=[instance_id])
            timing["outcome"] = "hung_terminated"
            timing["stopped_at"] = now.isoformat()
            break

        time.sleep(poll_interval)

    # ── Save timing log ───────────────────────────────────────────
    _clear_state()
    timing_path = STATE_DIR / f"{launch_time.strftime('%Y%m%d_%H%M%S')}.json"
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log.info("Timing log: %s", timing_path)

    _show_latest_phase_log(s3, cfg)

    if timing["outcome"] == "completed":
        log.info("Success.")
    elif timing["outcome"] == "interrupted":
        log.warning("Spot instance interrupted. Re-run to retry the benchmark.")
        sys.exit(2)
    else:
        log.error("Instance hung or failed.")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Quick status check — instance state + heartbeat."""
    cfg = load_config()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        log.info("No instance tracked.")
        return

    ec2 = boto3.client("ec2", region_name=_aws(cfg)["ec2_region"])
    s3 = boto3.client("s3", region_name=_aws(cfg)["s3_region"])

    try:
        desc = ec2.describe_instances(InstanceIds=[instance_id])
        inst = desc["Reservations"][0]["Instances"][0]
        ec2_state = inst["State"]["Name"]
        public_ip = inst.get("PublicIpAddress", "none")
    except Exception as e:
        log.error("Instance lookup failed: %s", e)
        return

    hb = _get_heartbeat(s3, cfg, instance_id)
    age = _heartbeat_age(hb)
    phase = hb.get("phase", "?") if hb else "none"

    print(f"Instance:  {instance_id}")
    print(f"State:     {ec2_state}")
    print(f"IP:        {public_ip}")
    print(f"Type:      {state.get('instance_type', '?')}")
    print(f"Mode:      {state.get('mode', '?')}")
    print(f"Launched:  {state.get('launched_at', '?')}")
    print(f"Heartbeat: {phase} ({age:.0f}s ago)" if hb else "Heartbeat: none")
    key_file = state.get("key_file", "?")
    print(f"SSH:       ssh -i {key_file} ubuntu@{public_ip}")



    """Enqueue missing benchmark IDs to the branch's SQS queue.

    Queries the local harness manifest for missing benchmarks and sends each
    permutation ID as an SQS message. The CDK-managed ASG will scale up
    workers to process them.

    Requires the CDK bench stack to be deployed (provides the SQS queue).
    """
    cfg = load_config()
    aws_cfg = _aws(cfg)
    branch = cfg["repo"]["branch"]
    safe_branch = _sanitize_branch(branch)
    category = cfg.get("benchmark", {}).get("category") or ""
    limit = args.limit

    # Find the SQS queue URL from CloudFormation outputs
    cf = boto3.client("cloudformation", region_name=aws_cfg["ec2_region"])
    stack_name = f"MuninnBench-{safe_branch}"

    try:
        stack = cf.describe_stacks(StackName=stack_name)
        outputs = {o["OutputKey"]: o["OutputValue"] for o in stack["Stacks"][0].get("Outputs", [])}
        queue_url = outputs.get("QueueUrl")
        if not queue_url:
            log.error("Stack %s has no QueueUrl output. Is it deployed?", stack_name)
            sys.exit(1)
    except ClientError as e:
        if "does not exist" in str(e):
            log.error("Stack %s not found. Deploy it first with CDK.", stack_name)
            log.error(
                "  npx aws-cdk@latest deploy %s --app 'uv run --group cdk benchmarks/infra/cdk/app.py' "
                "-c branch=%s -c ami_id=%s -c account=%s",
                stack_name, branch, aws_cfg["ami_id"],
                os.environ.get("CDK_DEFAULT_ACCOUNT", "<account>"),
            )
            sys.exit(1)
        raise

    # Query the harness manifest for missing benchmarks
    manifest_cmd = [
        "uv", "run", "--no-sync", "-m", "benchmarks.harness",
        "manifest", "--commands", "--missing",
    ]
    if category:
        manifest_cmd.extend(["--category", category])
    if limit is not None:
        manifest_cmd.extend(["--limit", str(limit)])

    log.info("Querying manifest for missing benchmarks (category=%s, limit=%s)...", category or "all", limit or "all")

    result = subprocess.run(manifest_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log.error("Manifest query failed: %s", result.stderr)
        sys.exit(1)

    # Parse benchmark IDs from the manifest output
    # Each line: "uv run -m benchmarks.harness benchmark --id {permutation_id}"
    bench_ids = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.search(r"--id\s+(\S+)", line)
        if match:
            bench_ids.append(match.group(1))

    if not bench_ids:
        log.info("No missing benchmarks found. Nothing to enqueue.")
        return

    # Enqueue to SQS
    sqs = boto3.client("sqs", region_name=aws_cfg["ec2_region"])

    log.info("Enqueuing %d benchmark(s) to %s", len(bench_ids), queue_url)

    # SQS SendMessageBatch handles up to 10 per call
    for i in range(0, len(bench_ids), 10):
        batch = bench_ids[i : i + 10]
        entries = [
            {"Id": str(j), "MessageBody": bid}
            for j, bid in enumerate(batch)
        ]
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = resp.get("Failed", [])
        if failed:
            log.error("Failed to enqueue %d message(s): %s", len(failed), failed)
        else:
            for bid in batch:
                log.info("  enqueued: %s", bid)

    log.info("")
    log.info("Submitted %d benchmark(s) to SQS.", len(bench_ids))
    log.info("Queue: %s", queue_url)
    log.info("The ASG will scale up workers automatically.")



    """Quick status check — instance state + heartbeat."""
    cfg = load_config()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        log.info("No instance tracked.")
        return

    ec2 = boto3.client("ec2", region_name=_aws(cfg)["ec2_region"])
    s3 = boto3.client("s3", region_name=_aws(cfg)["s3_region"])

    try:
        desc = ec2.describe_instances(InstanceIds=[instance_id])
        inst = desc["Reservations"][0]["Instances"][0]
        ec2_state = inst["State"]["Name"]
        public_ip = inst.get("PublicIpAddress", "none")
    except Exception as e:
        log.error("Instance lookup failed: %s", e)
        return

    hb = _get_heartbeat(s3, cfg, instance_id)
    age = _heartbeat_age(hb)
    phase = hb.get("phase", "?") if hb else "none"

    print(f"Instance:  {instance_id}")
    print(f"State:     {ec2_state}")
    print(f"IP:        {public_ip}")
    print(f"Type:      {state.get('instance_type', '?')}")
    print(f"Mode:      {state.get('mode', '?')}")
    print(f"Launched:  {state.get('launched_at', '?')}")
    print(f"Heartbeat: {phase} ({age:.0f}s ago)" if hb else "Heartbeat: none")
    key_file = state.get("key_file", "?")
    print(f"SSH:       ssh -i {key_file} ubuntu@{public_ip}")


def cmd_submit(args: argparse.Namespace) -> None:
    """Enqueue missing benchmark IDs to the branch's SQS queue.

    Queries the local harness manifest for missing benchmarks and sends each
    permutation ID as an SQS message. The CDK-managed ASG will scale up
    workers to process them.
    """
    cfg = load_config()
    aws_cfg = _aws(cfg)
    branch = cfg["repo"]["branch"]
    safe_branch = _sanitize_branch(branch)
    category = cfg.get("benchmark", {}).get("category") or ""
    limit = args.limit

    # Find the SQS queue URL from CloudFormation outputs
    cf = boto3.client("cloudformation", region_name=aws_cfg["ec2_region"])
    stack_name = f"MuninnBench-{safe_branch}"

    try:
        stack = cf.describe_stacks(StackName=stack_name)
        outputs = {o["OutputKey"]: o["OutputValue"] for o in stack["Stacks"][0].get("Outputs", [])}
        queue_url = outputs.get("QueueUrl")
        if not queue_url:
            log.error("Stack %s has no QueueUrl output. Is it deployed?", stack_name)
            sys.exit(1)
    except ClientError as e:
        if "does not exist" in str(e):
            log.error("Stack %s not found. Deploy it first with CDK.", stack_name)
            sys.exit(1)
        raise

    # Upload the rendered worker script to S3 — the AMI's systemd service
    # downloads this on boot (bypasses cloud-init entirely)
    worker_template = PROJECT_ROOT / "benchmarks" / "infra" / "worker_user_data.sh"
    worker_script = worker_template.read_text(encoding="utf-8")
    worker_script = worker_script.replace("__S3_BUCKET__", aws_cfg["s3_bucket"])
    worker_script = worker_script.replace("__S3_REGION__", aws_cfg["s3_region"])
    worker_script = worker_script.replace("__SQS_QUEUE_URL__", queue_url)
    worker_script = worker_script.replace("__REPO_URL__", cfg["repo"]["url"])
    worker_script = worker_script.replace("__BRANCH__", branch)

    s3 = boto3.client("s3", region_name=aws_cfg["s3_region"])
    s3.put_object(
        Bucket=aws_cfg["s3_bucket"],
        Key="scripts/worker.sh",
        Body=worker_script.encode("utf-8"),
    )
    log.info("Uploaded worker script to s3://%s/scripts/worker.sh", aws_cfg["s3_bucket"])

    # Query the harness manifest for missing benchmarks
    manifest_cmd = [
        "uv", "run", "--no-sync", "-m", "benchmarks.harness",
        "manifest", "--commands", "--missing",
    ]
    if category:
        manifest_cmd.extend(["--category", category])
    if limit is not None:
        manifest_cmd.extend(["--limit", str(limit)])

    log.info("Querying manifest for missing benchmarks (category=%s, limit=%s)...", category or "all", limit or "all")

    result = subprocess.run(manifest_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log.error("Manifest query failed: %s", result.stderr)
        sys.exit(1)

    # Parse benchmark IDs from manifest output
    bench_ids = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.search(r"--id\s+(\S+)", line)
        if match:
            bench_ids.append(match.group(1))

    if not bench_ids:
        log.info("No missing benchmarks found. Nothing to enqueue.")
        return

    # Enqueue to SQS (batches of 10)
    sqs_client = boto3.client("sqs", region_name=aws_cfg["ec2_region"])

    log.info("Enqueuing %d benchmark(s) to %s", len(bench_ids), queue_url)

    for i in range(0, len(bench_ids), 10):
        batch = bench_ids[i : i + 10]
        entries = [{"Id": str(j), "MessageBody": bid} for j, bid in enumerate(batch)]
        resp = sqs_client.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = resp.get("Failed", [])
        if failed:
            log.error("Failed to enqueue %d message(s): %s", len(failed), failed)
        else:
            for bid in batch:
                log.info("  enqueued: %s", bid)

    log.info("")
    log.info("Submitted %d benchmark(s). ASG will scale up automatically.", len(bench_ids))


def cmd_teardown(args: argparse.Namespace) -> None:
    """Terminate instance and optionally clean up all AWS resources."""
    cfg = load_config()
    aws_cfg = _aws(cfg)
    state = _load_state()
    instance_id = state.get("instance_id")

    ec2 = boto3.client("ec2", region_name=aws_cfg["ec2_region"])

    if instance_id:
        log.info("Terminating %s...", instance_id)
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
        except ClientError:
            pass

        s3 = boto3.client("s3", region_name=aws_cfg["s3_region"])
        try:
            s3.delete_object(Bucket=aws_cfg["s3_bucket"], Key=f"heartbeat/{instance_id}.json")
        except Exception:
            pass

    _clear_state()

    if args.all:
        iam = boto3.client("iam")
        role_name = "muninn-bench-ec2-role"
        profile_name = "muninn-bench-ec2-profile"

        log.info("Cleaning up IAM...")
        _safe_call(iam.delete_role_policy, RoleName=role_name, PolicyName="s3-access")
        _safe_call(
            iam.detach_role_policy,
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        )
        _safe_call(iam.remove_role_from_instance_profile, InstanceProfileName=profile_name, RoleName=role_name)
        _safe_call(iam.delete_instance_profile, InstanceProfileName=profile_name)
        _safe_call(iam.delete_role, RoleName=role_name)

        log.info("Cleaning up EC2 resources...")
        if aws_cfg.get("key_name"):
            _safe_call(ec2.delete_key_pair, KeyName=aws_cfg["key_name"])
        if aws_cfg.get("security_group_id"):
            # Wait for instance termination before deleting SG
            if instance_id:
                log.info("Waiting for instance termination...")
                try:
                    ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
                except Exception:
                    pass
            _safe_call(ec2.delete_security_group, GroupId=aws_cfg["security_group_id"])

        log.info("All resources cleaned up.")

    log.info("Done.")


# ── Helpers ───────────────────────────────────────────────────────


def _safe_call(fn, **kwargs):
    """Call an AWS API, ignoring NotFound/NoSuchEntity errors."""
    try:
        fn(**kwargs)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchEntity", "DeleteConflict", "InvalidGroup.NotFound", "InvalidKeyPair.NotFound"):
            pass
        else:
            log.warning("  %s: %s", code, e.response["Error"]["Message"])


def _show_latest_phase_log(s3, cfg: dict) -> None:
    """Find and display the latest phase log from S3."""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        phases_files = []
        for page in paginator.paginate(Bucket=_aws(cfg)["s3_bucket"], Prefix="runs/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_phases.log"):
                    phases_files.append(obj)

        if not phases_files:
            return

        latest = max(phases_files, key=lambda o: o["LastModified"])
        resp = s3.get_object(Bucket=_aws(cfg)["s3_bucket"], Key=latest["Key"])
        content = resp["Body"].read().decode("utf-8")
        print(f"\n=== Phase log: {latest['Key']} ===")
        print(content)
    except Exception:
        pass


def _help(p: argparse.ArgumentParser):
    """Return a handler that prints help for parser p."""
    def _print_help(_: argparse.Namespace) -> None:
        p.print_help()
    return _print_help


# ── CLI ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="benchmarks/infra/runner.py",
        description="EC2 benchmark runner — launch, monitor, and manage remote benchmark instances",
    )
    parser.set_defaults(func=_help(parser))
    subs = parser.add_subparsers(dest="command", required=False)

    subs.add_parser("setup", help="Create AWS resources (IAM, key pair, security group)").set_defaults(func=cmd_setup)

    subs.add_parser("prime", help="Cold start an instance, create AMI for warm launches").set_defaults(func=cmd_prime)

    subs.add_parser("run", help="Launch instance, run benchmarks, monitor heartbeat").set_defaults(func=cmd_run)

    submit_p = subs.add_parser("submit", help="Enqueue missing benchmarks to SQS for parallel workers")
    submit_p.add_argument("--limit", type=int, default=None, help="Max benchmarks to enqueue (default: all missing)")
    submit_p.set_defaults(func=cmd_submit)

    subs.add_parser("status", help="Quick status check").set_defaults(func=cmd_status)

    teardown_p = subs.add_parser("teardown", help="Terminate instance and clean up")
    teardown_p.add_argument("--all", action="store_true", help="Also delete IAM, key pair, security group")
    teardown_p.set_defaults(func=cmd_teardown)

    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
