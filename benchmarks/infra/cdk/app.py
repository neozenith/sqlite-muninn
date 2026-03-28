#!/usr/bin/env python3
"""CDK app for muninn benchmark infrastructure.

Deploys per-branch benchmark stacks (SQS + ASG + scaling) and a shared
AMI cleanup stack (Lambda + EventBridge weekly schedule).

Usage:
    # Bootstrap (once per account/region)
    npx cdk bootstrap --app "uv run --group cdk benchmarks/infra/cdk/app.py"

    # Deploy benchmark stack for a branch
    npx cdk deploy MuninnBench-feat-foo \
        --app "uv run --group cdk benchmarks/infra/cdk/app.py" \
        -c branch=feat/foo \
        -c ami_id=ami-xxx

    # Deploy the cleanup stack (once)
    npx cdk deploy MuninnCleanup \
        --app "uv run --group cdk benchmarks/infra/cdk/app.py"

    # Tear down a branch
    npx cdk destroy MuninnBench-feat-foo \
        --app "uv run --group cdk benchmarks/infra/cdk/app.py"
"""

import os
import re
import sys
from pathlib import Path

# Ensure the cdk/ directory is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aws_cdk as cdk

from bench_stack import BenchStack
from cleanup_stack import CleanupStack

app = cdk.App()

# ── Shared config from context ────────────────────────────────────
s3_bucket = app.node.try_get_context("s3_bucket") or "muninn-benchmarks-389956346255"
s3_region = app.node.try_get_context("s3_region") or "us-west-2"
ec2_region = app.node.try_get_context("ec2_region") or "ap-southeast-2"

account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")
env = cdk.Environment(account=account, region=ec2_region)

# ── Cleanup stack (global, not per-branch) ────────────────────────
CleanupStack(app, "MuninnCleanup", env=env)

# ── Benchmark stack (per-branch, only if branch + ami_id provided) ─
branch = app.node.try_get_context("branch")
ami_id = app.node.try_get_context("ami_id")

if branch and ami_id:
    # Sanitize branch name for CloudFormation: [a-zA-Z0-9-] only
    safe_branch = re.sub(r"[^a-zA-Z0-9]", "-", branch).strip("-")[:64]
    stack_name = f"MuninnBench-{safe_branch}"
    max_workers = int(app.node.try_get_context("max_workers") or "5")

    BenchStack(
        app,
        stack_name,
        branch=branch,
        ami_id=ami_id,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        max_workers=max_workers,
        env=env,
    )

app.synth()
