# CLAUDE.md -- Benchmark Infrastructure

Guidance for Claude Code when working within `benchmarks/infra/`.

## What This Is

Config-driven tooling for running muninn benchmarks on AWS EC2. Two execution models: single-instance via `runner.py` and parallel workers via CDK (SQS + ASG).

## Commands

```bash
# Single instance mode
uv run benchmarks/infra/runner.py setup       # create IAM, key pair, security group
uv run benchmarks/infra/runner.py prime       # cold start + create AMI
uv run benchmarks/infra/runner.py run         # launch spot, run 1 benchmark, monitor
uv run benchmarks/infra/runner.py status      # check instance + heartbeat
uv run benchmarks/infra/runner.py teardown    # terminate instance
uv run benchmarks/infra/runner.py teardown --all  # terminate + delete all resources

# Parallel mode (CDK)
uv run benchmarks/infra/runner.py submit --limit 10  # enqueue to SQS
npx aws-cdk@latest deploy MuninnBench-<branch> --app "uv run --group cdk benchmarks/infra/cdk/app.py" -c account=<id> -c branch=<branch> -c ami_id=<ami>
npx aws-cdk@latest destroy MuninnBench-<branch> --app "uv run --group cdk benchmarks/infra/cdk/app.py"

# Dashboard
uv run benchmarks/infra/dashboard.py          # http://localhost:8050
```

## Architecture

- **runner.py**: client-side CLI. Manages EC2 lifecycle, monitors S3 heartbeat, uploads worker script to S3, enqueues benchmarks to SQS.
- **user_data.sh**: manifest-based bootstrap for single instances. Also installs a systemd service (`muninn-worker.service`) that downloads and executes `worker.sh` from S3 on every boot.
- **worker_user_data.sh**: SQS-based worker script. Polls SQS for benchmark IDs, runs each via the harness CLI, deletes messages on completion.
- **cdk/**: CDK app defining per-branch infrastructure (SQS + DLQ + ASG + scaling) and a shared AMI cleanup Lambda.
- **dashboard.py**: Plotly Dash live monitoring dashboard backed by CloudWatch metrics and ASG scaling activities.
- **config.yml**: gitignored configuration (copy from `config.yml.example`).

## Gotchas

### Cloud-init only runs user-data on first boot

AMIs created from a primed instance have cloud-init's "already ran" state baked in. Subsequent launches from the AMI skip user-data entirely.

**Fix**: the prime step installs a systemd oneshot service (`muninn-worker.service`) that runs on every boot after `network-online.target`. It downloads `scripts/worker.sh` from S3 — no cloud-init involvement. The `submit` command uploads the rendered worker script to S3 before enqueuing benchmarks.

Do NOT attempt to fix this with:
- `cloud-init clean` (races with shutdown, unreliable)
- `#cloud-boothook` (runs before networking, can't access S3)
- MIME multipart user-data (still subject to cloud-init caching)

### Spot instances get SIGKILL on termination

AWS terminates spot instances with SIGKILL. Bash `trap` handlers (EXIT) do NOT fire. Phase logs uploaded in the EXIT trap are lost on spot termination.

**Why results survive**: the harness's `sync_to_s3()` in `run_treatment()` uploads each JSONL result immediately after the benchmark completes — not at shutdown. Benchmark data is safe even if the instance is killed mid-way.

**What's lost**: phase timing logs and heartbeat cleanup. For timing data from spot workers, check CloudWatch metrics or the S3 heartbeat history.

### SQS scaling: visible + in-flight, not visible alone

The ASG scale-in policy must consider both `ApproximateNumberOfMessagesVisible` AND `ApproximateNumberOfMessagesNotVisible`. When a worker pulls a message, it becomes invisible. A naive alarm on visible-only scales in while workers are actively processing.

The CDK stack uses a CloudWatch math expression `visible + inflight` with 10-period evaluation before scale-in.

### SQS CloudWatch metrics lag on idle queues

After a queue is idle for hours, CloudWatch metrics for that queue stop being emitted. When new messages arrive, it takes up to 15 minutes for CloudWatch to start reporting again. The ASG alarm can't fire until CloudWatch reports the metric.

**Workaround**: `runner.py submit` could manually set `desired_capacity=1` on the ASG after enqueueing, but currently relies on the CW alarm. For urgent runs, manually scale with `aws autoscaling set-desired-capacity`.

### llama.cpp build requires t3.xlarge (16GB)

The `unicode-data.cpp` compilation in llama.cpp requires ~6GB RAM for a single `cc1plus` process. t3.medium (4GB) and t3.large (8GB) both OOM even with swap. Use t3.xlarge minimum.

Set `CMAKE_BUILD_PARALLEL_LEVEL=1` to prevent parallel cmake builds from doubling memory usage.

### Python benchmark group

`llama-cpp-python` and other ML dependencies are in the `benchmark` dependency group, not the main dependencies. Always use `uv sync --group benchmark` on EC2 instances. Building `llama-cpp-python` from source requires `CMAKE_ARGS="-DGGML_NATIVE=OFF"`.

### Ubuntu 24.04 is the required AMI

Amazon Linux 2023 does not have `ccache` in its repos. Ubuntu 24.04 (`ami-095e8c26af3940dc2` in ap-southeast-2) is the tested base AMI.

### S3 bucket is cross-region

The S3 bucket is in `us-west-2` but EC2 instances run in `ap-southeast-2`. All S3 commands must specify `--region us-west-2`. The worker script and heartbeat use `S3_REGION` variable for this.

### Category exclusion

KG benchmark categories (`kg-extract`, `kg-re`, `kg-resolve`, `kg-graphrag`) are excluded from the registry by default because their prep pipeline has not been validated for cloud execution. Override with `BENCH_EXCLUDE_CATEGORIES=""` to include all.

## Config

All values in `config.yml` can be overridden with `BENCH_*` environment variables:

| Config key | Env var | Notes |
|-----------|---------|-------|
| `aws.ec2_region` | `BENCH_EC2_REGION` | Instance launch region |
| `aws.s3_bucket` | `BENCH_S3_BUCKET` | Results + heartbeat storage |
| `aws.s3_region` | `BENCH_S3_REGION` | S3 bucket region (may differ from EC2) |
| `aws.instance_type` | `BENCH_INSTANCE_TYPE` | t3.xlarge minimum for builds |
| `aws.use_spot` | `BENCH_USE_SPOT` | true/false, falls back to on-demand |
| `aws.ami_id` | `BENCH_AMI_ID` | Updated by `prime` command |
| `benchmark.category` | `BENCH_CATEGORY` | Filter for `run` and `submit` |
| `benchmark.limit` | `BENCH_LIMIT` | Max benchmarks per `run` invocation |

## Testing

The runner.py, dashboard.py, and CDK stacks do not have unit tests. They are integration-tested against live AWS resources. Always test with `--limit 1` or `--limit 2` before scaling up.

## Cost

| Resource | At rest | Active |
|----------|---------|--------|
| AMI snapshot (~20GB) | ~$1/month | ~$1/month |
| SQS queue | $0 | ~$0 |
| ASG (0 instances) | $0 | N * instance $/hr |
| t3.xlarge spot | - | ~$0.067/hr |
| S3 (results) | ~$0.02/GB/month | ~$0.02/GB/month |
