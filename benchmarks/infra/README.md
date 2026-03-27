# Benchmark Infrastructure

Run muninn benchmarks on EC2 instead of locally. Supports single-instance runs for quick POCs and scaled-out parallel runs via SQS + Auto Scaling Groups.

## Architecture

```
                                    ┌────────────────────────────┐
                                    │   runner.py (local CLI)    │
                                    │   prime / run / submit /   │
                                    │   monitor / status /       │
                                    │   teardown                 │
                                    └──────┬─────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
              ┌─────▼─────┐         ┌──────▼──────┐        ┌─────▼─────┐
              │ S3 Bucket │         │  SQS Queue  │        │    EC2    │
              │           │         │  (per branch)│        │ Instance  │
              │ /prep/    │◄────────│             │───────▶│ (spot or  │
              │ /heartbeat│  enqueue│  + DLQ      │  poll  │  on-demand)│
              │ /runs/    │         └─────────────┘        └───────────┘
              └───────────┘                                      │
                    ▲                                             │
                    │              ┌─────────────┐               │
                    └──────────────│  S3 Heartbeat│◄──────────────┘
                     results +     │  every 15s   │   PUT heartbeat
                     phase logs    └──────────────┘
```

## Two Execution Models

### 1. Single Instance (`runner.py run`)

Best for quick POCs, debugging, and running a handful of benchmarks.

- Launches one instance (spot with on-demand fallback)
- Instance picks the next missing benchmark via `manifest --commands --missing --limit 1`
- Monitors heartbeat every 15 seconds from the client side
- Auto-terminates hung instances (>180s stale heartbeat)
- Instance shuts down after completing; spot instances terminate, on-demand instances stop (EBS preserved)

```bash
# Setup AWS resources (IAM, key pair, security group)
uv run benchmarks/infra/runner.py setup

# Launch, run 1 benchmark, monitor until done
uv run benchmarks/infra/runner.py run

# Check status anytime
uv run benchmarks/infra/runner.py status

# Clean up
uv run benchmarks/infra/runner.py teardown       # instance only
uv run benchmarks/infra/runner.py teardown --all  # instance + IAM + SG + key
```

### 2. Parallel Workers (`cdk/` + SQS)

Best for running many benchmarks in parallel across multiple workers.

- CDK deploys per-branch infrastructure: SQS queue, DLQ, ASG, scaling policies
- Workers are launched from a pre-baked AMI (skip cold start)
- Each worker polls SQS for benchmark IDs, runs them, uploads results to S3
- ASG scales from 0 to N based on queue depth, scales back to zero when idle
- Spot interruption handled automatically: message reappears in SQS for retry
- Poison pill benchmarks (always fail) go to DLQ after 3 attempts

```bash
# Bootstrap CDK (once per account/region)
npx aws-cdk@latest bootstrap \
  --app "uv run --group cdk benchmarks/infra/cdk/app.py" \
  -c account=$(aws sts get-caller-identity --query Account --output text)

# Deploy per-branch stack
npx aws-cdk@latest deploy MuninnBench-feat-my-branch \
  --app "uv run --group cdk benchmarks/infra/cdk/app.py" \
  -c account=$(aws sts get-caller-identity --query Account --output text) \
  -c branch=feat/my-branch \
  -c ami_id=ami-xxx

# Enqueue benchmarks (TODO: runner.py submit)
# Workers scale up automatically, run benchmarks, scale to zero

# Tear down branch infrastructure
npx aws-cdk@latest destroy MuninnBench-feat-my-branch \
  --app "uv run --group cdk benchmarks/infra/cdk/app.py"
```

## Configuration

Copy `config.yml.example` to `config.yml` (gitignored) and fill in your values:

```yaml
aws:
  ec2_region: ap-southeast-2
  s3_bucket: my-benchmark-bucket
  s3_region: us-west-2
  instance_type: t3.xlarge
  use_spot: true
  ami_id: ami-095e8c26af3940dc2    # Ubuntu 24.04 in ap-southeast-2
  key_name: ""                      # populated by 'setup'
  security_group_id: ""             # populated by 'setup'
  instance_profile_name: ""         # populated by 'setup'

repo:
  url: https://github.com/neozenith/sqlite-muninn.git
  branch: feat/my-branch

benchmark:
  category: graph                   # or "" for all categories
  limit: 1                          # benchmarks per single-instance run

heartbeat:
  interval: 15
  stale_threshold: 60
  hung_threshold: 180
```

All values can be overridden with `BENCH_*` environment variables (e.g., `BENCH_EC2_REGION`, `BENCH_S3_BUCKET`, `BENCH_BRANCH`).

## File Reference

```
benchmarks/infra/
├── README.md                   # This file
├── config.yml.example          # Config template (tracked)
├── config.yml                  # Your config (gitignored)
├── runner.py                   # Client-side CLI: setup, run, monitor, status, teardown
├── user_data.sh                # EC2 bootstrap: manifest-based (single instance mode)
├── worker_user_data.sh         # EC2 bootstrap: SQS-based (parallel worker mode)
└── cdk/
    ├── cdk.json                # CDK app config
    ├── app.py                  # CDK entry point — creates stacks from context vars
    ├── bench_stack.py          # Per-branch: SQS + DLQ + ASG + step scaling + IAM
    ├── cleanup_stack.py        # Global: Lambda + EventBridge for AMI pruning
    └── lambda_fn/
        └── ami_cleanup.py      # Prunes AMIs tagged project=muninn-benchmarks >7 days
```

## Observability

### S3 Heartbeat (primary health signal)

Every 15 seconds the instance writes to `s3://{bucket}/heartbeat/{instance_id}.json`:

```json
{
  "timestamp": "2026-03-27T07:15:30Z",
  "phase": "03_build",
  "instance": "i-0abc123",
  "run_id": "20260327_071200"
}
```

The client-side monitor (`runner.py run` or `runner.py monitor`) polls this every 15 seconds. If the heartbeat is stale for >60s, it warns. If stale for >180s, it auto-terminates the instance.

### SSH

```bash
ssh -i tmp/ec2_runs/keys/muninn-bench-key-ap-southeast-2.pem ubuntu@<ip>
tail -f /var/log/muninn/benchmark.log
```

### SSM (no SSH key needed)

```bash
aws ssm start-session --target i-0abc123 --region ap-southeast-2
```

### Phase Logs (S3)

After each run, timing logs are uploaded to `s3://{bucket}/runs/{run_id}_phases.log`:

```
01_deps: 41s
02_source: 34s
03_build: 209s
04_python_deps: 270s
bench_graph_muninn_bfs_erdos_renyi_n50000_deg5: 197s
05_benchmarks_total: 510s
06_total: 1064s
```

Full console output is at `s3://{bucket}/runs/{run_id}_full.log`.

### Local Timing Logs

Each `runner.py run` invocation saves a JSON timing log to `tmp/ec2_runs/{timestamp}.json` with launch time, first heartbeat, phase transitions, and outcome.

## AMI Lifecycle

### Creating a Primed AMI

A primed AMI captures: OS packages, build tools, ccache, git repo, compiled muninn.so, Python venv with all 215 benchmark dependencies. Workers launched from this AMI skip the 10-minute cold start.

```bash
# 1. Launch and cold-start an instance (via runner.py or manually)
uv run benchmarks/infra/runner.py run
# Wait for it to complete, then stop it (don't terminate)

# 2. Create AMI from the stopped instance
aws ec2 create-image \
  --instance-id i-0abc123 \
  --name "muninn-bench-$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)" \
  --no-reboot \
  --region ap-southeast-2 \
  --tag-specifications 'ResourceType=image,Tags=[{Key=project,Value=muninn-benchmarks},{Key=branch,Value=feat/my-branch}]'
```

### What to bake in vs inject at boot

| Bake into AMI (slow to create, rarely changes) | Inject via user-data (changes per run) |
|------------------------------------------------|---------------------------------------|
| OS packages (gcc, cmake, ccache, jq) | Git branch name |
| AWS CLI, SSM agent | S3 bucket name |
| uv binary | SQS queue URL |
| Git repo clone with submodules | Benchmark category/limit |
| Compiled llama.cpp + muninn.so | |
| Python .venv (215 packages) | |
| ccache (populated from cold build) | |
| 4GB swap file | |

On warm start from AMI, the user-data does `git pull` + `make` (ccache hit) + `uv sync` (no-op) — about 30 seconds instead of 10 minutes.

### Automatic AMI Pruning

The `MuninnCleanup` CDK stack deploys a Lambda that runs weekly (Sunday 00:00 UTC) and prunes AMIs tagged `project=muninn-benchmarks` that are older than 7 days. It also deletes the backing EBS snapshots.

Invoke manually anytime:

```bash
aws lambda invoke --function-name MuninnAmiCleanup /dev/stdout --region ap-southeast-2
```

## Cost

### Single Instance Mode

| Phase | Time | Cost (t3.xlarge spot) |
|-------|------|-----------------------|
| Cold start (first run) | ~10 min | ~$0.035 |
| Warm start (from AMI) | ~30s setup + benchmark time | varies |
| Benchmark (N=50000 graph) | ~3-5 min each | ~$0.015 each |

### Parallel Mode (CDK)

| Resource | At rest (scale-to-zero) | Active (N workers) |
|----------|------------------------|-------------------|
| SQS queue | $0 | $0 (pennies per million requests) |
| ASG (0 instances) | $0 | N * instance cost |
| AMI snapshot (~20GB) | ~$1/month | ~$1/month |
| CW alarm | $0.10/month | $0.10/month |
| Lambda (weekly prune) | $0 | $0 |
| **Total at rest** | **~$1.10/month per branch** | |

### Instance Sizing

| Type | vCPU | RAM | Cost/hr (spot) | Notes |
|------|------|-----|---------------|-------|
| t3.xlarge | 4 | 16 GB | ~$0.067 | Proven. Needs CMAKE_BUILD_PARALLEL_LEVEL=1 + 4GB swap |
| t3.large | 2 | 8 GB | ~$0.033 | OOMs on llama.cpp unicode-data.cpp even with swap |
| t3.medium | 2 | 4 GB | ~$0.017 | OOMs during parallel cmake builds |

## Learnings

Documented in detail at [neozenith/sqlite-muninn#7](https://github.com/neozenith/sqlite-muninn/issues/7).

Key takeaways:
- **EC2 console output has 5+ minute lag** — useless for real-time monitoring. Use S3 heartbeat instead.
- **CloudWatch Logs agent doesn't work in cloud-init boothook context** — the `exec > >(tee ...)` redirect breaks. S3 heartbeat is the only reliable signal.
- **`#cloud-boothook` runs before networking** — cannot download from S3 or access any network resource. Use SSM `send-command` for warm restart execution.
- **Ubuntu 24.04 has ccache in default repos**, Amazon Linux 2023 does not.
- **`uv sync` vs `uv sync --group benchmark`** — the benchmark dependencies (including `llama-cpp-python`) are in a separate dependency group. Without `--group benchmark`, the harness fails to import.
- **`CMAKE_ARGS="-DGGML_NATIVE=OFF"`** is required when building `llama-cpp-python` from source on EC2.
- **Spot instances terminate on interruption** (EBS destroyed). For warm cache via EBS, use on-demand with stop behavior. For cheap disposable runs, use spot.
- **AMI is the canonical warm cache pattern** — bake once, launch many. Combine with `git pull` + ccache for code updates without rebuilding the AMI.
