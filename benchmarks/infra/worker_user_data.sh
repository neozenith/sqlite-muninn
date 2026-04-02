#!/bin/bash
set -euo pipefail

# ── Injected by CDK launch template (replaced at synth time) ─────
S3_BUCKET="__S3_BUCKET__"
S3_REGION="__S3_REGION__"
SQS_QUEUE_URL="__SQS_QUEUE_URL__"
REPO="__REPO_URL__"
BRANCH="__BRANCH__"

# ── Derived ───────────────────────────────────────────────────────
WORK_DIR="/home/ubuntu/muninn"
LOG_DIR="/var/log/muninn"
LIVE_LOG="${LOG_DIR}/benchmark.log"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
PHASE_LOG="/tmp/phases_${RUN_ID}.log"
CURRENT_PHASE_FILE="/tmp/muninn_current_phase"

# ── IMDSv2 ────────────────────────────────────────────────────────
IMDS_TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
_imds() {
    curl -sf -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}" \
        "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || echo "unknown"
}
INSTANCE_ID=$(_imds instance-id)
INSTANCE_REGION=$(_imds placement/region)

# ── Logging ───────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LIVE_LOG") 2>&1

# ── Heartbeat (S3 PUT every 15s) ─────────────────────────────────
echo "starting" > "$CURRENT_PHASE_FILE"

_heartbeat_loop() {
    local hb_path="s3://${S3_BUCKET}/heartbeat/${INSTANCE_ID}.json"
    while true; do
        local phase ts payload
        phase=$(cat "$CURRENT_PHASE_FILE" 2>/dev/null || echo "unknown")
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        payload="{\"timestamp\":\"${ts}\",\"phase\":\"${phase}\",\"instance\":\"${INSTANCE_ID}\",\"run_id\":\"${RUN_ID}\"}"
        aws s3 cp --region "$S3_REGION" --no-cli-pager - "$hb_path" <<< "$payload" \
            >/dev/null 2>&1 || true
        sleep 15
    done
}

_heartbeat_loop &
HEARTBEAT_PID=$!

# ── Cleanup on exit ───────────────────────────────────────────────
_cleanup() {
    echo "stopped" > "$CURRENT_PHASE_FILE"
    sleep 2
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    aws s3 cp "$PHASE_LOG" "s3://${S3_BUCKET}/runs/${RUN_ID}_phases.log" \
        --region "$S3_REGION" --no-cli-pager 2>/dev/null || true
    aws s3 cp "$LIVE_LOG" "s3://${S3_BUCKET}/runs/${RUN_ID}_full.log" \
        --region "$S3_REGION" --no-cli-pager 2>/dev/null || true
    aws s3 rm "s3://${S3_BUCKET}/heartbeat/${INSTANCE_ID}.json" \
        --region "$S3_REGION" --no-cli-pager 2>/dev/null || true
}
trap _cleanup EXIT

# ── Helpers ───────────────────────────────────────────────────────
TOTAL_START=$SECONDS

log_phase() {
    local elapsed=$(( SECONDS - $2 ))
    echo "$1: ${elapsed}s" | tee -a "$PHASE_LOG"
}

set_phase() {
    echo "$1" > "$CURRENT_PHASE_FILE"
    echo ">>> $1"
}

# ── Banner ────────────────────────────────────────────────────────
echo "=== Muninn Benchmark Worker ==="
echo "Run ID:   ${RUN_ID}"
echo "Branch:   ${BRANCH}"
echo "Instance: ${INSTANCE_ID}"
echo "Type:     $(_imds instance-type)"
echo "Region:   ${INSTANCE_REGION}"
echo "Queue:    ${SQS_QUEUE_URL}"
echo ""

# ── Phase 1: Dependencies (idempotent) ────────────────────────────
set_phase "01_deps"
PHASE_START=$SECONDS

if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile
    swapon /swapfile
else
    swapon /swapfile 2>/dev/null || true
fi

if ! command -v cmake &>/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential cmake libsqlite3-dev git ccache unzip jq

    if ! command -v aws &>/dev/null; then
        curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
        unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscliv2.zip
    fi

    snap install amazon-ssm-agent --classic 2>/dev/null || true
    systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
    systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
else
    echo "  Build tools already installed (AMI warm start)"
    systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
fi

export CCACHE_DIR="/home/ubuntu/.ccache"
mkdir -p "$CCACHE_DIR"
ccache --max-size=2G
export PATH="/usr/lib/ccache:/root/.local/bin:/home/ubuntu/.local/bin:/usr/local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

log_phase "01_deps" "$PHASE_START"

# ── Phase 2: Source ───────────────────────────────────────────────
set_phase "02_source"
PHASE_START=$SECONDS

if [ -d "$WORK_DIR/.git" ]; then
    echo "  Pulling latest..."
    git -C "$WORK_DIR" fetch origin
    git -C "$WORK_DIR" checkout "$BRANCH"
    git -C "$WORK_DIR" reset --hard "origin/${BRANCH}"
    git -C "$WORK_DIR" submodule update --init --recursive
else
    echo "  Cloning..."
    git clone --recursive --branch "$BRANCH" "$REPO" "$WORK_DIR"
fi
echo "  HEAD: $(git -C "$WORK_DIR" log --oneline -1)"

log_phase "02_source" "$PHASE_START"

# ── Phase 3: Build ───────────────────────────────────────────────
set_phase "03_build"
PHASE_START=$SECONDS

export CC="ccache gcc"
export CXX="ccache g++"
export CMAKE_BUILD_PARALLEL_LEVEL=1

ccache --zero-stats >/dev/null
make -C "$WORK_DIR" all
echo "  ccache: $(ccache --show-stats 2>/dev/null | grep -E 'Hitrate' | head -1 || echo 'n/a')"

log_phase "03_build" "$PHASE_START"

# ── Phase 4: Python deps ─────────────────────────────────────────
set_phase "04_python_deps"
PHASE_START=$SECONDS

CMAKE_ARGS="-DGGML_NATIVE=OFF" uv sync --group benchmark --directory "$WORK_DIR"

log_phase "04_python_deps" "$PHASE_START"

# ── Phase 4b: Sync prep data from S3 (vectors, texts, models) ────
set_phase "04b_sync_prep"
PHASE_START=$SECONDS

echo "  Syncing prep data from S3..."
aws s3 sync "s3://${S3_BUCKET}/prep/benchmarks/vectors/" "${WORK_DIR}/benchmarks/vectors/" \
    --region "$S3_REGION" --no-cli-pager 2>/dev/null || echo "  WARN: vector sync failed"
aws s3 sync "s3://${S3_BUCKET}/prep/benchmarks/texts/" "${WORK_DIR}/benchmarks/texts/" \
    --region "$S3_REGION" --no-cli-pager 2>/dev/null || echo "  WARN: texts sync failed"
aws s3 sync "s3://${S3_BUCKET}/prep/models/" "${WORK_DIR}/models/" \
    --region "$S3_REGION" --no-cli-pager 2>/dev/null || echo "  WARN: models sync failed"

# Count available vector caches
VECTOR_COUNT=$(ls "${WORK_DIR}/benchmarks/vectors/"*.npy 2>/dev/null | wc -l | tr -d ' ')
echo "  Vector caches available: ${VECTOR_COUNT}"

log_phase "04b_sync_prep" "$PHASE_START"

# ── Phase 5: Poll SQS and run benchmarks ─────────────────────────
set_phase "05_sqs_poll"
PHASE_START=$SECONDS

BENCHMARKS_RUN=0
BENCHMARKS_FAILED=0
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=3

while true; do
    # Long-poll SQS (up to 20s wait)
    MSG_JSON=$(aws sqs receive-message \
        --queue-url "$SQS_QUEUE_URL" \
        --max-number-of-messages 1 \
        --wait-time-seconds 20 \
        --region "$INSTANCE_REGION" \
        --no-cli-pager 2>/dev/null || echo '{}')

    BENCH_ID=$(echo "$MSG_JSON" | jq -r '.Messages[0].Body // empty')
    RECEIPT=$(echo "$MSG_JSON" | jq -r '.Messages[0].ReceiptHandle // empty')

    if [ -z "$BENCH_ID" ]; then
        echo "  Queue empty. No more work."
        break
    fi

    echo "$BENCH_ID" > "$CURRENT_PHASE_FILE"
    BENCH_START=$SECONDS
    echo "  >>> Running: $BENCH_ID"

    # Run the benchmark — harness auto-uploads JSONL to S3 via --s3-bucket
    if uv run --directory "$WORK_DIR" --no-sync \
        -m benchmarks.harness --s3-bucket "$S3_BUCKET" \
        benchmark --id "$BENCH_ID" --force; then
        echo "  SUCCESS: $BENCH_ID"
        CONSECUTIVE_FAILURES=0
    else
        EXIT_CODE=$?
        echo "  FAILED: $BENCH_ID (exit code $EXIT_CODE)"
        BENCHMARKS_FAILED=$((BENCHMARKS_FAILED + 1))
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    fi

    BENCH_ELAPSED=$(( SECONDS - BENCH_START ))
    echo "  bench_${BENCH_ID}: ${BENCH_ELAPSED}s" | tee -a "$PHASE_LOG"
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))

    # Delete message from queue (success or failure — prevents poison pill loop;
    # failed benchmarks go to DLQ after maxReceiveCount retries)
    aws sqs delete-message \
        --queue-url "$SQS_QUEUE_URL" \
        --receipt-handle "$RECEIPT" \
        --region "$INSTANCE_REGION" \
        --no-cli-pager 2>/dev/null || true

    # Circuit breaker: stop if too many consecutive failures
    # Prevents burning through the entire queue when there's a systemic issue
    # (e.g., missing prep data, broken extension, import errors)
    if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
        echo "  CIRCUIT BREAKER: $CONSECUTIVE_FAILURES consecutive failures. Stopping."
        echo "  Total: $BENCHMARKS_RUN run, $BENCHMARKS_FAILED failed"
        break
    fi
done

log_phase "05_benchmarks_total (${BENCHMARKS_RUN} run, ${BENCHMARKS_FAILED} failed)" "$PHASE_START"

# ── Summary ───────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo "06_total: ${TOTAL_ELAPSED}s" | tee -a "$PHASE_LOG"

echo ""
echo "=== PHASE SUMMARY ==="
cat "$PHASE_LOG"

# ── Clean cloud-init so next boot from this AMI re-runs user-data ─
cloud-init clean --logs 2>/dev/null || true
rm -f /var/lib/cloud/instance/sem/config_scripts_user 2>/dev/null || true

# ── Shutdown ──────────────────────────────────────────────────────
set_phase "shutdown"
echo ">>> Shutting down (ran $BENCHMARKS_RUN benchmark(s))..."
shutdown -h now
