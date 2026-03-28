#!/bin/bash
set -euo pipefail

# ── Injected by runner.py (replaced at launch time) ──────────────
S3_BUCKET="__S3_BUCKET__"
S3_REGION="__S3_REGION__"
REPO="__REPO_URL__"
BRANCH="__BRANCH__"
BENCH_CATEGORY="__BENCH_CATEGORY__"
BENCH_LIMIT="__BENCH_LIMIT__"

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

# ── Cleanup on exit (success or failure) ──────────────────────────
_cleanup() {
    echo "stopped" > "$CURRENT_PHASE_FILE"
    sleep 2  # let one final heartbeat fire with "stopped"
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
echo "=== Muninn Benchmark Runner ==="
echo "Run ID:   ${RUN_ID}"
echo "Branch:   ${BRANCH}"
echo "Instance: ${INSTANCE_ID}"
echo "Type:     $(_imds instance-type)"
echo "Region:   $(_imds placement/region)"
echo "Category: ${BENCH_CATEGORY:-all}"
echo "Limit:    ${BENCH_LIMIT}"
echo ""

# ── Phase 1: Dependencies (idempotent) ────────────────────────────
set_phase "01_deps"
PHASE_START=$SECONDS

# Swap (idempotent)
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "  4GB swap created"
else
    swapon /swapfile 2>/dev/null || true
fi

if ! command -v cmake &>/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential cmake libsqlite3-dev git ccache unzip

    # AWS CLI
    if ! command -v aws &>/dev/null; then
        curl -sf "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
        unzip -q /tmp/awscliv2.zip -d /tmp
        /tmp/aws/install
        rm -rf /tmp/aws /tmp/awscliv2.zip
    fi

    # SSM agent
    snap install amazon-ssm-agent --classic 2>/dev/null || true
    systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
    systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
else
    echo "  Build tools already installed (warm restart)"
    systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
fi

# ccache (EBS-persistent)
export CCACHE_DIR="/home/ubuntu/.ccache"
mkdir -p "$CCACHE_DIR"
ccache --max-size=2G
export PATH="/usr/lib/ccache:/root/.local/bin:/home/ubuntu/.local/bin:/usr/local/bin:$PATH"

# uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

log_phase "01_deps" "$PHASE_START"

# ── Phase 2: Source ───────────────────────────────────────────────
set_phase "02_source"
PHASE_START=$SECONDS

if [ -d "$WORK_DIR/.git" ]; then
    echo "  Pulling latest (warm restart)..."
    git -C "$WORK_DIR" fetch origin
    git -C "$WORK_DIR" checkout "$BRANCH"
    git -C "$WORK_DIR" reset --hard "origin/${BRANCH}"
    git -C "$WORK_DIR" submodule update --init --recursive
else
    echo "  Cloning (cold start)..."
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

# ── Phase 5: Pick and run benchmarks ─────────────────────────────
set_phase "05_benchmarks"
PHASE_START=$SECONDS

# If limit is 0, skip benchmarks entirely (prime-only mode)
if [ "$BENCH_LIMIT" = "0" ]; then
    echo "  Prime-only mode (limit=0). Skipping benchmarks."
    BENCH_CMDS=""
else
    MANIFEST_CMD="uv run --directory $WORK_DIR --no-sync -m benchmarks.harness"
    MANIFEST_CMD="$MANIFEST_CMD --s3-bucket $S3_BUCKET"
    MANIFEST_CMD="$MANIFEST_CMD manifest --commands --missing --limit $BENCH_LIMIT"
    if [ -n "$BENCH_CATEGORY" ]; then
        MANIFEST_CMD="$MANIFEST_CMD --category $BENCH_CATEGORY"
    fi

    echo "  Querying manifest: $MANIFEST_CMD"
    BENCH_CMDS=$(eval "$MANIFEST_CMD" 2>/dev/null || true)
fi

if [ -z "$BENCH_CMDS" ]; then
    echo "  No missing benchmarks found. Nothing to do."
    log_phase "05_benchmarks_total" "$PHASE_START"
else
    BENCH_COUNT=$(echo "$BENCH_CMDS" | wc -l | tr -d ' ')
    echo "  Found $BENCH_COUNT benchmark(s) to run"

    echo "$BENCH_CMDS" | while IFS= read -r cmd; do
        # Extract the permutation ID from the command
        BENCH_ID=$(echo "$cmd" | grep -oE '\-\-id [^ ]+' | awk '{print $2}')
        if [ -z "$BENCH_ID" ]; then
            echo "  WARN: Could not parse benchmark ID from: $cmd"
            continue
        fi

        echo "$BENCH_ID" > "$CURRENT_PHASE_FILE"
        BENCH_START=$SECONDS
        echo "  >>> Running: $BENCH_ID"

        uv run --directory "$WORK_DIR" --no-sync \
            -m benchmarks.harness --s3-bucket "$S3_BUCKET" \
            benchmark --id "$BENCH_ID" --force

        BENCH_ELAPSED=$(( SECONDS - BENCH_START ))
        echo "  bench_${BENCH_ID}: ${BENCH_ELAPSED}s" | tee -a "$PHASE_LOG"
    done

    log_phase "05_benchmarks_total" "$PHASE_START"
fi

# ── Summary ───────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo "06_total: ${TOTAL_ELAPSED}s" | tee -a "$PHASE_LOG"

echo ""
echo "=== PHASE SUMMARY ==="
cat "$PHASE_LOG"

# ── Clean cloud-init state so next boot re-runs user-data ─────────
# Without this, AMIs created from this instance would skip user-data
# on launch (cloud-init considers it "already ran").
cloud-init clean --logs 2>/dev/null || true
rm -f /var/lib/cloud/instance/sem/config_scripts_user 2>/dev/null || true

# ── Shutdown ──────────────────────────────────────────────────────
set_phase "shutdown"
echo ">>> Shutting down..."
shutdown -h now
