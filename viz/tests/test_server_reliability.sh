#!/usr/bin/env bash
# Server reliability test — start/stop/query cycle.
# Usage: bash viz/tests/test_server_reliability.sh [ITERATIONS]
#
# Verifies that the viz server can reliably:
#   1. Start without port conflicts
#   2. Respond to health checks
#   3. Execute a KG search query
#   4. Shut down cleanly
#
# Repeats ITERATIONS times (default 20). Any failure resets the counter.

set -euo pipefail

ITERATIONS="${1:-20}"
PORT=8202  # Dedicated port for reliability testing
HEALTH_URL="http://localhost:${PORT}/api/health"
KG_URL="http://localhost:${PORT}/api/kg/query"
VIZ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "${VIZ_DIR}/.." && pwd)"

consecutive_passes=0
total_attempts=0

# Kill ALL processes on the test port — finds the actual socket holder, not just uv wrapper.
kill_port() {
    local pids
    pids=$(lsof -ti:"${PORT}" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        return 0
    fi

    # SIGTERM first — allows graceful socket close (avoids TIME_WAIT)
    echo "$pids" | xargs kill 2>/dev/null || true

    # Wait up to 3s for graceful shutdown
    local waited=0
    while [ $waited -lt 6 ]; do
        pids=$(lsof -ti:"${PORT}" 2>/dev/null || true)
        if [ -z "$pids" ]; then
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    # Still alive? SIGKILL as last resort
    pids=$(lsof -ti:"${PORT}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Force-killing stubborn PIDs: ${pids}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Wait for port to be completely free (no TIME_WAIT, no LISTEN)
wait_port_free() {
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if ! lsof -ti:"${PORT}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "  Port ${PORT} still in use after ${max_wait}s"
    return 1
}

wait_for_health() {
    local max_wait=30  # Allow time for model loading on first run
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    echo "  FAIL: Health check did not pass within $((max_wait / 2))s"
    return 1
}

run_kg_query() {
    local response http_code
    # Capture both body and HTTP status code
    response=$(curl -s -w "\n%{http_code}" -X POST "${KG_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "division of labor", "k": 5}' \
        --max-time 30 2>&1) || {
        echo "  FAIL: KG query curl error: ${response}"
        return 1
    }
    http_code=$(echo "$response" | tail -1)
    response=$(echo "$response" | sed '$d')

    if [ "$http_code" != "200" ]; then
        echo "  FAIL: KG query HTTP ${http_code}: ${response:0:200}"
        return 1
    fi

    # Verify response has expected fields
    if ! echo "$response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
assert 'query' in r, 'missing query field'
assert 'fts_results' in r, 'missing fts_results'
assert 'vss_results' in r, 'missing vss_results'
assert 'graph_nodes' in r, 'missing graph_nodes'
assert 'graph_edges' in r, 'missing graph_edges'
assert 'node_community' in r, 'missing node_community'
assert 'community_labels' in r, 'missing community_labels'
assert 'available_resolutions' in r, 'missing available_resolutions'
n_fts = len(r['fts_results'])
n_vss = len(r['vss_results'])
n_nodes = len(r['graph_nodes'])
n_edges = len(r['graph_edges'])
n_comm = r.get('community_count', 0)
print(f'  OK: fts={n_fts} vss={n_vss} nodes={n_nodes} edges={n_edges} communities={n_comm}')
" 2>&1; then
        echo "  FAIL: Response validation failed"
        echo "  Response: ${response:0:200}"
        return 1
    fi
    return 0
}

echo "=== Server Reliability Test ==="
echo "  Port: ${PORT}"
echo "  Target: ${ITERATIONS} consecutive passes"
echo ""

# Initial cleanup
kill_port
wait_port_free || { echo "FATAL: Cannot free port ${PORT}"; exit 1; }

trap 'kill_port' EXIT

while [ "$consecutive_passes" -lt "$ITERATIONS" ]; do
    total_attempts=$((total_attempts + 1))
    echo "[Attempt ${total_attempts}] Pass ${consecutive_passes}/${ITERATIONS}"

    # 1. Start the server (no reload, background, suppress logs)
    uv run --directory "${VIZ_DIR}" python -m server --port "${PORT}" > /dev/null 2>&1 &

    # 2. Wait for health
    if ! wait_for_health; then
        echo "  Server failed to become healthy"
        kill_port
        wait_port_free || { echo "Cannot free port"; consecutive_passes=0; continue; }
        consecutive_passes=0
        continue
    fi
    echo "  Server ready"

    # 3. Run KG query
    if ! run_kg_query; then
        kill_port
        wait_port_free || true
        consecutive_passes=0
        continue
    fi

    # 4. Run a second query to verify stability
    if ! run_kg_query; then
        echo "  Second query failed"
        kill_port
        wait_port_free || true
        consecutive_passes=0
        continue
    fi

    # 5. Shut down cleanly — kill by port, not by PID
    kill_port

    # 6. Verify port is actually freed
    if ! wait_port_free; then
        echo "  FAIL: Port not freed after shutdown"
        consecutive_passes=0
        continue
    fi

    consecutive_passes=$((consecutive_passes + 1))
    echo "  PASS (${consecutive_passes}/${ITERATIONS})"
    echo ""
done

echo "=== ALL ${ITERATIONS} PASSES CONSECUTIVE ==="
echo "  Total attempts: ${total_attempts}"
