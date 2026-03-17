#!/usr/bin/env bash
# Browser reliability test — Playwright-based KG query search cycle.
# Usage: bash viz/tests/test_browser_reliability.sh [ITERATIONS]
#
# Tests the full browser experience: navigate → search → verify results.
# Server + browser start once; each iteration navigates fresh → search → verify.
# Uses sequential playwright-cli commands (goto, fill, snapshot) — no run-code.

set -euo pipefail

ITERATIONS="${1:-20}"
BACKEND_PORT=8203
FRONTEND_PORT=5283
HEALTH_URL="http://localhost:${BACKEND_PORT}/api/health"
PAGE_URL="http://localhost:${FRONTEND_PORT}/kg/query/"
VIZ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "${VIZ_DIR}/.." && pwd)"
SNAP_FILE="${PROJECT_ROOT}/.browser-test-snap.yml"
SNAP_BASENAME=".browser-test-snap.yml"

consecutive_passes=0
total_attempts=0

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"${port}" 2>/dev/null || true)
    if [ -z "$pids" ]; then return 0; fi
    echo "$pids" | xargs kill 2>/dev/null || true
    local waited=0
    while [ $waited -lt 6 ]; do
        pids=$(lsof -ti:"${port}" 2>/dev/null || true)
        [ -z "$pids" ] && return 0
        sleep 0.5
        waited=$((waited + 1))
    done
    pids=$(lsof -ti:"${port}" 2>/dev/null || true)
    [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
}

cleanup_all() {
    playwright-cli close 2>/dev/null || true
    kill_port "${FRONTEND_PORT}"
    kill_port "${BACKEND_PORT}"
    rm -f "${SNAP_FILE}" 2>/dev/null || true
}

wait_for_url() {
    local url=$1 max_wait=$2
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# Extract the ref for the search textbox from a snapshot file
find_search_ref() {
    local snap_file=$1
    grep -o 'textbox "Search the knowledge graph[^"]*" *\[.*ref=e[0-9]*' "$snap_file" \
        | grep -o 'ref=e[0-9]*' | head -1 | sed 's/ref=//'
}

# Ensure browser is alive — reopen if crashed
ensure_browser() {
    if playwright-cli list 2>&1 | grep -q "status: open"; then
        return 0
    fi
    echo "  [recovery] Browser died, reopening..."
    playwright-cli kill-all 2>/dev/null || true
    sleep 1
    if ! playwright-cli open "${PAGE_URL}" > /dev/null 2>&1; then
        return 1
    fi
    sleep 1
    return 0
}

# Navigate to URL with browser recovery — retries if goto fails
navigate_with_recovery() {
    local url=$1
    local attempt=0
    while [ $attempt -lt 3 ]; do
        if ! ensure_browser; then
            attempt=$((attempt + 1))
            continue
        fi
        if playwright-cli goto "$url" > /dev/null 2>&1; then
            return 0
        fi
        # goto failed — browser probably died mid-command
        echo "  [recovery] goto failed, reopening browser..."
        playwright-cli kill-all 2>/dev/null || true
        sleep 1
        if playwright-cli open "$url" > /dev/null 2>&1; then
            sleep 1
            return 0
        fi
        attempt=$((attempt + 1))
    done
    return 1
}

echo "=== Browser Reliability Test ==="
echo "  Backend: ${BACKEND_PORT}, Frontend: ${FRONTEND_PORT}"
echo "  Target: ${ITERATIONS} consecutive passes"
echo ""

trap cleanup_all EXIT

# Clean ports and stale browsers
playwright-cli kill-all 2>/dev/null || true
kill_port "${BACKEND_PORT}"
kill_port "${FRONTEND_PORT}"

# Start backend
echo "Starting backend..."
uv run --directory "${VIZ_DIR}" python -m server --port "${BACKEND_PORT}" > /dev/null 2>&1 &
if ! wait_for_url "${HEALTH_URL}" 30; then
    echo "FATAL: Backend failed to start"
    exit 1
fi
echo "  Backend ready"

# Start frontend
echo "Starting frontend..."
VITE_API_PORT="${BACKEND_PORT}" npm --prefix "${VIZ_DIR}/frontend" run dev -- --port "${FRONTEND_PORT}" > /dev/null 2>&1 &
if ! wait_for_url "http://localhost:${FRONTEND_PORT}/" 15; then
    echo "FATAL: Frontend failed to start"
    exit 1
fi
echo "  Frontend ready"

# Open browser once — keep it open for all iterations
echo "Opening browser..."
if ! playwright-cli open "${PAGE_URL}" > /dev/null 2>&1; then
    echo "FATAL: Could not open browser"
    exit 1
fi
sleep 1
echo "  Browser ready"
echo ""

while [ "$consecutive_passes" -lt "$ITERATIONS" ]; do
    total_attempts=$((total_attempts + 1))
    echo "[Attempt ${total_attempts}] Pass ${consecutive_passes}/${ITERATIONS}"

    # 1. Navigate to KG query page (with browser recovery if needed)
    if ! navigate_with_recovery "${PAGE_URL}"; then
        echo "  FAIL: Could not navigate after 3 recovery attempts"
        consecutive_passes=0
        continue
    fi
    sleep 1  # Let page fully render before snapshot

    # 2. Take snapshot to find the search input ref (retry up to 3 times)
    search_ref=""
    snap_retries=0
    while [ $snap_retries -lt 3 ] && [ -z "$search_ref" ]; do
        rm -f "${SNAP_FILE}" 2>/dev/null || true
        if playwright-cli snapshot --filename="${SNAP_BASENAME}" > /dev/null 2>&1 && [ -f "${SNAP_FILE}" ]; then
            search_ref=$(find_search_ref "${SNAP_FILE}")
        fi
        if [ -z "$search_ref" ]; then
            sleep 0.5
            snap_retries=$((snap_retries + 1))
        fi
    done
    if [ -z "$search_ref" ]; then
        echo "  FAIL: Could not find search textbox after ${snap_retries} snapshot retries"
        consecutive_passes=0
        continue
    fi

    # 3. Fill search box
    if ! playwright-cli fill "${search_ref}" "division of labor" > /dev/null 2>&1; then
        echo "  FAIL: Could not fill search input"
        consecutive_passes=0
        continue
    fi

    # 4. Wait for results — poll snapshots until "points" appears
    results_found=false
    poll_waited=0
    max_poll=30  # Up to 15 seconds (30 * 0.5s)
    while [ $poll_waited -lt $max_poll ]; do
        sleep 0.5
        poll_waited=$((poll_waited + 1))

        rm -f "${SNAP_FILE}" 2>/dev/null || true
        playwright-cli snapshot --filename="${SNAP_BASENAME}" > /dev/null 2>&1 || continue
        [ -f "${SNAP_FILE}" ] || continue

        if grep -q "points" "${SNAP_FILE}" 2>/dev/null; then
            results_found=true
            break
        fi
    done

    if [ "$results_found" != "true" ]; then
        echo "  FAIL: Search did not produce results within $((max_poll / 2))s"
        consecutive_passes=0
        continue
    fi

    # 5. Verify panels have content
    has_points=$(grep -c "points" "${SNAP_FILE}" 2>/dev/null || echo "0")
    has_nodes=$(grep -c "nodes" "${SNAP_FILE}" 2>/dev/null || echo "0")
    has_communities=$(grep -c "communities\|Community" "${SNAP_FILE}" 2>/dev/null || echo "0")
    has_chunks=$(grep -c "Chunk" "${SNAP_FILE}" 2>/dev/null || echo "0")

    if [ "$has_points" -lt 1 ] || [ "$has_nodes" -lt 1 ]; then
        echo "  FAIL: Missing panel content (points=${has_points} nodes=${has_nodes} communities=${has_communities} chunks=${has_chunks})"
        consecutive_passes=0
        continue
    fi

    consecutive_passes=$((consecutive_passes + 1))
    echo "  OK: points=${has_points} nodes=${has_nodes} communities=${has_communities} chunks=${has_chunks} (poll: ${poll_waited})"
    echo "  PASS (${consecutive_passes}/${ITERATIONS})"
done

echo ""
echo "=== ALL ${ITERATIONS} BROWSER PASSES CONSECUTIVE ==="
echo "  Total attempts: ${total_attempts}"
