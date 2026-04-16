#!/usr/bin/env bash
# Start backend + frontend + (optional) cloudflare tunnels.
# Usage:
#   ./start.sh           # local only
#   ./start.sh --tunnel  # expose via Cloudflare tunnel (Vast.ai / remote)
#
# Press Ctrl+C to stop everything.

cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

TUNNEL=false
[ "$1" = "--tunnel" ] && TUNNEL=true

PIDS=()

cleanup() {
    echo ""
    echo "Stopping all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait "${PIDS[@]}" 2>/dev/null
    rm -f /tmp/cf_be.log /tmp/cf_fe.log
    echo "Done."
}
trap cleanup SIGINT SIGTERM

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[BE] Starting FastAPI on :8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
PIDS+=($!)

# Chờ backend sẵn sàng
echo "[BE] Waiting for backend..."
for i in $(seq 1 30); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && break
    sleep 1
done
echo "[BE] Backend ready."

# ── Tunnel cho Backend ────────────────────────────────────────────────────────
API_URL="http://localhost:8000"

if [ "$TUNNEL" = true ]; then
    cloudflared tunnel --url http://localhost:8000 --no-autoupdate > /tmp/cf_be.log 2>&1 &
    PIDS+=($!)

    echo "[Tunnel] Waiting for API tunnel URL..."
    for i in $(seq 1 20); do
        API_URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' /tmp/cf_be.log 2>/dev/null | head -1)
        [ -n "$API_URL" ] && break
        sleep 1
    done

    if [ -z "$API_URL" ]; then
        echo "[Tunnel] Warning: could not detect API tunnel URL, using localhost."
        API_URL="http://localhost:8000"
    fi
    echo "[Tunnel] API URL: $API_URL"
fi

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[FE] Starting Streamlit on :8501 ..."
# Streamlit runs server-side → always call API via localhost, not tunnel
API_URL="http://localhost:8000" streamlit run ui/app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false &
PIDS+=($!)

# ── Tunnel cho Frontend ───────────────────────────────────────────────────────
FE_URL="http://localhost:8501"

if [ "$TUNNEL" = true ]; then
    sleep 2
    cloudflared tunnel --url http://localhost:8501 --no-autoupdate > /tmp/cf_fe.log 2>&1 &
    PIDS+=($!)

    echo "[Tunnel] Waiting for UI tunnel URL..."
    for i in $(seq 1 20); do
        FE_URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' /tmp/cf_fe.log 2>/dev/null | head -1)
        [ -n "$FE_URL" ] && break
        sleep 1
    done

    if [ -z "$FE_URL" ]; then
        echo "[Tunnel] Warning: could not detect UI tunnel URL."
        FE_URL="http://localhost:8501"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Backend  → $API_URL"
echo "  Frontend → $FE_URL"
[ "$TUNNEL" = true ] && echo "  (Streamlit gọi API qua: $API_URL)"
echo "  Press Ctrl+C to stop all."
echo "=========================================="

wait "${PIDS[@]}"
