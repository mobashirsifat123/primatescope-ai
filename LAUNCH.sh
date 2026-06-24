#!/usr/bin/env bash
# =============================================================================
# PrimateScope AI — Launch Script (macOS / Linux)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo ""
    echo "  [ERROR] venv not found. Run SETUP.sh first."
    exit 1
fi

# Kill any existing instance on port 8501 to avoid conflicts.
if lsof -ti :8501 &>/dev/null; then
    echo "  Port 8501 in use — stopping previous instance..."
    lsof -ti :8501 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

PORT=8501

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PrimateScope AI — Starting..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  URL: http://localhost:${PORT}"
echo "  Press Ctrl+C to stop."
echo ""

# Open browser after a short delay (macOS only).
if [[ "$(uname)" == "Darwin" ]]; then
    (sleep 3 && open "http://localhost:${PORT}") &
fi

exec "$SCRIPT_DIR/venv/bin/streamlit" run app.py \
    --server.port ${PORT} \
    --server.address 127.0.0.1 \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#2EEAD3" \
    --theme.backgroundColor "#0A0F0D" \
    --theme.secondaryBackgroundColor "#141C19" \
    --theme.textColor "#FFFFFF"
