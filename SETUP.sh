#!/usr/bin/env bash
# =============================================================================
# PrimateScope AI — One-Click Setup Script (macOS / Linux)
# =============================================================================
# Run this ONCE before first use. Takes ~2-3 minutes depending on connection.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PrimateScope AI — Field Intelligence System (Production v1.0)"
echo "  Setting up your environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# --- Step 1: Find a suitable Python (3.10-3.13 required, 3.14 blocked) ---
echo "  Searching for Python 3.10-3.13..."

# Try common Python executables in order of preference.
PYTHON_CANDIDATES=(
    "python3.12" "python3.11" "python3.13" "python3.10"
    "/opt/homebrew/opt/python@3.12/bin/python3.12"
    "/opt/homebrew/opt/python@3.11/bin/python3.11"
    "/usr/local/opt/python@3.12/bin/python3.12"
    "/usr/local/opt/python@3.11/bin/python3.11"
    "python3"
)

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" &> /dev/null || [ -x "$candidate" ]; then
        PV=$("$candidate" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}.{v.micro} {v.major} {v.minor}")' 2>/dev/null)
        if [ $? -eq 0 ]; then
            PMAJOR=$(echo "$PV" | awk '{print $2}')
            PMINOR=$(echo "$PV" | awk '{print $3}')
            if [ "$PMAJOR" -eq 3 ] && [ "$PMINOR" -ge 10 ] && [ "$PMINOR" -lt 14 ]; then
                PYTHON_BIN="$candidate"
                PYTHON_VERSION=$(echo "$PV" | awk '{print $1}')
                echo "    [OK] Found Python $PYTHON_VERSION at $PYTHON_BIN"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "  [ERROR] No suitable Python found."
    echo "    PrimateScope AI requires Python 3.10, 3.11, 3.12, or 3.13."
    echo "    Python 3.14 is NOT supported (SpeciesNet/MegaDetector incompatibility)."
    echo ""
    echo "    Install Python 3.12:"
    echo "      macOS (Homebrew):  brew install python@3.12"
    echo "      Or download from:  https://www.python.org/downloads/"
    exit 1
fi

# --- Step 2: Create virtual environment ---
echo ""
echo "  Creating virtual environment with $PYTHON_BIN..."
if [ -d "venv" ]; then
    # Check if existing venv is acceptable.
    EXISTING_VER=$("$SCRIPT_DIR/venv/bin/python" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")' 2>/dev/null)
    EXISTING_MINOR=$("$SCRIPT_DIR/venv/bin/python" -c 'import sys; print(sys.version_info[1])' 2>/dev/null)
    if [ -n "$EXISTING_MINOR" ] && [ "$EXISTING_MINOR" -lt 14 ] 2>/dev/null; then
        echo "    venv already exists (Python $EXISTING_VER) — skipping"
    else
        echo "    Existing venv uses unsupported Python — recreating..."
        rm -rf venv
        "$PYTHON_BIN" -m venv venv
        echo "    [OK] venv created with $PYTHON_VERSION"
    fi
else
    "$PYTHON_BIN" -m venv venv
    echo "    [OK] venv created with $PYTHON_VERSION"
fi

# --- Step 3: Upgrade pip ---
echo ""
echo "  Upgrading pip..."
VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/python" -m pip install --upgrade pip -q
echo "    [OK] pip upgraded"

# --- Step 4: Install core dependencies ---
echo ""
echo "  Installing core dependencies (~2 min)..."
VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/pip" install -r requirements.txt -q
echo "    [OK] Core dependencies installed"

# --- Step 5: Try installing SpeciesNet + MegaDetector (if not already installed) ---
echo ""
echo "  Checking SpeciesNet + MegaDetector..."
if VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/python" -c "import speciesnet" 2>/dev/null; then
    echo "    [OK] SpeciesNet already installed"
else
    echo "    Installing SpeciesNet..."
    if VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/pip" install speciesnet --use-pep517 -q 2>&1; then
        echo "    [OK] SpeciesNet installed"
    else
        echo "    [WARN] SpeciesNet install failed. App runs in Demo mode without it."
    fi
fi
if VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/python" -c "import megadetector" 2>/dev/null; then
    echo "    [OK] MegaDetector already installed"
else
    echo "    Installing MegaDetector..."
    if VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/pip" install megadetector -q 2>&1; then
        echo "    [OK] MegaDetector installed"
    else
        echo "    [WARN] MegaDetector install failed. App runs in Demo mode without it."
    fi
fi

# --- Step 6: YOLOv8n model download (for demo mode) ---
echo ""
echo "  Downloading YOLOv8n model (~6 MB)..."
MODEL_PATH="$SCRIPT_DIR/yolov8n.pt"
if [ -f "$MODEL_PATH" ]; then
    echo "    Model already present"
else
    VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/python" -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
" 2>/dev/null || echo "    [WARN] Model download skipped — demo simulation mode"
fi

# --- Step 7: Environment check ---
echo ""
echo "  Running environment check..."
VIRTUAL_ENV=1 "$SCRIPT_DIR/venv/bin/python" scripts/check_environment.py || true

# --- Done ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [OK] Setup complete!"
echo ""
echo "  Next step — launch the app:"
echo "    ./LAUNCH.sh"
echo ""
echo "  The app opens at http://localhost:8501"
echo "  Default mode: Demo Simulation"
echo "  For real inference: switch to Real Inference mode in the sidebar"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
