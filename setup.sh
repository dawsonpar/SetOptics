#!/usr/bin/env bash
# One-shot setup for SetOptics rally detection.
# Creates a Python 3.11 venv, installs locked deps, prompts for GEMINI_API_KEY.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> SetOptics setup"
echo "    Repo: $REPO_ROOT"

# 1. Check Python 3.11+
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install Python 3.11 first." >&2
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "ERROR: Python 3.11+ required (found $PY_VERSION)." >&2
    exit 1
fi
echo "    Python: $PY_VERSION"

# 2. Check ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found. Install it (brew install ffmpeg / apt-get install ffmpeg) before running detection." >&2
else
    echo "    ffmpeg: $(ffmpeg -version | head -1 | awk '{print $3}')"
fi

# 3. Create venv
if [ ! -d ".venv" ]; then
    echo "==> Creating .venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Install deps
echo "==> Installing dependencies (this can take a few minutes)"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# 5. .env scaffold
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "==> Created .env from .env.example"
fi

if grep -q "^GEMINI_API_KEY=$" .env; then
    echo ""
    echo "GEMINI_API_KEY is not set in .env."
    echo "Get a key at https://aistudio.google.com/apikey"
    read -r -p "Paste your GEMINI_API_KEY now (or leave blank to skip): " USER_KEY
    if [ -n "$USER_KEY" ]; then
        # Replace blank line with the key, in place.
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^GEMINI_API_KEY=$|GEMINI_API_KEY=$USER_KEY|" .env
        else
            sed -i "s|^GEMINI_API_KEY=$|GEMINI_API_KEY=$USER_KEY|" .env
        fi
        echo "    Saved to .env"
    else
        echo "    Skipped. Edit .env later if you want to use the LLM modes."
    fi
fi

# 6. YOLO weights check
WEIGHTS="models/volleyball_yolo26n.pt"
if [ ! -f "$WEIGHTS" ]; then
    echo "WARNING: $WEIGHTS not found." >&2
    echo "    The ball tracker will not work without it." >&2
else
    SIZE=$(du -h "$WEIGHTS" | awk '{print $1}')
    echo "    YOLO weights: $WEIGHTS ($SIZE)"
fi

# 7. Smoke test
echo "==> Smoke test"
python -c "import setoptics; print(f'    setoptics v{setoptics.__version__} importable')"

echo ""
echo "==> Done."
echo ""
echo "Activate the venv:    source .venv/bin/activate"
echo "Run signal detector:  python scripts/signal_rally_detector.py --help"
echo "Run LLM annotation:   python tools/annotation/annotate_sliding_window.py --help"
