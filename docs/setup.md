# Setup

The short version: `./setup.sh`. The long version follows.

## System requirements

- **Python 3.11.** Newer versions usually work; 3.10 and older will not.
  On macOS use `pyenv` or the python.org installer; on Linux the system
  python is usually fine if it is 3.11+.
- **ffmpeg.** Needed for clip extraction and video re-encoding. Install
  via Homebrew (`brew install ffmpeg`) or apt (`apt-get install ffmpeg`).
- **git.** For cloning.
- **~2 GB free disk.** Most of that is PyTorch.

Optional but recommended:

- **GPU.** A CUDA GPU (NVIDIA) or Apple Silicon (MPS) speeds up the YOLO
  inference path significantly. CPU works for short videos.

## Step-by-step

### 1. Clone and run setup

```bash
git clone https://github.com/dawsonpar/SetOptics.git
cd SetOptics
./setup.sh
```

`setup.sh` creates `.venv/`, installs all dependencies from
`requirements.txt`, and prompts for your `GEMINI_API_KEY`. It is
idempotent: rerun it any time.

### 2. Activate the venv

```bash
source .venv/bin/activate
```

You will need to do this in every new shell.

### 3. Verify

```bash
python -c "import setoptics; print(setoptics.__version__)"
python scripts/signal_rally_detector.py --help
```

If both commands return cleanly, you are done.

## API keys

Only `GEMINI_API_KEY` is required, and only if you want to use the LLM
detection or annotation paths. The signal and ensemble (`--mode fast`)
modes run fully local with no key.

See [`api-keys.md`](api-keys.md) for details on where to get one and how
the loader resolves keys.

## Troubleshooting

- **`No module named 'av'`.** Activate the venv (`source .venv/bin/activate`).
- **`ImportError: libGL.so.1`** on Linux. Install `libgl1` via apt.
- **YOLO weights missing.** `setup.sh` assumes
  `models/volleyball_yolo26n.pt` exists in the repo. If you removed it
  (or are on a sparse checkout), grab it from the GitHub release.
- **Gemini 429 / quota.** The free tier has tight rate limits. Throttle
  with `--parallel 1` in the annotation tools, or upgrade your Gemini
  plan.
