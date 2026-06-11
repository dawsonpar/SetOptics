---
name: setoptics-detect
description: Detect volleyball rallies in a local video and optionally export a rallies-only MP4. Use when the user asks to find rallies, cut dead time, run rally detection, or trim volleyball footage.
argument-hint: <video-path> [--mode signal|ensemble|llm] [--export]
---

# SetOptics rally detection

Run rally detection on a local video using the canonical SetOptics scripts.
Never write new wrapper scripts — AGENTS.md rule 2. All commands run from the
repo root with the project venv active.

## Arguments

- `$ARGUMENTS` contains the video path and optional flags.
- `--mode`: `signal` (default), `ensemble`, or `llm`.
- `--export`: also produce a rallies-only MP4 next to the input video.
- If no video path is given, ask the user for one. Verify the file exists
  before running anything.

## Accuracy — tell the user up front

State the expected accuracy of the chosen mode BEFORE running, so the user
can switch modes if the trade-off is wrong for them:

| Mode | F1 (indoor footage) | Cost | Needs |
|------|--------------------|------|-------|
| `signal` | ~47% | Free, local | nothing |
| `ensemble` | ~69-72% | Free, local (accurate mode may use LLM) | trained weights for VideoMAE path |
| `llm` | ~94% | Gemini API tokens, 3-4x slower | `GEMINI_API_KEY` in `.env` |

These numbers come from the project's own eval set; accuracy on unseen
recording conditions (different gym, camera, mic) can be lower.

## Step 1: Environment

1. If `.venv/` is missing at the repo root, run `./setup.sh` (it is fine to
   skip the GEMINI_API_KEY prompt unless `llm` mode was requested).
2. Activate it: `source .venv/bin/activate`.
3. For `llm` mode: check that `.env` at the repo root contains
   `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). If not, ask the user for a key
   before proceeding. Never echo the key back or write it anywhere except
   `.env`.
4. For `--export`: check `ffmpeg` and `ffprobe` are on PATH.

## Step 2: Detect

Let `V` be the resolved video path. Run exactly one of:

```bash
# signal (default)
python scripts/signal_rally_detector.py --video V --output V_signal.json

# ensemble
python scripts/ensemble_rally_detector.py --video V --output V_ensemble.json --mode accurate

# llm — writes <video>_raw_annotations.json next to the video
python tools/annotation/annotate_sliding_window.py V
```

Notes:
- Ensemble accepts `--mode fast` (signal+videomae only, quicker) and
  `--domain indoor|beach` (default auto-detect).
- Detection on long footage takes minutes; run it in the background and
  report progress rather than blocking.

## Step 3: Report

The output JSON contains `{"segments": [{"start_ms", "end_ms", "type"}, ...]}`
where `type` is `in-play` or `break` (canonical format, see
`setoptics/rally_detector_base.py`). Report to the user:

- number of `in-play` segments found,
- total in-play time vs. total video duration,
- path of the JSON file.

## Step 4: Export (only with --export or when the user asks)

```bash
python scripts/export_rallies.py --segments OUT.json --video V --output V_rallies.mp4
```

- Add `--pad 0.5` if the user wants a beat of context around each rally.
- `--mode fast` is ~10x quicker via stream copy, but boundaries snap to
  keyframes and can drift up to ~2s. Default `accurate` re-encodes.
- `--min-rally-sec N` drops blips shorter than N seconds.

## Troubleshooting

- Output MP4 much shorter than expected / rallies missing: re-run the signal
  detector with `--fusion-threshold 0.35` (default 0.45), or suggest
  `ensemble`/`llm` mode per `docs/rally-detection.md`.
- Too many false positives (warm-up motion detected as rallies): raise
  `--fusion-threshold`, or add `--min-rally 5.0`.
- `llm` mode quality is tuned for `gemini-2.5-flash`; do not switch models
  unless the user explicitly wants to benchmark another one.
