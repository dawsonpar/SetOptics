# Rally detection

Four modes. Pick based on speed, accuracy, and whether you want to spend
API tokens.

| Mode | Script | Speed | Accuracy | API cost |
|------|--------|-------|----------|----------|
| Signal | `scripts/signal_rally_detector.py` | ~real-time | F1 ~47% | None |
| VideoMAE | `scripts/infer_rally_detector.py` | ~real-time | High recall, needs weights | None |
| Ensemble (fast) | `scripts/ensemble_rally_detector.py --mode fast` | ~real-time | F1 ~69% | None |
| Ensemble (accurate) | `scripts/ensemble_rally_detector.py --mode accurate` | medium | F1 ~72% | Free tier, then paid |
| LLM | `tools/annotation/annotate_sliding_window.py` | ~3-4x slower | F1 ~94% | Free tier, then paid |

All scripts write JSON: a list of `{start_ms, end_ms, type}` segments
where `type == "in-play"` marks a rally.

## Signal-based detection

Pure audio + optical-flow heuristic. No models beyond the ball detector.

```bash
python scripts/signal_rally_detector.py \
    --video FOOTAGE.mp4 \
    --output FOOTAGE_signal.json
```

Tuned for indoor volleyball. Beach footage will need different weights;
see the dataclass at the top of the script.

## VideoMAE detection

```bash
python scripts/infer_rally_detector.py \
    --video FOOTAGE.mp4 \
    --model PATH/TO/rally_detector.pth \
    --output FOOTAGE_videomae.json \
    --trim
```

`rally_detector.pth` is not shipped. Train your own (see
[`training.md`](training.md)) or skip this mode.

## Ensemble detection

Combines signal + VideoMAE (+ optional LLM gating).

```bash
# Fast: signal + VideoMAE, fully local.
python scripts/ensemble_rally_detector.py \
    --video FOOTAGE.mp4 \
    --output FOOTAGE_ensemble.json \
    --mode fast

# Accurate: adds Gemini gating, requires GEMINI_API_KEY.
python scripts/ensemble_rally_detector.py \
    --video FOOTAGE.mp4 \
    --output FOOTAGE_ensemble.json \
    --mode accurate
```

Optimal fusion weights from the benchmark: signal=0.3, videomae=0.4,
llm=0.3, threshold=0.5.

## LLM (Gemini) detection

Highest accuracy. Slower because it makes multiple Gemini calls per
video (3-stage sliding window with boundary refinement).

```bash
python tools/annotation/annotate_sliding_window.py FOOTAGE.mp4
```

Writes `FOOTAGE_raw_annotations.json` next to the input. This is the same
pipeline used internally for ground-truth labeling; quality is the
benchmark the other detectors aim to match.

For batch use:

```bash
python tools/annotation/annotate_sliding_window.py \
    FOOTAGE_001.mp4 FOOTAGE_002.mp4 \
    --parallel 2
```

`--parallel 2` is the safe default for the Gemini free tier.

## Evaluation

If you have ground truth, evaluate any detector with the matching
`_evaluate.py` script:

```bash
python scripts/signal_rally_evaluate.py \
    --predictions FOOTAGE_signal.json \
    --ground-truth FOOTAGE_annotations_corrected.json

python scripts/ensemble_rally_evaluate.py \
    --predictions FOOTAGE_ensemble.json \
    --ground-truth FOOTAGE_annotations_corrected.json

python tools/annotation/eval_sliding_window.py FOOTAGE.mp4  # runs detection + eval in one pass
```

All evaluators report F1 at IoU >= 0.5 (primary) plus a center-mode
boundary-tolerance grid (±0.5s, ±1s, ±2s, ±5s).

## Exporting a rallies-only video

Every detector above outputs JSON segments. To turn that JSON into an MP4
that contains only the rallies (breaks removed), use `export_rallies.py`:

```bash
python scripts/export_rallies.py \
    --segments FOOTAGE_signal.json \
    --video FOOTAGE.mp4 \
    --output FOOTAGE_rallies.mp4
```

The script reads any rally-detection JSON (signal, ensemble, or LLM
annotation output) and concatenates only the in-play segments.

| Flag | Default | What it does |
|------|---------|--------------|
| `--mode accurate` | (default) | Re-encodes via ffmpeg filter_complex. Exact ms-aligned boundaries. ~real-time speed. |
| `--mode fast` | | Stream-copy via ffmpeg concat demuxer. ~10x faster. Boundaries snap to the nearest keyframe (drift up to ~2s). |
| `--pad SECONDS` | `0.0` | Add this much padding before and after each rally. |
| `--min-rally-sec SECONDS` | `0.0` | Drop rallies shorter than this. |
| `--output PATH` | `<video-stem>_rallies.mp4` | Output path. |

Requires `ffmpeg` and `ffprobe` on PATH. Both are installed by `setup.sh`'s
prerequisite check on macOS (`brew install ffmpeg`) and Linux
(`apt-get install ffmpeg`).
