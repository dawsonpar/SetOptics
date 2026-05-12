<p align="center">
  <img src="assets/logo.png#gh-light-mode-only" alt="SetOptics" width="200" />
  <img src="assets/logo-white.png#gh-dark-mode-only" alt="SetOptics" width="200" />
</p>

# SetOptics

Volleyball rally detection for any agent harness (Claude Code, Cursor, Codex,
Aider, plain Python). Point it at a video, get back the time ranges where the
ball is actually in play.

This repository is the open-source release of the rally-detection layer of
SetOptics. The hosted product (auth, billing, web UI) is closed-source; the
detection pipelines are here in full.

## What's in the box

| Mode | Method | Speed | Accuracy | Cost |
|------|--------|-------|----------|------|
| `signal` | Audio + optical flow heuristics | Fast | F1 ~47% | Free, fully local |
| `videomae` | VideoMAE V2 + MLP head | Medium | High segment recall | Free, fully local, needs trained weights |
| `ensemble` | Signal + VideoMAE (+ optional LLM) | Medium | F1 ~69-72% | Free unless LLM enabled |
| `llm` | Gemini 2.5 Flash, 3-stage sliding window | Slow | F1 ~94% | Free tier, then paid |

The annotation tools (`tools/annotation/`) are the same pipeline SetOptics
uses internally to build ground truth. They are useful even if you do not run
detection at all.

## Quickstart

```bash
git clone https://github.com/dawsonpar/SetOptics.git
cd SetOptics
./setup.sh                          # creates .venv, installs deps, prompts for GEMINI_API_KEY
source .venv/bin/activate
python scripts/signal_rally_detector.py \
    --video path/to/your/footage.mp4 \
    --output path/to/output.json
```

Output is a JSON file listing in-play segments by start/end milliseconds.

## Repository layout

```
setoptics/        Importable Python package
  ball_tracker.py            YOLO + BoT-SORT ball tracking
  gemini_rally_detector.py   3-stage Gemini rally pipeline
  gemini_pipeline.py         Clip extraction, timeline merge, IoU helpers
  rally_detector_base.py     Base class
  prompts.py                 Gemini prompt templates
scripts/          Standalone CLI entry points (research-grade)
  signal_rally_detector.py
  ensemble_rally_detector.py
  infer_rally_detector.py    VideoMAE inference (requires user-trained weights)
  signal_rally_evaluate.py
  ensemble_rally_evaluate.py
  train_volleyball_yolo.py
  validate_ball_tracking.py
tools/            Annotation and evaluation tooling
  annotation/                Ground-truth labeling (Gemini-driven)
  shared/eval/               Temporal F1, IoU, matching framework
  shared/providers/          LLM provider abstractions (Google only in OSS)
models/           Pretrained weights (YOLO ball detector ships in-repo)
docs/             User guides
```

## Modes in detail

### Signal-based detection
Pure heuristic, no model weights, fully local. Good baseline and useful as
one input to the ensemble. F1 hovers around 47% on indoor footage.

```bash
python scripts/signal_rally_detector.py \
    --video footage.mp4 --output segments.json
```

### Ensemble
Combines signal + VideoMAE (+ optional LLM) with learned fusion weights.
Default mode disables the LLM call. Enable with `--mode accurate` to add
Gemini, which gates the ensemble through an additional API call but
materially improves precision.

```bash
python scripts/ensemble_rally_detector.py \
    --video footage.mp4 --output segments.json --mode fast
```

### VideoMAE
The script (`scripts/infer_rally_detector.py`) ships, but the model weights
do not. Bring your own `rally_detector.pth`, or train one on your own
labeled data (see `docs/training.md`).

### LLM (Gemini) annotation
The highest-quality path. Used internally for ground-truth labeling.

```bash
python tools/annotation/annotate_sliding_window.py path/to/video.mp4
```

Writes `<video>_raw_annotations.json` next to the input video. Requires
`GEMINI_API_KEY`.

## Setup

See [`docs/setup.md`](docs/setup.md) for the long form. The short form:

1. Python 3.11+, `ffmpeg`, and `git` on your path.
2. `./setup.sh` creates a venv, installs locked deps, and prompts for your
   Gemini key.
3. Drop a video at any path and run one of the scripts above.

## Bring your own footage and data

Out of the box, you can run detection on any video. To evaluate a detector
against your own ground truth, see [`docs/annotation.md`](docs/annotation.md)
for the label format and Label Studio workflow.

Training data is not shipped. The YOLO ball-detector weights at
`models/volleyball_yolo26n.pt` are included and ready to use.

## Agent use

This repository is structured to be agent-friendly. Drop it into Claude Code,
Cursor, Codex, or any other harness, and the model can navigate it from
`AGENTS.md`. The conventions there are universal (no harness-specific
features).

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Trademark

"SetOptics" and the SetOptics logo (the eye-with-volleyball-iris mark) are
trademarks of Dawson Par. The Apache 2.0 license covers the code in this
repository only; it does not grant permission to use the name or logo on
derivative products or services. Forks are welcome, but please rename your
project and remove or replace the logo if you distribute it as your own.

## Roadmap

- A native Claude Code skill that exposes detection as a single
  agent-callable tool. The first attempt did not generalize beyond the
  training set, so the work is paused until the eval dataset grows. Tracked
  as a GitHub issue.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome on detector
improvements, eval framework, and annotation UX. The hosted product layer
(auth, web UI, infra) lives elsewhere and is out of scope.

## Security

See [`SECURITY.md`](SECURITY.md) for vulnerability reporting.
