# AGENTS.md

Conventions for any agent (Claude Code, Cursor, Codex, Aider, autonomous
or interactive) working inside this repository. Harness-agnostic.

## What this repo is

Open-source rally-detection pipelines for volleyball video. Library +
scripts, no service layer. The detection modes are:
`signal`, `videomae`, `ensemble`, `llm` (Gemini).

## Where the code lives

- `setoptics/` is the importable Python package (ball tracker, Gemini
  pipeline, prompts, base classes).
- `scripts/` holds standalone CLI entry points. They are research-grade
  but functional. Each one is its own `__main__` and uses `argparse`.
- `tools/annotation/` holds the ground-truth labeling pipeline. It is
  importable and runnable.
- `tools/shared/eval/temporal.py` is the canonical evaluation framework.
  Use it for any new rally/segment metric work; do not write parallel
  metric code.

## Working rules for agents

1. **Read `README.md` first.** It is the source of truth for what each mode
   does and how to invoke it.
2. **Use the canonical entry points.** Detection runs through
   `scripts/*_rally_detector.py`. Annotation runs through
   `tools/annotation/annotate_sliding_window.py` (high quality) or
   `annotate_fast.py` (fast iteration). Do not invent new wrappers.
3. **Activate the venv before running Python.** All scripts assume the
   project venv is active (`source .venv/bin/activate`).
4. **Do not commit user data.** `data/` and any `*_annotations*.json` files
   are gitignored. Do not stage them.
5. **One file = one purpose.** Hard cap of 500 lines per code file. If a
   change pushes over this, split.
6. **No new directories without reason.** The layout is fixed; new modules
   go in `setoptics/`, new CLIs in `scripts/`, new annotation tools in
   `tools/annotation/`.
7. **Apache 2.0.** All new files inherit the repo license. Do not add
   conflicting license headers.

## Common tasks

### Run rally detection on a video

```bash
python scripts/signal_rally_detector.py --video FOOTAGE.mp4 --output OUT.json
python scripts/ensemble_rally_detector.py --video FOOTAGE.mp4 --output OUT.json --mode fast
python tools/annotation/annotate_sliding_window.py FOOTAGE.mp4   # LLM path
```

### Evaluate a detector against ground truth

```bash
python scripts/signal_rally_evaluate.py --predictions OUT.json --ground-truth GT.json
python scripts/ensemble_rally_evaluate.py --predictions OUT.json --ground-truth GT.json
python tools/annotation/eval_sliding_window.py FOOTAGE.mp4
```

The ground truth format and IoU/F1 conventions are documented in
`tools/shared/eval/temporal.py`. Read it before adding a new metric.

### Add a new detector

1. Implement the detector as a module under `setoptics/` if it is reusable,
   or under `scripts/` if it is a one-shot CLI.
2. Subclass `setoptics.rally_detector_base.RallyDetectorBase` if you want
   the existing evaluation harness to pick it up automatically.
3. Add an `_evaluate.py` companion under `scripts/` that uses
   `tools/shared/eval/temporal.py`. Do not duplicate metric code.

### Train your own ball detector

See `docs/training.md`. The YOLO trainer is `scripts/train_volleyball_yolo.py`.

## Environment

- Python 3.11 (use `pyenv` or system).
- `ffmpeg` on path.
- A `.env` at the repo root with `GEMINI_API_KEY=...` for any LLM path.
- Optional `GOOGLE_API_KEY` alias works for the annotation tools.
- The repo never ships secrets. Never write a key into source.

## What is NOT here

- No FastAPI server, Cloud Run, GCS wiring, Stripe, Clerk, or web UI. Those
  belong to the hosted SetOptics product and are intentionally excluded.
- No training datasets. Bring your own.

## When in doubt

Prefer:
- editing an existing file over creating a new one,
- using `tools/shared/eval/temporal.py` over writing new metric code,
- adding a CLI flag over creating a new script.

If a task seems to require violating any of the above, stop and surface the
trade-off in your PR description.
