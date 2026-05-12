# Architecture

A high-level map of how the pieces fit. For per-script details, read
`docs/rally-detection.md`.

## Layers

```
+-----------------------------------------------------------------+
|  CLI entry points                                               |
|    scripts/*_rally_detector.py                                  |
|    scripts/*_rally_evaluate.py                                  |
|    tools/annotation/*.py                                        |
+-----------------------------------------------------------------+
|  setoptics/ package                                             |
|    rally_detector_base.py    Common detector interface          |
|    gemini_rally_detector.py  Production-grade LLM pipeline      |
|    gemini_pipeline.py        Clip extraction, IoU, timeline     |
|    ball_tracker.py           YOLO + BoT-SORT ball tracking      |
|    prompts.py                Gemini prompt templates            |
+-----------------------------------------------------------------+
|  tools/shared/                                                  |
|    eval/temporal.py          IoU, F1, matching, reporting       |
|    providers/google_provider.py  Gemini SDK abstraction         |
+-----------------------------------------------------------------+
|  External                                                       |
|    Gemini API (LLM modes only)                                  |
|    PyTorch, Ultralytics, OpenCV, PyAV, librosa (local modes)    |
+-----------------------------------------------------------------+
```

## Data flow (signal mode)

1. Decode video frames with PyAV at 2 fps.
2. Decode audio with librosa.
3. Per-second feature extraction:
   - Whistle-band energy (2-4 kHz).
   - Silence ratio.
   - Optical-flow motion percentiles.
4. Smoothed derivatives + leaky integration produce a continuous "rally
   likelihood" signal.
5. Threshold + min-duration filter yields segments.

## Data flow (ensemble mode)

1. Run signal detector.
2. Run VideoMAE detector (if weights present).
3. Optionally run Gemini gating on borderline windows.
4. Weighted fusion of per-second scores.
5. Threshold + min-duration filter.

## Data flow (Gemini LLM mode)

1. **Stage 1** (sliding window detect): 2-minute windows, 30s stride.
   Gemini returns candidate rallies per window.
2. **Stage 2** (boundary refinement): 45-second clips around each
   candidate boundary, refined to second-level precision.
3. **Stage 3** (gap fill): long gaps between rallies re-scanned for
   missed candidates.
4. Timeline merge + IoU dedup.

The prompts have been tuned across 19 iterations against indoor
volleyball ground truth (94.6% F1 on training, 93.5% on held-out).

## Why this layout

- **`setoptics/` is the library.** Importable, stable, what an agent
  reaches for first.
- **`scripts/` is research-grade CLIs.** Easier to iterate on without
  breaking imports.
- **`tools/` is workflow tooling.** Annotation and evaluation, not
  inference.
- **Shared eval lives at one path** so all detectors are compared on
  the same metrics, never accidentally diverged.

## Extending

To add a new detector:

1. Write the detector under `setoptics/` or `scripts/`.
2. Make its output schema match the existing
   `{segments: [{start_ms, end_ms, type}]}` shape.
3. Reuse `tools/shared/eval/temporal.py` for evaluation.
4. Add an entry to `docs/rally-detection.md`.
