# Models

## `volleyball_yolo26n.pt`

YOLO v26 nano ball detector, trained on ~19k labeled volleyball frames
(mostly indoor). 5.4 MB. Optimized for Apple Silicon (MPS) and small
enough to run on CPU.

Used by `setoptics.ball_tracker.BallTracker`. The path resolves from the
package automatically; you do not need to pass `--model` unless you want
to use your own weights.

### Performance

On the indoor benchmark used during development:

- Detection rate: > 80% of frames with a visible ball.
- ID-switch rate: < 5% across continuous play.

These are working numbers, not guarantees. Your footage may differ
significantly. If the detection rate is poor on your data, you will need
to train your own weights against your scene distribution.

### Retraining

See `docs/training.md`. The trainer is `scripts/train_volleyball_yolo.py`.
On Apple Silicon, a full run takes ~40-50 hours; you do not need a full
run to beat this checkpoint on a narrow domain.

## `rally_detector.pth` (NOT shipped)

The VideoMAE V2 + MLP head model used by
`scripts/infer_rally_detector.py` is **not included**. The script ships,
the weights do not. If you want to use VideoMAE mode you must train your
own weights against your own labeled data.

The script exits with a clear error if `--model` is missing.
