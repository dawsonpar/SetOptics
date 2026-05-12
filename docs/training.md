# Training your own models

This repo ships a working YOLO ball detector
(`models/volleyball_yolo26n.pt`) but not the training data. To improve
detection on your own footage, you have two options.

## Option 1: Retrain the ball detector

Use `scripts/train_volleyball_yolo.py`. The trainer expects a standard
YOLO dataset structure:

```
your-dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

Run:

```bash
python scripts/train_volleyball_yolo.py \
    --data path/to/your-dataset/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --device mps   # or cuda:0, or cpu
```

The trainer writes checkpoints under `runs/detect/<run-name>/weights/`.
Copy the `best.pt` into `models/` and either rename it
`volleyball_yolo26n.pt` or pass `--model` to the ball tracker.

### Wall clock

On an Apple M-series, a full 100-epoch run on ~19k images takes ~40-50
hours. For domain adaptation (fine-tuning on a few hundred images of
your scene) you can usually stop after 20-30 epochs.

## Option 2: Train the VideoMAE rally classifier

`scripts/infer_rally_detector.py` runs a VideoMAE V2 backbone + a small
MLP head. The MLP training script is not included in this open-source
release because it is tightly coupled to the internal training data
pipeline. The architecture is documented in the inference script.

If you want to train your own:

1. Export per-rally clips with `tools/annotation/export_clips.py`.
2. Extract VideoMAE V2 embeddings for each clip (any of the published
   VideoMAE V2 checkpoints will do).
3. Train a 2-class MLP (`in-play` vs `out-of-play`) on the embeddings.
4. Save as `rally_detector.pth`. The MLP definition in
   `scripts/infer_rally_detector.py` is the contract.

This is more involved than the YOLO path; budget a weekend if you want
to do it cleanly.
