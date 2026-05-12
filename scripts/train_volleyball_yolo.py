# OFFLINE TRAINING — run manually, not called by the production API
"""
Training script for volleyball YOLO model.

Supports both YOLOv8 and YOLO26 with configurable model sizes.

Usage:
    cd backend && python scripts/train_volleyball_yolo.py --epochs 100 --batch 16
    cd backend && python scripts/train_volleyball_yolo.py --version yolo26 --size n
    cd backend && python scripts/train_volleyball_yolo.py --version yolo26 --size s
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO volleyball detector")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--version",
        type=str,
        default="yolov8",
        choices=["yolov8", "yolo26"],
        help="YOLO version (yolov8 or yolo26)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Override base model path (optional, computed from version+size if empty)",
    )
    parser.add_argument("--device", type=str, default="", help="Device (empty=auto, mps, cuda, cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Compute base model name from version and size if not overridden
    if args.model:
        base_model = args.model
        output_name = Path(args.model).stem.replace(".", "_")
    else:
        base_model = f"{args.version}{args.size}.pt"
        output_name = f"volleyball_{args.version}{args.size}"

    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_yaml = base_dir / "data" / "external" / "ball-detection" / "volleyball-v2-yolo26" / "data.yaml"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    if not data_yaml.exists():
        print(f"Error: Dataset not found at {data_yaml}")
        sys.exit(1)

    print(f"Training volleyball detector")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  YOLO version: {args.version}")
    print(f"  Model size: {args.size}")
    print(f"  Base model: {base_model}")
    print(f"  Output name: {output_name}")
    print(f"  Output dir: {models_dir}")

    # Determine device
    device = args.device
    if not device:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"  Device: {device}")

    # Load base model
    model = YOLO(base_model)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(models_dir),
        name=output_name,
        exist_ok=True,
        resume=args.resume,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Export best model to models directory
    best_model_path = models_dir / output_name / "weights" / "best.pt"
    output_path = models_dir / f"{output_name}.pt"

    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, output_path)
        print(f"\nBest model saved to: {output_path}")

        # Print validation metrics
        print("\nTraining complete!")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")


if __name__ == "__main__":
    main()
