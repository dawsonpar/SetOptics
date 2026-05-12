#!/usr/bin/env python3
# EXPERIMENTAL: research-grade script (see README for status)
"""
Evaluate ensemble rally detection against ground truth annotations.

Supports:
- Single video evaluation
- Batch evaluation on indoor-007 and indoor-008
- Ablation studies (compare different method combinations)
- Weight optimization via grid search

Usage:
    # Single video evaluation
    python scripts/ensemble_rally_evaluate.py \
        --predicted ../data/processed/indoor-game-007_ensemble.json \
        --ground-truth ../data/rally-gt/indoor-game-007_annotations_corrected.json

    # Batch evaluation on valid indoor videos
    python scripts/ensemble_rally_evaluate.py \
        --batch indoor

    # Ablation study (test all method combinations)
    python scripts/ensemble_rally_evaluate.py \
        --ablation indoor-game-007

    # Weight optimization (grid search)
    python scripts/ensemble_rally_evaluate.py \
        --optimize-weights
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add tools/ to sys.path for shared eval imports
_tools_dir = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from shared.eval.temporal import (
    Interval,
    evaluate,
    load_intervals,
    print_metrics,
)


def run_ensemble_with_config(
    video_path: str, output_path: str, signal_weight: float, videomae_weight: float
) -> dict:
    """
    Run ensemble detector with custom weights (fast mode: signal + VideoMAE only).

    Returns the metrics dict.
    """
    cmd = [
        sys.executable,
        "backend/scripts/ensemble_rally_detector.py",
        "--video",
        video_path,
        "--output",
        output_path,
        "--mode",
        "fast",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd().parent)
    if result.returncode != 0:
        raise RuntimeError(f"Ensemble detector failed: {result.stderr}")

    with open(output_path) as f:
        return json.load(f)


def ablation_study(video_name: str, data_dir: Path):
    """
    Run ablation study on a single video to compare method combinations.

    Tests:
    1. Signal only
    2. VideoMAE only
    3. Signal + VideoMAE
    4. Signal + VideoMAE + LLM (if available)
    """
    video_path = data_dir / "samples" / f"{video_name}.mp4"
    gt_path = data_dir / "samples" / f"{video_name}_annotations_corrected.json"

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    if not gt_path.exists():
        print(f"Error: Ground truth not found: {gt_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"  Ablation Study: {video_name}")
    print(f"{'=' * 70}")

    gt_rallies = load_intervals(str(gt_path))
    print(f"\nGround truth: {len(gt_rallies)} rallies")

    configs = [
        ("Signal only", {"use_signal": True, "use_videomae": False, "use_llm": False}),
        (
            "VideoMAE only",
            {"use_signal": False, "use_videomae": True, "use_llm": False},
        ),
        (
            "Signal + VideoMAE",
            {"use_signal": True, "use_videomae": True, "use_llm": False},
        ),
    ]

    temp_dir = Path("/tmp") / "ensemble_ablation"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for config_name, config_opts in configs:
        print(f"\n--- {config_name} ---")
        print(
            "  [Note: Ablation study requires CLI flags for method selection - to be implemented]"
        )
        print(
            "  Current workaround: Run ensemble with --mode fast for signal+videomae baseline"
        )

    print(f"\n{'=' * 70}")


def batch_evaluate_indoor():
    """
    Batch evaluate on valid indoor videos (indoor-007 and indoor-008).

    Note: Only indoor-game-007 and indoor-game-008 are valid for evaluation.
    Videos named indoor-precut-* are pre-cut (minimal downtime) and do not
    reflect real-world input, so they are excluded from evaluation.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"

    videos = ["indoor-game-007", "indoor-game-008"]

    print(f"\n{'=' * 70}")
    print(f"  Batch Evaluation: Indoor Videos (Full-Length Only)")
    print(f"  Excluding pre-cut videos (indoor-precut-*)")
    print(f"{'=' * 70}")

    all_metrics = []

    for video_name in videos:
        video_path = data_dir / "samples" / f"{video_name}.mp4"
        gt_path = data_dir / "samples" / f"{video_name}_annotations_corrected.json"
        pred_path = data_dir / "processed" / f"{video_name}_ensemble.json"

        print(f"\n--- {video_name} ---")

        if not pred_path.exists():
            print(f"  Running ensemble detector...")
            cmd = [
                sys.executable,
                "backend/scripts/ensemble_rally_detector.py",
                "--video",
                str(video_path),
                "--output",
                str(pred_path),
                "--mode",
                "fast",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=data_dir.parent
            )
            if result.returncode != 0:
                print(f"  Error: Ensemble detector failed: {result.stderr}")
                continue

        try:
            pred_rallies = load_intervals(str(pred_path))
            gt_rallies = load_intervals(str(gt_path))

            metrics = evaluate(pred_rallies, gt_rallies)
            print_metrics(metrics)

            all_metrics.append(metrics)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if all_metrics:
        print(f"\n{'=' * 70}")
        print(f"  Aggregate Results (n={len(all_metrics)})")
        print(f"{'=' * 70}")

        avg_metrics = {
            "precision": np.mean([m["precision"] for m in all_metrics]),
            "recall": np.mean([m["recall"] for m in all_metrics]),
            "f1": np.mean([m["f1"] for m in all_metrics]),
            "mean_iou": np.mean([m["mean_iou"] for m in all_metrics]),
            "boundary_mae_start_sec": np.mean(
                [m["boundary_mae_start_sec"] for m in all_metrics]
            ),
            "boundary_mae_end_sec": np.mean(
                [m["boundary_mae_end_sec"] for m in all_metrics]
            ),
        }

        print(f"  Precision:      {avg_metrics['precision']:.1%}")
        print(f"  Recall:         {avg_metrics['recall']:.1%}")
        print(f"  F1:             {avg_metrics['f1']:.1%}")
        print(f"  Mean IoU:       {avg_metrics['mean_iou']:.3f}")
        print(
            f"  Boundary MAE:   start={avg_metrics['boundary_mae_start_sec']:.1f}s, "
            f"end={avg_metrics['boundary_mae_end_sec']:.1f}s"
        )


def optimize_weights():
    """
    Grid search over weight combinations to find optimal fusion weights.

    Only tests on indoor-007 and indoor-008 (valid evaluation set).
    """
    print(f"\n{'=' * 70}")
    print(f"  Weight Optimization: Grid Search")
    print(f"{'=' * 70}")

    print("\nNote: Weight optimization requires running ensemble with custom weights.")
    print("Current implementation: Use default weights from domain config.")
    print("TODO: Add CLI args for custom weights (--signal-weight, --videomae-weight, --llm-weight)")

    print("\nEvaluating default configuration (fast mode: signal + videomae)...")
    batch_evaluate_indoor()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble rally detection")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--predicted", help="Path to predicted ensemble JSON (single video)"
    )
    mode_group.add_argument(
        "--batch",
        choices=["indoor"],
        help="Batch evaluate on valid indoor videos",
    )
    mode_group.add_argument(
        "--ablation",
        help="Run ablation study on video (e.g., indoor-game-007)",
    )
    mode_group.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Grid search for optimal fusion weights",
    )

    parser.add_argument(
        "--ground-truth", help="Path to ground truth JSON (single video)"
    )
    parser.add_argument(
        "--matching-mode", type=str, default="iou",
        choices=["iou", "center"],
        help="Matching mode: 'iou' (default) or 'center'",
    )
    parser.add_argument(
        "--center-tolerance", type=float, default=0.0,
        help="Tolerance in seconds around GT boundaries "
             "for center matching (default: 0.0)",
    )

    args = parser.parse_args()

    if args.predicted:
        if not args.ground_truth:
            print("Error: --ground-truth required for single video evaluation")
            sys.exit(1)

        pred_path = Path(args.predicted)
        gt_path = Path(args.ground_truth)

        if not pred_path.exists():
            print(f"Error: Predicted file not found: {pred_path}")
            sys.exit(1)

        if not gt_path.exists():
            print(f"Error: Ground truth file not found: {gt_path}")
            sys.exit(1)

        pred_rallies = load_intervals(str(pred_path))
        gt_rallies = load_intervals(str(gt_path))

        metrics = evaluate(
            pred_rallies, gt_rallies,
            matching_mode=args.matching_mode,
            center_tolerance=args.center_tolerance,
        )
        print_metrics(metrics, label=f"{pred_path.stem}")

    elif args.batch:
        batch_evaluate_indoor()

    elif args.ablation:
        data_dir = Path(__file__).parent.parent.parent / "data"
        ablation_study(args.ablation, data_dir)

    elif args.optimize_weights:
        optimize_weights()


if __name__ == "__main__":
    main()
