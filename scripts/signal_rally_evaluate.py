# EXPERIMENTAL: research-grade script (see README for status)
"""
Evaluate signal-based rally detection against corrected annotations.

Computes rally count accuracy, temporal IoU, precision/recall/F1,
boundary MAE, and total rally time error.

Usage:
    python scripts/signal_rally_evaluate.py \
        --predicted ../data/processed/beach-game-001_signal.json \
        --ground-truth ../data/rally-gt/beach-game-001_annotations_corrected.json

    # Batch evaluate all matching pairs:
    python scripts/signal_rally_evaluate.py \
        --predicted-dir ../data/processed/ \
        --ground-truth-dir ../data/rally-gt/ \
        --suffix _signal.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add tools/ to sys.path for shared eval imports
_tools_dir = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from shared.eval.temporal import (
    Interval,
    aggregate_metrics,
    evaluate,
    load_intervals,
    print_metrics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate signal rally detection against ground truth"
    )

    # Single video evaluation
    parser.add_argument("--predicted", type=str,
                        help="Path to signal detector output JSON")
    parser.add_argument("--ground-truth", type=str,
                        help="Path to corrected annotation JSON")

    # Batch evaluation
    parser.add_argument("--predicted-dir", type=str,
                        help="Directory with signal detector outputs")
    parser.add_argument("--ground-truth-dir", type=str,
                        help="Directory with corrected annotations")
    parser.add_argument("--suffix", type=str, default="_signal.json",
                        help="Suffix for predicted files (default: "
                             "_signal.json)")

    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for rally matching")
    parser.add_argument("--matching-mode", type=str, default="iou",
                        choices=["iou", "center"],
                        help="Matching mode: 'iou' (default) or 'center'")
    parser.add_argument("--center-tolerance", type=float, default=0.0,
                        help="Tolerance in seconds around GT boundaries "
                             "for center matching (default: 0.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics to JSON file")
    args = parser.parse_args()

    all_metrics = []

    if args.predicted and args.ground_truth:
        # Single evaluation
        pred = load_intervals(args.predicted)
        gt = load_intervals(args.ground_truth)
        metrics = evaluate(
            pred, gt, args.iou_threshold,
            matching_mode=args.matching_mode,
            center_tolerance=args.center_tolerance,
        )
        label = Path(args.predicted).stem
        print_metrics(metrics, label)
        all_metrics.append({"file": label, **metrics})

    elif args.predicted_dir and args.ground_truth_dir:
        # Batch evaluation
        pred_dir = Path(args.predicted_dir)
        gt_dir = Path(args.ground_truth_dir)

        for pred_file in sorted(pred_dir.glob(f"*{args.suffix}")):
            # Derive ground truth filename
            # e.g., beach-game-001_signal.json -> beach-game-001
            base = pred_file.name.replace(args.suffix, "")
            gt_file = gt_dir / f"{base}_annotations_corrected.json"

            if not gt_file.exists():
                print(f"Skipping {pred_file.name}: "
                      f"no ground truth at {gt_file}")
                continue

            pred = load_intervals(str(pred_file))
            gt = load_intervals(str(gt_file))
            metrics = evaluate(
                pred, gt, args.iou_threshold,
                matching_mode=args.matching_mode,
                center_tolerance=args.center_tolerance,
            )
            print_metrics(metrics, base)
            all_metrics.append({"file": base, **metrics})

        if len(all_metrics) > 1:
            agg = aggregate_metrics(all_metrics)
            print(f"\n{'=' * 60}")
            print(f"  AGGREGATE ({agg['n_videos']} videos)")
            print(f"{'=' * 60}")
            print(f"  Mean precision:     {agg['precision']:.1%}")
            print(f"  Mean recall:        {agg['recall']:.1%}")
            print(f"  Mean F1:            {agg['f1']:.1%}")
            print(f"  Mean IoU:           {agg['mean_iou']:.3f}")
            print(f"  Mean count acc:     {agg['rally_count_accuracy']:.1%}")
            print(f"  Mean boundary MAE:  "
                  f"start={agg['boundary_mae_start_sec']:.1f}s, "
                  f"end={agg['boundary_mae_end_sec']:.1f}s")
            print(f"  Mean time error:    {agg['total_time_error']:.1%}")
            print(f"  Total matched:      {agg['total_matched']} | "
                  f"FP: {agg['total_false_positives']} | "
                  f"Missed: {agg['total_missed']}")
            all_metrics.append({"file": "AGGREGATE", **agg})

    else:
        print("Error: Provide either --predicted + --ground-truth, "
              "or --predicted-dir + --ground-truth-dir")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
