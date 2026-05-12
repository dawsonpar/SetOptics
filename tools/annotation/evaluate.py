"""
Evaluate LLM rally predictions against ground truth annotations.

Uses temporal IoU matching to compute precision, recall, F1, and
boundary accuracy. Core metrics come from the shared eval module.

Usage:
    cd tools/annotation
    ../../python evaluate.py \
        --predicted annotations/indoor-game-001/rally/indoor-game-001_rally_annotations.json \
        --ground-truth ../../data/rally-gt/indoor-game-001_annotations_corrected.json

    # Batch mode:
    ../../python evaluate.py \
        --predicted-dir annotations/ \
        --ground-truth-dir ../../data/rally-gt/
"""

import argparse
import json
import sys
from pathlib import Path

# Add tools/ to sys.path for shared eval imports
_tools_dir = Path(__file__).resolve().parent.parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from shared.eval.temporal import (  # noqa: F401
    Interval,
    aggregate_metrics,
    evaluate,
    load_intervals as _load_intervals,
    print_metrics,
)


def _parse_timestamp(ts: str) -> float:
    """Parse 'M:SS' or 'H:MM:SS' timestamp to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def load_llm_rallies(pred_path: str) -> list[Interval]:
    """Load rally intervals from LLM annotation output.

    Supports both LLM format (timestamp_start/timestamp_end as "M:SS")
    and signal format (start_ms/end_ms in milliseconds).
    """
    with open(pred_path) as f:
        data = json.load(f)

    rallies = []
    for seg in data["segments"]:
        if seg["type"] != "in-play":
            continue

        if "start_ms" in seg:
            start = seg["start_ms"] / 1000
            end = seg["end_ms"] / 1000
        elif "timestamp_start" in seg:
            start = _parse_timestamp(seg["timestamp_start"])
            end = _parse_timestamp(seg["timestamp_end"])
        else:
            continue

        rallies.append(Interval(start=start, end=end))
    return rallies


def load_ground_truth_rallies(gt_path: str) -> list[Interval]:
    """Load ground truth rally intervals from corrected annotations."""
    return _load_intervals(gt_path, type_filter="in-play")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM rally predictions against ground truth"
    )
    parser.add_argument("--predicted", type=str,
                        help="Path to LLM annotation output JSON")
    parser.add_argument("--ground-truth", type=str,
                        help="Path to corrected annotation JSON")
    parser.add_argument("--predicted-dir", type=str,
                        help="Directory with LLM annotation outputs")
    parser.add_argument("--ground-truth-dir", type=str,
                        help="Directory with corrected annotations")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5)")
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
        pred = load_llm_rallies(args.predicted)
        gt = load_ground_truth_rallies(args.ground_truth)
        metrics = evaluate(
            pred, gt, args.iou_threshold,
            matching_mode=args.matching_mode,
            center_tolerance=args.center_tolerance,
        )
        label = Path(args.predicted).stem
        print_metrics(metrics, label)
        all_metrics.append({"file": label, **metrics})

    elif args.predicted_dir and args.ground_truth_dir:
        pred_dir = Path(args.predicted_dir)
        gt_dir = Path(args.ground_truth_dir)

        for video_dir in sorted(pred_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            rally_dir = video_dir / "rally"
            if not rally_dir.exists():
                continue

            pred_files = list(rally_dir.glob("*_rally_annotations.json"))
            if not pred_files:
                continue
            pred_file = pred_files[0]

            base = video_dir.name
            gt_file = gt_dir / f"{base}_annotations_corrected.json"
            if not gt_file.exists():
                print(f"Skipping {base}: no ground truth at {gt_file}")
                continue

            pred = load_llm_rallies(str(pred_file))
            gt = load_ground_truth_rallies(str(gt_file))
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
            all_metrics.append({"file": "AGGREGATE", **agg})

    else:
        print("Provide --predicted + --ground-truth, or "
              "--predicted-dir + --ground-truth-dir")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
