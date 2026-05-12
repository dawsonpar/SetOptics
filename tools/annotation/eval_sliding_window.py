#!/usr/bin/env python3
"""Evaluate the Gemini sliding window pipeline on a test video.

Usage:
    cd <repo-root>
    python tools/annotation/eval_sliding_window.py \
        data/rally-gt/indoor-game-007.mp4

Loads ground truth from <video>_annotations_corrected.json, runs the
3-stage pipeline, and reports F1 at IoU >= 0.5 and boundary tolerances.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent.parent
_backend_dir = _repo_root / "backend"
_tools_dir = _repo_root / "tools"
sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_tools_dir))

load_dotenv(_tools_dir / "shared" / ".env")

from shared.eval.temporal import Interval, evaluate as temporal_evaluate


def load_ground_truth(video_path: Path) -> list[Interval]:
    gt_path = video_path.parent / (
        video_path.stem + "_annotations_corrected.json"
    )
    if not gt_path.exists():
        raise FileNotFoundError(f"No ground truth at {gt_path}")
    with open(gt_path) as f:
        data = json.load(f)
    return [
        Interval(start=s["start_ms"] / 1000, end=s["end_ms"] / 1000)
        for s in data["segments"]
        if s["type"] == "in-play"
    ]


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_path>", file=sys.stderr)
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in tools/shared/.env")

    from setoptics.gemini_rally_detector import GeminiRallyDetector

    print(f"\n{'='*60}")
    print(f"  Video: {video_path.name}")
    print(f"{'='*60}")

    gt = load_ground_truth(video_path)
    print(f"  Ground truth: {len(gt)} in-play segments")

    t0 = time.time()
    detector = GeminiRallyDetector(api_key=api_key)
    segments, _, _ = detector.detect(video_path)
    elapsed = time.time() - t0

    pred = [
        Interval(start=s["start_ms"] / 1000, end=s["end_ms"] / 1000)
        for s in segments
        if s["type"] == "in-play"
    ]
    print(f"  Predicted:    {len(pred)} in-play segments")
    print(f"  Runtime:      {elapsed/60:.1f} min\n")

    # Primary metric: IoU >= 0.5
    m = temporal_evaluate(pred, gt, iou_threshold=0.5)
    tp = m["matched_rallies"]
    fp = m["false_positives"]
    fn = m["missed_rallies"]
    print(f"  IoU >= 0.5 (PRIMARY)")
    print(f"    F1={m['f1']:.1%}  P={m['precision']:.1%}  R={m['recall']:.1%}  "
          f"TP={tp}  FP={fp}  FN={fn}  mean_IoU={m['mean_iou']:.3f}")

    # Boundary tolerance tiers (center-mode with tolerance)
    for tol in [0.5, 1.0, 2.0, 5.0]:
        m2 = temporal_evaluate(
            pred, gt, matching_mode="center", center_tolerance=tol,
        )
        print(f"  ±{tol}s boundary     "
              f"F1={m2['f1']:.1%}  P={m2['precision']:.1%}  R={m2['recall']:.1%}  "
              f"TP={m2['matched_rallies']}  FP={m2['false_positives']}  FN={m2['missed_rallies']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
