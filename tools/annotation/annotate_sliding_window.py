#!/usr/bin/env python3
"""
==============================================================================
 GROUND-TRUTH rally annotation — 3-stage sliding-window pipeline
==============================================================================

 Pipeline:      Stage 1 (2-min window, 30s stride detect)
                -> Stage 2 (45s clip boundary refinement)
                -> Stage 3 (gap-fill scan of long breaks)
                All stages share the backend's RL-tuned prompts.
 Model:         `gemini-2.5-flash` by default (what the backend production
                path uses). RL-tuned over 19 iterations on this model —
                swapping models may regress quality. Override via --model
                if you are explicitly benchmarking another model.
 Config source: backend/app/services/gemini_rally_detector.py (prompts,
                stage params). This script does NOT read tools/annotation/
                config.yaml — that config belongs to annotate_fast.py.
 Measured F1:   94.6% on training game, 93.5% on held-out (commit f66d414)
 Output path:   next to each video: <video>_raw_annotations.json

 USE THIS FOR:
   - Creating or rebuilding ground-truth evaluation samples
   - Production-parity annotations (same code path the backend worker uses)
   - Cases where the extra ~19pp F1 over annotate_fast.py is worth the
     longer wall clock (~3-4x slower — multiple Gemini calls per video)

 DO NOT USE FOR:
   - Fast prompt iteration (use annotate_fast.py)
   - The annotation-ui "Run Detection" button (also uses annotate_fast.py)

 Raw outputs are unverified. Rename to `_annotations_corrected.json` only
 after human review in the annotation-ui.

 Usage:
     cd <repo-root>
     python tools/annotation/annotate_sliding_window.py \\
         data/rally-gt/indoor-game-001.mp4

     # Multiple videos with 2 parallel workers:
     python tools/annotation/annotate_sliding_window.py \\
         data/rally-gt/indoor-game-001.mp4 \\
         data/rally-gt/indoor-game-002.mp4 \\
         --parallel 2

     # Force re-annotation even if raw file exists:
     python tools/annotation/annotate_sliding_window.py \\
         data/rally-gt/indoor-game-001.mp4 --force

     # Benchmark a different model (off-the-tuned-path — quality will vary):
     python tools/annotation/annotate_sliding_window.py \\
         data/rally-gt/indoor-game-001.mp4 --model gemini-3-flash-preview
==============================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent.parent
_tools_dir = _repo_root / "tools"
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_tools_dir))

load_dotenv(_repo_root / ".env")


def build_annotation(video_path: Path, segments: list) -> dict:
    from setoptics.gemini_pipeline import get_duration_ms
    duration_ms = get_duration_ms(video_path)
    return {
        "video_metadata": {
            "path": str(video_path.resolve()),
            "fps": 0,
            "duration_seconds": round(duration_ms / 1000, 3),
            "total_frames": 0,
        },
        "segments": [
            {
                "segment_id": i + 1,
                "type": seg["type"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "start_frame": 0,
                "end_frame": 0,
                "rally_number": seg.get("rally_number"),
            }
            for i, seg in enumerate(segments)
        ],
    }


def process_video(
    video_path: Path,
    force: bool,
    api_key: str,
    model: str | None = None,
) -> dict:
    """Annotate a single video. Returns a result dict for summary reporting."""
    out_path = video_path.parent / (video_path.stem + "_raw_annotations.json")

    if out_path.exists() and not force:
        print(f"[{video_path.name}] Skipping — raw annotations already exist.")
        return {"video": video_path.name, "status": "skipped", "elapsed_min": 0, "segments": None}

    from setoptics.gemini_rally_detector import GeminiRallyDetector

    print(f"[{video_path.name}] Starting annotation...")
    t0 = time.time()
    detector_kwargs: dict = {"api_key": api_key}
    if model:
        detector_kwargs["model"] = model
    detector = GeminiRallyDetector(**detector_kwargs)
    segments, _, _ = detector.detect(video_path)
    elapsed = time.time() - t0

    in_play = [s for s in segments if s["type"] == "in-play"]
    annotation = build_annotation(video_path, segments)

    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"[{video_path.name}] Done — {len(in_play)} rallies in {elapsed/60:.1f} min → {out_path.name}")
    return {"video": video_path.name, "status": "ok", "elapsed_min": elapsed / 60, "segments": len(in_play)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_paths", type=Path, nargs="+",
                        metavar="video_path")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if raw annotations file exists")
    parser.add_argument("--parallel", type=int, default=2, metavar="N",
                        help="Max concurrent videos (default: 2; free-tier safe)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help=(
            "Override Gemini model (default: backend's gemini-2.5-flash, "
            "which the 3-stage prompts were RL-tuned against). Example: "
            "--model gemini-3-flash-preview. Use for benchmarking only — "
            "quality is untested off the tuned path."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    missing = [p for p in args.video_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: {p} not found", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in tools/shared/.env")

    results = []
    workers = min(args.parallel, len(args.video_paths))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_video, p, args.force, api_key, args.model): p
            for p in args.video_paths
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                video_path = futures[future]
                print(f"[{video_path.name}] ERROR: {exc}", file=sys.stderr)
                results.append({"video": video_path.name, "status": "error", "elapsed_min": 0, "segments": None})

    # Summary
    print(f"\n{'='*60}")
    print(f"  {'Video':<35} {'Status':<8} {'Min':>5} {'Rallies':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*5} {'-'*8}")
    for r in sorted(results, key=lambda x: x["video"]):
        segs = str(r["segments"]) if r["segments"] is not None else "—"
        print(f"  {r['video']:<35} {r['status']:<8} {r['elapsed_min']:>5.1f} {segs:>8}")
    print(f"{'='*60}\n")

    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
