#!/usr/bin/env python3
"""
==============================================================================
 FAST rally annotation — single-pass Gemini Files API
==============================================================================

 Pipeline:      re-encode -> chunk -> upload -> one prompt per chunk -> merge
 Model:         from config.yaml (default `gemini-3-flash-preview`)
 Config:        tools/annotation/config.yaml (ONLY this script reads it)
 Measured F1:   ~75% avg on indoor volleyball (commit cf2766f)
 Output path:   tools/annotation/annotations/<video>/rally/
                    <video>_rally_annotations.json

 USE THIS WHEN:
   - Rapidly iterating on prompts or model selection
   - The annotation-ui's "Run Gemini Detection" button (the electron
     handler shells out to this script via `--video <name>`)
   - Comparing providers / prompt variants

 DO NOT USE FOR GROUND TRUTH: it undershoots the production pipeline by
 ~20pp F1. For GT sample creation, use `annotate_sliding_window.py`.

 Usage:
     cd tools/annotation
     ../../python annotate_fast.py --video indoor-game-001
     ../../python annotate_fast.py --all
     ../../python annotate_fast.py --video indoor-game-001 --evaluate
==============================================================================
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

_script_dir = Path(__file__).parent
load_dotenv(_script_dir / ".env")

_tools_dir = _script_dir.parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

_project_root = _tools_dir.parent

from shared.providers import get_provider
from shared.video_utils import chunk_video, get_video_duration, reencode_video
from prompts import build_video_rally_prompt


# -------------------------------------------------------------------
# Timestamp conversion
# -------------------------------------------------------------------

def _parse_timestamp(ts: str) -> float:
    """Parse MM:SS or H:MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def convert_to_annotation_format(raw_result: dict) -> dict:
    """Convert MM:SS timestamps to start_ms/end_ms format."""
    segments = []
    for seg in raw_result.get("segments", []):
        start_sec = _parse_timestamp(seg["timestamp_start"])
        end_sec = _parse_timestamp(seg["timestamp_end"])
        segments.append({
            "type": seg["type"],
            "start_ms": int(start_sec * 1000),
            "end_ms": int(end_sec * 1000),
            "rally_number": seg.get("rally_number"),
            "description": seg.get("description", ""),
        })
    return {"segments": segments}


# -------------------------------------------------------------------
# Chunk merging
# -------------------------------------------------------------------

def merge_chunk_results(
    chunk_results: list[tuple[dict, float]],
) -> dict:
    """Merge segment results from multiple chunks.

    Args:
        chunk_results: List of (annotation_dict, chunk_start_ms) pairs.
    """
    all_segments = []
    rally_offset = 0

    for result, chunk_start_ms in chunk_results:
        for seg in result.get("segments", []):
            adjusted = dict(seg)
            adjusted["start_ms"] = seg["start_ms"] + int(chunk_start_ms)
            adjusted["end_ms"] = seg["end_ms"] + int(chunk_start_ms)
            if adjusted.get("rally_number") is not None:
                adjusted["rally_number"] += rally_offset
            all_segments.append(adjusted)

        rally_nums = [
            s.get("rally_number", 0) or 0
            for s in result.get("segments", [])
        ]
        if rally_nums:
            rally_offset = max(rally_nums)

    merged = _stitch_adjacent_segments(all_segments)
    return {"segments": merged}


def _stitch_adjacent_segments(segments: list[dict]) -> list[dict]:
    """Merge adjacent segments of the same type at chunk boundaries."""
    if not segments:
        return []

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        gap = abs(seg["start_ms"] - prev["end_ms"])
        if prev["type"] == seg["type"] and gap <= 2000:
            prev["end_ms"] = seg["end_ms"]
            if seg.get("description"):
                prev["description"] = (
                    prev.get("description", "") + "; " + seg["description"]
                )
        else:
            result.append(seg)
    return result


# -------------------------------------------------------------------
# Core pipeline
# -------------------------------------------------------------------

def _retry_wait(attempt: int, error_str: str) -> int:
    """Return seconds to wait before retrying."""
    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
    if is_rate_limit:
        return 60 * attempt
    return 5 * (2 ** (attempt - 1))


def process_video_chunk(
    provider,
    chunk_path: Path,
    prompt: str,
    max_retries: int = 3,
) -> tuple[dict | None, str]:
    """Upload chunk -> prompt -> parse -> cleanup. Returns (result, raw)."""
    for attempt in range(1, max_retries + 1):
        try:
            uploaded = provider.upload_video(chunk_path)
            try:
                result, raw_text = provider.generate_with_video(
                    uploaded, prompt, parse_json=True,
                )
                return result, raw_text
            finally:
                provider.delete_file(uploaded.name)
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries:
                wait = _retry_wait(attempt, error_str)
                print(f"  Attempt {attempt} failed: {error_str}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {error_str}")
                return None, error_str


def annotate_video(
    video_path: Path,
    config: dict,
    output_dir: Path,
    max_retries: int = 3,
) -> dict:
    """Run the full video annotation pipeline.

    Steps:
    1. Re-encode video (crf32, 720p)
    2. Chunk into segments
    3. For each chunk: upload -> prompt -> parse -> cleanup
    4. Merge chunk results
    """
    video_name = video_path.stem
    work_dir = output_dir / video_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Re-encode
    enc = config.get("encoding", {})
    crf = enc.get("crf", 32)
    scale = enc.get("scale", "1280:720")
    print(f"Re-encoding video (crf={crf}, scale={scale})...")
    encoded_path = reencode_video(
        video_path, crf, scale, work_dir / "encoded",
    )
    size_mb = encoded_path.stat().st_size / 1e6
    print(f"  Encoded: {encoded_path.name} ({size_mb:.1f} MB)")

    # Chunk
    chunk_sec = config.get("chunking", {}).get("chunk_duration_sec", 600)
    print(f"Chunking into {chunk_sec // 60}-minute segments...")
    chunks = chunk_video(encoded_path, chunk_sec, work_dir / "chunks")
    print(f"  Created {len(chunks)} chunks")

    # Initialize provider
    provider_name = config.get("provider", "google")
    provider = get_provider(provider_name, config)
    print(f"Provider: {provider_name}/{provider.model}")

    # Process chunks
    prompt = build_video_rally_prompt()
    chunk_results = []

    for i, (chunk_path, start_sec) in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)} "
              f"(start={start_sec:.0f}s)...")
        t0 = time.time()

        result, raw_text = process_video_chunk(
            provider, chunk_path, prompt, max_retries,
        )

        elapsed = time.time() - t0

        # Save raw response
        raw_dir = work_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"chunk_{i:02d}_raw.txt"
        raw_path.write_text(raw_text or "")

        if result:
            converted = convert_to_annotation_format(result)
            chunk_results.append((converted, start_sec * 1000))
            n_rallies = sum(
                1 for s in converted["segments"]
                if s["type"] == "in-play"
            )
            print(f"  {n_rallies} rallies detected in {elapsed:.1f}s")
        else:
            print(f"  Chunk {i + 1} failed — skipping")

    if not chunk_results:
        raise RuntimeError("All chunks failed — no results to merge")

    # Merge
    print(f"\nMerging {len(chunk_results)} chunk results...")
    merged = merge_chunk_results(chunk_results)

    # Add segment IDs
    for i, seg in enumerate(merged["segments"], 1):
        seg["segment_id"] = i

    # Add video metadata
    duration_sec = get_video_duration(video_path)
    merged["video_metadata"] = {
        "filename": video_path.name,
        "duration_seconds": duration_sec,
    }

    # Add annotation metadata
    merged["annotation_metadata"] = {
        "annotator": f"{provider_name}/{provider.model}",
        "annotation_date": datetime.now().isoformat(),
        "schema_version": "2.0.0",
        "pipeline": "video",
        "encoding_crf": crf,
        "encoding_scale": scale,
        "chunk_duration_sec": chunk_sec,
    }

    rally_count = sum(
        1 for s in merged["segments"] if s["type"] == "in-play"
    )
    print(f"Total rallies: {rally_count}")
    print(f"Total segments: {len(merged['segments'])}")

    return merged


# -------------------------------------------------------------------
# Video finder
# -------------------------------------------------------------------

def find_video(config: dict, video_name: str) -> Path:
    """Locate a video file, checking samples and rally-edits directories."""
    samples_dir = Path(config["samples_dir"])
    if not samples_dir.is_absolute():
        samples_dir = _script_dir / samples_dir

    search_dirs = [samples_dir, samples_dir.parent / "rally-edits"]

    for directory in search_dirs:
        for ext in (".mp4", ".mov", ".MOV", ".avi"):
            path = directory / f"{video_name}{ext}"
            if path.exists():
                return path

    raise FileNotFoundError(
        f"Video '{video_name}' not found in {samples_dir} or {search_dirs[1]}"
    )


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Video-based rally annotation via Gemini Files API"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Video name (e.g. indoor-game-001)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all eval videos from config",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate against ground truth after annotation",
    )
    parser.add_argument(
        "--config", type=Path,
        default=_script_dir / "config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per chunk on failure (default: 3)",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(
        config.get("output", {}).get("directory", "./annotations")
    )
    if not output_dir.is_absolute():
        output_dir = _script_dir / output_dir

    video_names = []
    if args.video:
        video_names = [args.video]
    elif args.all:
        video_names = config.get("eval_videos", [])
    else:
        parser.error("Provide --video or --all")

    for video_name in video_names:
        print(f"\n{'=' * 60}")
        print(f"  Annotating: {video_name}")
        print(f"{'=' * 60}\n")

        video_path = find_video(config, video_name)
        print(f"Video: {video_path}")

        merged = annotate_video(
            video_path, config, output_dir, args.max_retries,
        )

        # Save output
        out_path = output_dir / video_name / "rally"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{video_name}_rally_annotations.json"
        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nSaved: {out_file}")

        if args.evaluate:
            _run_evaluation(config, video_name, merged)


def _run_evaluation(config: dict, video_name: str, predicted: dict):
    """Evaluate predicted segments against ground truth."""
    from shared.eval.temporal import (
        Interval, evaluate, print_metrics, load_intervals,
    )

    samples_dir = Path(config["samples_dir"])
    if not samples_dir.is_absolute():
        samples_dir = _script_dir / samples_dir

    gt_path = samples_dir / f"{video_name}_annotations_corrected.json"
    if not gt_path.exists():
        print(f"  No ground truth found at {gt_path}")
        return

    pred_intervals = [
        Interval(start=s["start_ms"] / 1000, end=s["end_ms"] / 1000)
        for s in predicted.get("segments", [])
        if s["type"] == "in-play"
    ]

    gt_intervals = load_intervals(str(gt_path))
    metrics = evaluate(pred_intervals, gt_intervals)
    print_metrics(metrics, video_name)


if __name__ == "__main__":
    main()
