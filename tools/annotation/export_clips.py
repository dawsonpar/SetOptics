#!/usr/bin/env python3
"""
Export training clips from corrected annotation JSON files.

Takes corrected annotations and source video, extracts overlapping clips,
and organizes them by class (in-play/break) for model training.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# Default extraction parameters (from downtime-removal.md)
DEFAULT_CLIP_DURATION = 6.0  # seconds
DEFAULT_OVERLAP = 0.5  # 50% overlap
DEFAULT_TARGET_FRAMES = 24  # frames per clip
DEFAULT_RESOLUTION = 224  # 224x224 output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training clips from corrected annotations"
    )
    parser.add_argument(
        "annotation_json",
        type=Path,
        help="Path to corrected annotation JSON file",
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to source video file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("data/rally_data"),
        help="Output directory for clips (default: data/rally_data)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=DEFAULT_CLIP_DURATION,
        help=f"Clip duration in seconds (default: {DEFAULT_CLIP_DURATION})",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Overlap ratio 0-1 (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=f"Frames per clip (default: {DEFAULT_TARGET_FRAMES})",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Output resolution WxH (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without extracting clips",
    )
    return parser.parse_args()


def load_annotation(path: Path) -> dict:
    """Load and validate annotation JSON."""
    with open(path) as f:
        data = json.load(f)

    required_keys = ["video_metadata", "segments"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    return data


def get_video_fps(video_path: Path) -> float:
    """Get video frame rate using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse fraction like "30000/1001" or "30/1"
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def generate_clip_windows(
    segment_start: float,
    segment_end: float,
    clip_duration: float,
    overlap: float,
) -> list[tuple[float, float]]:
    """
    Generate overlapping clip windows within a segment.

    Returns list of (start_time, end_time) tuples.
    """
    stride = clip_duration * (1 - overlap)
    windows = []

    current_start = segment_start
    while current_start + clip_duration <= segment_end:
        windows.append((current_start, current_start + clip_duration))
        current_start += stride

    # Handle partial clip at end if segment is long enough
    # Only include if at least half the clip duration remains
    if current_start < segment_end and (segment_end - current_start) >= clip_duration * 0.5:
        # Shift window back to fit within segment
        windows.append((segment_end - clip_duration, segment_end))

    return windows


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    target_frames: int,
    resolution: int,
    dry_run: bool = False,
) -> bool:
    """
    Extract a single clip using FFmpeg.

    Returns True if successful.
    """
    # Calculate effective fps for frame sampling
    effective_fps = target_frames / duration

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"yadif,fps={effective_fps},scale={resolution}:{resolution}",
        "-frames:v", str(target_frames),
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # No audio
        str(output_path),
    ]

    if dry_run:
        print(f"  [DRY RUN] Would extract: {output_path.name}")
        return True

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to extract {output_path.name}: {e.stderr.decode()}")
        return False


def export_clips(
    annotation: dict,
    video_path: Path,
    output_dir: Path,
    clip_duration: float,
    overlap: float,
    target_frames: int,
    resolution: int,
    dry_run: bool = False,
) -> dict:
    """
    Export all clips from annotation.

    Returns metadata dict with clip information.
    """
    video_metadata = annotation["video_metadata"]
    segments = annotation["segments"]

    # Get extraction_fps to convert frame numbers to seconds
    extraction_fps = video_metadata.get("extraction_fps", 1)
    source_fps = get_video_fps(video_path)

    # Create output directories
    in_play_dir = output_dir / "in-play"
    break_dir = output_dir / "break"

    if not dry_run:
        in_play_dir.mkdir(parents=True, exist_ok=True)
        break_dir.mkdir(parents=True, exist_ok=True)

    # Track metadata for all clips
    video_entry = {
        "filename": video_path.name,
        "path": str(video_path.absolute()),
        "source_fps": source_fps,
        "extraction_fps": extraction_fps,
    }

    metadata = {
        "clip_duration": clip_duration,
        "overlap": overlap,
        "target_frames": target_frames,
        "resolution": resolution,
        "videos": [video_entry],
        "clips": [],
    }

    clip_counter = 0
    video_name = video_path.stem

    for segment in segments:
        seg_type = segment["type"]
        if seg_type not in ("in-play", "break"):
            print(f"  [WARN] Unknown segment type: {seg_type}, skipping")
            continue

        # Handle both annotation formats:
        # - Corrected: start_ms/end_ms (milliseconds)
        # - Original: frame_start/frame_end (with extraction_fps)
        if "start_ms" in segment:
            seg_start = segment["start_ms"] / 1000.0
            seg_end = segment["end_ms"] / 1000.0
        else:
            seg_start = segment["frame_start"] / extraction_fps
            seg_end = segment["frame_end"] / extraction_fps

        seg_duration = seg_end - seg_start

        # Minimum segment duration to extract (2 seconds for very short rallies)
        min_segment_duration = 2.0

        # Skip segments that are too short
        if seg_duration < min_segment_duration:
            print(f"  [SKIP] Segment {segment.get('segment_id', '?')} too short ({seg_duration:.1f}s < {min_segment_duration}s)")
            continue

        # For short segments (< clip_duration), extract single clip at actual duration
        if seg_duration < clip_duration:
            windows = [(seg_start, seg_end)]
            actual_duration = seg_duration
            print(f"  [SHORT] Segment {segment.get('segment_id', '?')} is {seg_duration:.1f}s, extracting as single clip")
        else:
            # Generate overlapping clip windows for normal segments
            windows = generate_clip_windows(seg_start, seg_end, clip_duration, overlap)
            actual_duration = clip_duration

        # Determine output directory
        out_dir = in_play_dir if seg_type == "in-play" else break_dir

        for window_start, window_end in windows:
            clip_counter += 1
            clip_id = f"{video_name}_clip_{clip_counter:04d}"
            output_path = out_dir / f"{clip_id}.mp4"

            # Use actual_duration for short segments
            clip_dur = actual_duration if seg_duration < clip_duration else clip_duration

            success = extract_clip(
                video_path=video_path,
                output_path=output_path,
                start_time=window_start,
                duration=clip_dur,
                target_frames=target_frames,
                resolution=resolution,
                dry_run=dry_run,
            )

            if success:
                metadata["clips"].append({
                    "id": clip_id,
                    "source_video": video_path.name,
                    "source_fps": source_fps,
                    "start_time": window_start,
                    "end_time": window_end,
                    "duration": clip_dur,
                    "label": seg_type,
                    "segment_id": segment["segment_id"],
                    "rally_number": segment.get("rally_number"),
                })

    # Set clip count on the video entry
    video_entry["clip_count"] = len(metadata["clips"])

    return metadata


def main():
    args = parse_args()

    # Validate inputs
    if not args.annotation_json.exists():
        print(f"Error: Annotation file not found: {args.annotation_json}")
        sys.exit(1)

    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    print(f"Loading annotation: {args.annotation_json}")
    annotation = load_annotation(args.annotation_json)

    print(f"Source video: {args.video_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Clip settings: {args.clip_duration}s duration, {args.overlap*100:.0f}% overlap")
    print(f"Frame settings: {args.target_frames} frames at {args.resolution}x{args.resolution}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No clips will be extracted]\n")

    # Count segments by type
    in_play_count = sum(1 for s in annotation["segments"] if s["type"] == "in-play")
    break_count = sum(1 for s in annotation["segments"] if s["type"] == "break")
    print(f"Segments: {in_play_count} in-play, {break_count} break")

    print("\nExporting clips...")
    metadata = export_clips(
        annotation=annotation,
        video_path=args.video_path,
        output_dir=args.output_dir,
        clip_duration=args.clip_duration,
        overlap=args.overlap,
        target_frames=args.target_frames,
        resolution=args.resolution,
        dry_run=args.dry_run,
    )

    # Count clips by label
    in_play_clips = sum(1 for c in metadata["clips"] if c["label"] == "in-play")
    break_clips = sum(1 for c in metadata["clips"] if c["label"] == "break")
    print(f"\nExtracted {len(metadata['clips'])} clips: {in_play_clips} in-play, {break_clips} break")

    # Save metadata
    if not args.dry_run:
        metadata_path = args.output_dir / "metadata.json"

        # Merge with existing metadata if present
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing = json.load(f)

            # Merge videos list (add or update by filename)
            existing_videos = {
                v["filename"]: v for v in existing.get("videos", [])
            }
            for video in metadata["videos"]:
                existing_videos[video["filename"]] = video
            existing["videos"] = list(existing_videos.values())

            # Append new clips, avoiding duplicates by ID
            existing_ids = {c["id"] for c in existing.get("clips", [])}
            for clip in metadata["clips"]:
                if clip["id"] not in existing_ids:
                    existing.setdefault("clips", []).append(clip)

            # Update clip counts per video
            clip_counts = Counter(c["source_video"] for c in existing["clips"])
            for video in existing["videos"]:
                video["clip_count"] = clip_counts.get(video["filename"], 0)

            metadata = existing

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
