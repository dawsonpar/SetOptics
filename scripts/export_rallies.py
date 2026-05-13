"""Export a rallies-only video from rally-detection JSON output.

Takes the output of any rally detector (`signal_rally_detector.py`,
`ensemble_rally_detector.py`, `annotate_sliding_window.py`) and the original
video, and produces an MP4 containing only the in-play segments concatenated
back to back.

Two modes:
    --mode accurate (default): re-encodes via ffmpeg filter_complex. Rally
        boundaries land exactly on the millisecond. Slower (~real-time).
    --mode fast: stream copy via ffmpeg concat demuxer. ~10x faster but
        rally boundaries snap to the nearest keyframe (drift up to ~2s
        at standard GOP sizes). Use when speed matters more than precision.

Usage:
    python scripts/export_rallies.py \\
        --segments FOOTAGE_signal.json \\
        --video FOOTAGE.mp4 \\
        --output FOOTAGE_rallies.mp4

    # Add 0.5s of padding before and after each rally
    python scripts/export_rallies.py \\
        --segments FOOTAGE_signal.json --video FOOTAGE.mp4 \\
        --pad 0.5

    # Drop rallies shorter than 5 seconds
    python scripts/export_rallies.py \\
        --segments FOOTAGE_signal.json --video FOOTAGE.mp4 \\
        --min-rally-sec 5

    # Fast mode (keyframe-snapped, ~10x speed)
    python scripts/export_rallies.py \\
        --segments FOOTAGE_signal.json --video FOOTAGE.mp4 \\
        --mode fast

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def load_rally_spans(
    segments_path: Path,
    pad_sec: float,
    min_rally_sec: float,
    video_duration_sec: float,
) -> list[tuple[float, float]]:
    """Read the detector JSON and return in-play (start_sec, end_sec) spans.

    Applies optional padding (clamped to [0, video_duration]) and the
    minimum-rally-length filter.
    """
    with open(segments_path) as f:
        data = json.load(f)
    if "segments" not in data:
        raise ValueError(
            f"{segments_path}: expected key 'segments'. Is this the output "
            f"of a rally detector? See docs/rally-detection.md."
        )

    spans: list[tuple[float, float]] = []
    for seg in data["segments"]:
        if seg.get("type") != "in-play":
            continue
        start = max(0.0, seg["start_ms"] / 1000.0 - pad_sec)
        end = min(video_duration_sec, seg["end_ms"] / 1000.0 + pad_sec)
        if end - start < min_rally_sec:
            continue
        spans.append((start, end))
    return spans


def probe_duration(video_path: Path) -> float:
    """ffprobe the video to get exact duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def export_accurate(
    video_path: Path, spans: list[tuple[float, float]], output_path: Path,
) -> None:
    """Re-encode via filter_complex. Exact boundaries, slower."""
    if not spans:
        raise ValueError("No rallies to export.")

    filters = []
    concat_inputs = []
    has_audio = _video_has_audio(video_path)
    for i, (start, end) in enumerate(spans):
        filters.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")
        if has_audio:
            filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[a{i}]")

    n = len(spans)
    if has_audio:
        concat = (
            f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
        )
        map_args = ["-map", "[outv]", "-map", "[outa]"]
    else:
        concat = f"{''.join(concat_inputs)}concat=n={n}:v=1[outv]"
        map_args = ["-map", "[outv]"]

    filter_complex = "; ".join(filters + [concat])

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-filter_complex", filter_complex,
        *map_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(output_path)]

    subprocess.run(cmd, check=True)


def export_fast(
    video_path: Path, spans: list[tuple[float, float]], output_path: Path,
) -> None:
    """Stream copy via concat demuxer. ~10x faster, keyframe-snapped."""
    if not spans:
        raise ValueError("No rallies to export.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        list_path = tmp_dir / "concat.txt"
        parts = []
        for i, (start, end) in enumerate(spans):
            part_path = tmp_dir / f"part_{i:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", str(video_path), "-c", "copy",
                    "-avoid_negative_ts", "make_zero", str(part_path),
                ],
                check=True,
                capture_output=True,
            )
            parts.append(part_path)
        with open(list_path, "w") as f:
            for p in parts:
                f.write(f"file '{p.as_posix()}'\n")
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_path), "-c", "copy", str(output_path),
            ],
            check=True,
            capture_output=True,
        )


def _video_has_audio(video_path: Path) -> bool:
    """Return True if the video has at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a rallies-only video from detection JSON."
    )
    p.add_argument(
        "--segments", required=True,
        help="Path to detector output JSON (signal/ensemble/llm).",
    )
    p.add_argument(
        "--video", required=True,
        help="Path to the original video the JSON was produced from.",
    )
    p.add_argument(
        "--output", default=None,
        help="Path to output MP4. Default: <video-stem>_rallies.mp4 "
             "next to the input video.",
    )
    p.add_argument(
        "--mode", choices=["accurate", "fast"], default="accurate",
        help="accurate: re-encode for exact boundaries (default). "
             "fast: stream-copy, ~10x faster but keyframe-snapped.",
    )
    p.add_argument(
        "--pad", type=float, default=0.0,
        help="Seconds of padding to add before/after each rally. Default 0.",
    )
    p.add_argument(
        "--min-rally-sec", type=float, default=0.0,
        help="Drop rallies shorter than this many seconds. Default 0.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print(
            "ERROR: ffmpeg / ffprobe not found on PATH. Install with "
            "`brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux).",
            file=sys.stderr,
        )
        return 1

    video_path = Path(args.video).resolve()
    segments_path = Path(args.segments).resolve()
    for p, label in [(video_path, "video"), (segments_path, "segments")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 1

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.with_name(f"{video_path.stem}_rallies.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    spans = load_rally_spans(
        segments_path, args.pad, args.min_rally_sec, duration,
    )

    if not spans:
        print(
            "No in-play segments after filtering. Try lowering "
            "--min-rally-sec or check the detector output.",
            file=sys.stderr,
        )
        return 2

    total_rally_sec = sum(end - start for start, end in spans)
    print(f"Input video:    {video_path.name} ({duration:.1f}s)")
    print(f"Rally count:    {len(spans)}")
    print(f"Rally total:    {total_rally_sec:.1f}s "
          f"({total_rally_sec / duration * 100:.1f}% of input)")
    print(f"Output mode:    {args.mode}")
    print(f"Output path:    {output_path}")
    print()

    try:
        if args.mode == "accurate":
            export_accurate(video_path, spans, output_path)
        else:
            export_fast(video_path, spans, output_path)
    except subprocess.CalledProcessError as e:
        print(f"\nffmpeg failed (exit {e.returncode}).", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr,
                  file=sys.stderr)
        return 3

    out_duration = probe_duration(output_path)
    print(f"\nDone. Exported {out_duration:.1f}s "
          f"({out_duration / duration * 100:.0f}% of input). "
          f"Skipped {duration - out_duration:.1f}s of dead time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
