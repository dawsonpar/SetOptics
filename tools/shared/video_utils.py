"""Shared video manipulation utilities (ffmpeg/ffprobe wrappers)."""

import subprocess
from pathlib import Path


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def reencode_video(
    video_path: Path,
    crf: int,
    scale: str | None,
    output_dir: Path,
) -> Path:
    """Re-encode a video with given CRF and optional scale.

    Returns the output path. Skips re-encoding if output already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_tag = scale.replace(":", "x") if scale else "original"
    out_path = output_dir / f"{video_path.stem}_crf{crf}_{scale_tag}.mp4"
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-crf", str(crf),
    ]
    if scale:
        cmd += ["-vf", f"scale={scale}"]
    cmd += ["-c:a", "aac", "-b:a", "64k", str(out_path)]

    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def _trim_video(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    output_dir: Path,
) -> Path:
    """Trim a video segment using stream copy (fast, no re-encode)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_path = (
        output_dir
        / f"{stem}_trim_{int(start_sec)}s_{int(duration_sec)}s.mp4"
    )
    if out_path.exists():
        return out_path

    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration_sec),
            "-c", "copy",
            str(out_path),
        ],
        capture_output=True, check=True,
    )
    return out_path


def chunk_video(
    video_path: Path,
    chunk_sec: int,
    output_dir: Path,
    min_chunk_sec: float = 30.0,
) -> list[tuple[Path, float]]:
    """Split video into N-second chunks.

    If the final chunk would be shorter than min_chunk_sec, it is
    merged into the previous chunk to avoid hallucinated output from
    very short video clips.

    Returns list of (chunk_path, start_sec) pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)

    # Calculate chunk boundaries first
    boundaries = []
    start = 0.0
    while start < duration:
        boundaries.append(start)
        start += chunk_sec

    # If the last chunk would be too short, merge it into the previous
    if len(boundaries) > 1:
        last_chunk_dur = duration - boundaries[-1]
        if last_chunk_dur < min_chunk_sec:
            boundaries.pop()

    # Create chunk files
    chunks = []
    for s in boundaries:
        chunk_dur = min(chunk_sec, duration - s)
        # Last boundary gets the remaining duration
        if s == boundaries[-1]:
            chunk_dur = duration - s
        chunk_path = _trim_video(video_path, s, chunk_dur, output_dir)
        chunks.append((chunk_path, s))

    return chunks
