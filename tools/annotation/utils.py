"""Annotation pipeline utilities."""

import json
import subprocess
from pathlib import Path


def get_video_metadata(video_path: Path) -> dict:
    """Get video metadata using FFprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError("No video stream found")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = (
        float(fps_parts[0]) / float(fps_parts[1])
        if len(fps_parts) == 2
        else 30.0
    )

    duration = float(data.get("format", {}).get("duration", 0))
    total_frames = int(duration * fps)

    return {
        "filename": video_path.name,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
    }
