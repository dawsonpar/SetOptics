"""Private helpers for the Gemini 3-stage rally detection pipeline."""

import json
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------
class RallyWindow(BaseModel):
    start_ms: int
    end_ms: int


class ClipRallies(BaseModel):
    rallies: list[RallyWindow]


class RallyBoundary(BaseModel):
    start_offset_ms: int
    end_offset_ms: int


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------
def _video_arg(video_path: Path | str) -> str:
    """Pass-through for strings, coerce Path — preserves http(s):// URLs.

    ``str(Path("https://..."))`` collapses to ``https:/...`` (single slash),
    which ffmpeg rejects. Callers that want to stream from a GCS signed URL
    pass the URL as a plain string; local paths stay as Path.
    """
    return video_path if isinstance(video_path, str) else str(video_path)


def _is_url(video_path: Path | str) -> bool:
    return isinstance(video_path, str) and video_path.startswith(("http://", "https://"))


def get_duration_ms(video_path: Path | str) -> int:
    """Return video duration in milliseconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            _video_arg(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(float(result.stdout.strip()) * 1000)


def extract_clip(
    video_path: Path | str,
    start_s: float,
    duration_s: float,
    output_path: Path,
) -> None:
    cmd: list[str] = ["ffmpeg", "-y"]
    if _is_url(video_path):
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "10",
        ]
    cmd += [
        "-ss", str(start_s),
        "-i", _video_arg(video_path),
        "-t", str(duration_s),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        "-loglevel", "error",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")


def upload_and_wait(client, clip_path: Path):
    """Upload clip to Files API, poll until ACTIVE, return File object."""
    clip_file = client.files.upload(
        file=str(clip_path),
        config={"display_name": clip_path.stem},
    )
    for _ in range(60):
        if clip_file.state.name != "PROCESSING":
            break
        time.sleep(3)
        clip_file = client.files.get(name=clip_file.name)
    if clip_file.state.name != "ACTIVE":
        raise RuntimeError(f"Upload stuck in state: {clip_file.state.name}")
    return clip_file


def delete_file(client, file_name: str) -> None:
    try:
        client.files.delete(name=file_name)
    except Exception:
        pass


def compute_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def merge_windows(
    windows: list[tuple[int, int]],
    gap_ms: int = 5_000,
    min_dur_ms: int = 2_000,
) -> list[tuple[int, int]]:
    """Sort, merge overlapping/nearby windows, drop too-short segments."""
    if not windows:
        return []
    sorted_w = sorted(windows, key=lambda x: x[0])
    merged = [sorted_w[0]]
    for s, e in sorted_w[1:]:
        if s <= merged[-1][1] + gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) >= min_dur_ms]


def build_timeline(
    in_play: list[tuple[int, int]],
    duration_ms: int,
) -> list[dict]:
    """Build contiguous [0, duration_ms] segments from in-play intervals."""
    sorted_segs = sorted(in_play, key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    for s, e in sorted_segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    result: list[dict] = []
    cursor = 0
    for s, e in merged:
        s = max(cursor, min(s, duration_ms))
        e = max(s, min(e, duration_ms))
        if s > cursor:
            result.append({"start_ms": cursor, "end_ms": s, "type": "break"})
        if e > s:
            result.append({"start_ms": s, "end_ms": e, "type": "in-play"})
            cursor = e
    if cursor < duration_ms:
        result.append({"start_ms": cursor, "end_ms": duration_ms, "type": "break"})
    return result
