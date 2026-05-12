# EXPERIMENTAL: research-grade script (see README for status)
"""
Inference script for volleyball rally detection.

Processes a full-length video through the trained VideoMAE V2 + MLP pipeline
to detect rally segments.

Pipeline:
  Full video → Sliding window (6s, 3s step) → VideoMAE V2 features → MLP
    → Temporal smoothing → Threshold → Min duration filter → Segments JSON
    → (optional) FFmpeg trim → Rallies-only video

Uses PyAV for single-pass sequential video decoding to avoid the O(windows *
frames_per_window) random-seek bottleneck of cv2.VideoCapture.set().

Usage:
    python scripts/infer_rally_detector.py \
        --video ../data/videos/full-match.mp4 \
        --model ../models/rally_detector.pth \
        --output ../data/processed/segments.json \
        --trim
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP head (must match training definition)
# ---------------------------------------------------------------------------

class RallyMLP(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> tuple[float, float, int]:
    """Get video metadata using PyAV.

    Returns (duration_sec, fps, total_frames).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    total_frames = stream.frames
    if total_frames == 0:
        # Fallback: estimate from duration
        duration = float(stream.duration * stream.time_base)
        total_frames = int(duration * fps)
    else:
        duration = total_frames / fps
    container.close()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    return duration, fps, total_frames


def _build_window_plan(duration: float, fps: float,
                       window_sec: float, step_sec: float,
                       num_frames: int = 16):
    """Pre-compute which global frame indices each sliding window needs.

    Returns:
        windows: list of dicts, each with keys:
            - start_time: float
            - frame_indices: list[int] (global frame numbers, sorted)
        frame_to_windows: dict mapping global_frame_idx →
            list of (window_idx, slot_idx) tuples
        all_needed_frames: sorted list of all unique frame indices needed
    """
    windows = []
    t = 0.0
    while t + window_sec <= duration:
        start_frame = int(t * fps)
        end_frame = int((t + window_sec) * fps)
        window_length = end_frame - start_frame

        if window_length <= 0:
            t += step_sec
            continue

        local_indices = np.linspace(
            0, window_length - 1, num_frames, dtype=int)
        global_indices = (local_indices + start_frame).tolist()

        windows.append({
            "start_time": t,
            "frame_indices": global_indices,
        })
        t += step_sec

    # Build reverse lookup: frame_idx → [(window_idx, slot_idx), ...]
    frame_to_windows = {}
    for win_idx, win in enumerate(windows):
        for slot_idx, frame_idx in enumerate(win["frame_indices"]):
            if frame_idx not in frame_to_windows:
                frame_to_windows[frame_idx] = []
            frame_to_windows[frame_idx].append((win_idx, slot_idx))

    all_needed_frames = sorted(frame_to_windows.keys())
    return windows, frame_to_windows, all_needed_frames


# ---------------------------------------------------------------------------
# Sliding window inference (single-pass sequential decode)
# ---------------------------------------------------------------------------

def _process_window(frames: np.ndarray, backbone, processor, mlp,
                    device: str) -> float:
    """Run a single 16-frame window through backbone + MLP.

    Args:
        frames: (16, 224, 224, 3) RGB uint8 array.

    Returns:
        rally_prob: float probability that this window is in-play.
    """
    frame_list = [frames[i] for i in range(frames.shape[0])]
    inputs = processor(frame_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = backbone(**inputs)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        logits = mlp(features.cpu())
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()


def run_inference(video_path: str, backbone, processor, mlp, device: str,
                  window_sec: float = 6.0, step_sec: float = 3.0):
    """Run sliding window inference via single-pass sequential decode.

    Decodes the video once with PyAV and distributes frames to overlapping
    windows using a pre-computed lookup table. Each frame is decoded and
    resized exactly once, eliminating the per-frame random-seek bottleneck.

    Returns list of dicts with keys: start, end, rally_prob.
    """
    duration, fps, total_frames = get_video_info(video_path)
    print(f"Video duration: {duration:.1f}s ({total_frames} frames "
          f"@ {fps:.1f} fps)")

    num_frames_per_window = 16
    windows, frame_to_windows, all_needed = _build_window_plan(
        duration, fps, window_sec, step_sec, num_frames_per_window)

    if not windows:
        print("  No windows to process (video shorter than window size)")
        return []

    print(f"  Windows: {len(windows)}, unique frames needed: "
          f"{len(all_needed)}")

    needed_set = set(all_needed)

    # Pre-allocate frame buffers for each window
    # Each slot holds a (224, 224, 3) uint8 array
    window_buffers = [
        [None] * num_frames_per_window for _ in range(len(windows))
    ]
    # Track how many frames each window has received
    window_fill_count = [0] * len(windows)

    # Results array indexed by window_idx
    predictions = [None] * len(windows)

    # Track which windows are ready to process (in order)
    next_window_to_emit = 0
    windows_completed = 0

    # Single-pass decode
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    decode_start = time.time()
    frame_idx = 0

    for frame in container.decode(stream):
        if frame_idx in needed_set:
            # Convert to RGB numpy and resize once
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (224, 224))

            # Distribute to all windows that need this frame
            for win_idx, slot_idx in frame_to_windows[frame_idx]:
                window_buffers[win_idx][slot_idx] = img
                window_fill_count[win_idx] += 1

            # Process any completed windows (in order)
            while (next_window_to_emit < len(windows)
                   and window_fill_count[next_window_to_emit]
                   == num_frames_per_window):
                win = windows[next_window_to_emit]
                frames_array = np.stack(
                    window_buffers[next_window_to_emit])

                rally_prob = _process_window(
                    frames_array, backbone, processor, mlp, device)

                predictions[next_window_to_emit] = {
                    "start": win["start_time"],
                    "end": win["start_time"] + window_sec,
                    "rally_prob": rally_prob,
                }

                # Free buffer memory for completed window
                window_buffers[next_window_to_emit] = None
                windows_completed += 1

                if windows_completed % 50 == 0:
                    elapsed = time.time() - decode_start
                    print(
                        f"  Processed {windows_completed}/{len(windows)} "
                        f"windows ({win['start_time']:.0f}/"
                        f"{duration:.0f}s) [{elapsed:.0f}s elapsed]")

                next_window_to_emit += 1

        frame_idx += 1

        # Early exit once all windows are done
        if next_window_to_emit >= len(windows):
            break

    container.close()

    # Handle any remaining windows that didn't fill completely
    # (e.g., if the video is slightly shorter than expected)
    while next_window_to_emit < len(windows):
        win = windows[next_window_to_emit]
        buf = window_buffers[next_window_to_emit]

        # Fill missing slots with the last available frame
        last_good = None
        for i in range(num_frames_per_window):
            if buf[i] is not None:
                last_good = buf[i]
            elif last_good is not None:
                buf[i] = last_good.copy()

        if any(f is None for f in buf):
            print(f"  Warning: window {next_window_to_emit} at "
                  f"t={win['start_time']:.1f}s has unfilled frames, "
                  f"skipping")
            next_window_to_emit += 1
            continue

        frames_array = np.stack(buf)
        rally_prob = _process_window(
            frames_array, backbone, processor, mlp, device)

        predictions[next_window_to_emit] = {
            "start": win["start_time"],
            "end": win["start_time"] + window_sec,
            "rally_prob": rally_prob,
        }
        window_buffers[next_window_to_emit] = None
        windows_completed += 1
        next_window_to_emit += 1

    # Filter out any None entries (skipped windows)
    predictions = [p for p in predictions if p is not None]

    elapsed = time.time() - decode_start
    print(f"  Total windows: {windows_completed}, "
          f"decode+inference: {elapsed:.1f}s")
    return predictions


# ---------------------------------------------------------------------------
# Temporal smoothing and segment extraction
# ---------------------------------------------------------------------------

def smooth_predictions(predictions: list[dict],
                       window_size: int = 3) -> list[dict]:
    """Apply rolling average smoothing to rally probabilities."""
    probs = [p["rally_prob"] for p in predictions]
    smoothed = []

    for i in range(len(probs)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(probs), i + window_size // 2 + 1)
        avg_prob = np.mean(probs[start_idx:end_idx])
        smoothed.append({
            **predictions[i],
            "rally_prob_raw": predictions[i]["rally_prob"],
            "rally_prob": float(avg_prob),
        })

    return smoothed


def extract_segments(predictions: list[dict], threshold: float = 0.5,
                     min_rally_sec: float = 3.0,
                     min_break_sec: float = 5.0) -> list[dict]:
    """Convert per-window predictions to contiguous segments."""
    if not predictions:
        return []

    # Classify each window
    segments = []
    current_label = "rally" if predictions[0]["rally_prob"] >= threshold \
        else "break"
    current_start = predictions[0]["start"]
    current_probs = [predictions[0]["rally_prob"]]

    for i in range(1, len(predictions)):
        label = "rally" if predictions[i]["rally_prob"] >= threshold \
            else "break"

        if label != current_label:
            segments.append({
                "start": current_start,
                "end": predictions[i - 1]["end"],
                "label": current_label,
                "confidence": float(np.mean(current_probs)),
            })
            current_label = label
            current_start = predictions[i]["start"]
            current_probs = [predictions[i]["rally_prob"]]
        else:
            current_probs.append(predictions[i]["rally_prob"])

    # Final segment
    segments.append({
        "start": current_start,
        "end": predictions[-1]["end"],
        "label": current_label,
        "confidence": float(np.mean(current_probs)),
    })

    # Apply minimum duration filters by merging short segments
    filtered = _apply_min_duration(segments, min_rally_sec, min_break_sec)
    return filtered


def _apply_min_duration(segments: list[dict], min_rally_sec: float,
                        min_break_sec: float) -> list[dict]:
    """Merge segments shorter than minimum duration into neighbors."""
    if len(segments) <= 1:
        return segments

    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_duration = seg["end"] - seg["start"]
            min_dur = min_rally_sec if seg["label"] == "rally" \
                else min_break_sec

            if seg_duration < min_dur and len(new_segments) > 0:
                # Merge into previous segment
                new_segments[-1]["end"] = seg["end"]
                changed = True
            else:
                new_segments.append(seg.copy())
            i += 1

        segments = new_segments

    # Merge consecutive segments with same label
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = (merged[-1]["confidence"] +
                                        seg["confidence"]) / 2
        else:
            merged.append(seg.copy())

    return merged


# ---------------------------------------------------------------------------
# FFmpeg trimming
# ---------------------------------------------------------------------------

def trim_to_rallies(video_path: str, segments: list[dict],
                    output_path: str):
    """Concatenate rally segments into a single output video using FFmpeg."""
    rally_segments = [s for s in segments if s["label"] == "rally"]
    if not rally_segments:
        print("No rally segments found. Skipping trim.")
        return

    print(f"Assembling {len(rally_segments)} rally segments...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        part_files = []

        for i, seg in enumerate(rally_segments):
            part_path = tmp_dir / f"part_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start"]),
                "-i", video_path,
                "-t", str(seg["end"] - seg["start"]),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(part_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            part_files.append(part_path)

        # Write concat list
        concat_file = tmp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for part_path in part_files:
                f.write(f"file '{part_path}'\n")

        # Concatenate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    total_rally = sum(s["end"] - s["start"] for s in rally_segments)
    print(f"Trimmed video saved to {output_path}")
    print(f"  Rally time: {total_rally:.1f}s "
          f"({len(rally_segments)} segments)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Infer rally segments from a volleyball video")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained rally_detector.pth")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for output segments JSON")
    parser.add_argument("--trim", action="store_true",
                        help="Also produce a trimmed rallies-only video")
    parser.add_argument("--trim-output", type=str, default="",
                        help="Path for trimmed video (default: "
                             "<output>_rallies.mp4)")
    parser.add_argument("--window", type=float, default=6.0,
                        help="Sliding window size in seconds")
    parser.add_argument("--step", type=float, default=3.0,
                        help="Sliding window step in seconds")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Rally probability threshold")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Smoothing window size (number of predictions)")
    parser.add_argument("--min-rally", type=float, default=3.0,
                        help="Minimum rally duration in seconds")
    parser.add_argument("--min-break", type=float, default=5.0,
                        help="Minimum break duration in seconds")
    parser.add_argument("--device", type=str, default="",
                        help="Device (auto/cpu/mps/cuda)")
    parser.add_argument("--save-raw", type=str, default="",
                        help="Save raw per-window predictions to JSON")
    parser.add_argument("--load-raw", type=str, default="",
                        help="Load raw predictions from JSON (skip inference)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Device
    if args.device:
        device = args.device
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if args.load_raw:
        # Skip inference, load cached raw predictions
        print(f"Loading raw predictions from {args.load_raw}")
        with open(args.load_raw) as f:
            raw_predictions = json.load(f)
        print(f"  Loaded {len(raw_predictions)} window predictions")
    else:
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=True)

        mlp = RallyMLP(
            input_dim=checkpoint.get("input_dim", 768),
            hidden_dim=checkpoint.get("hidden_dim", 256),
            num_classes=checkpoint.get("num_classes", 2),
            dropout=checkpoint.get("dropout", 0.3),
        )
        mlp.load_state_dict(checkpoint["model_state_dict"])
        mlp.eval()

        # Load backbone
        from transformers import VideoMAEModel, VideoMAEImageProcessor

        backbone_name = checkpoint.get(
            "backbone", "MCG-NJU/videomae-base-finetuned-kinetics")
        print(f"Loading backbone: {backbone_name}")
        processor = VideoMAEImageProcessor.from_pretrained(backbone_name)
        backbone = VideoMAEModel.from_pretrained(backbone_name)
        backbone = backbone.to(device)
        backbone.eval()

        # Run inference
        print(f"\nProcessing: {video_path}")
        start = time.time()
        raw_predictions = run_inference(
            str(video_path), backbone, processor, mlp, device,
            window_sec=args.window, step_sec=args.step,
        )
        print(f"Inference time: {time.time() - start:.1f}s")

        # Save raw predictions if requested
        if args.save_raw:
            raw_path = Path(args.save_raw)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w") as f:
                json.dump(raw_predictions, f, indent=2)
            print(f"Raw predictions saved to {raw_path}")

    # Smooth and extract segments
    smoothed = smooth_predictions(raw_predictions, args.smooth_window)
    segments = extract_segments(
        smoothed,
        threshold=args.threshold,
        min_rally_sec=args.min_rally,
        min_break_sec=args.min_break,
    )

    # Summary
    rally_segments = [s for s in segments if s["label"] == "rally"]
    break_segments = [s for s in segments if s["label"] == "break"]
    total_rally = sum(s["end"] - s["start"] for s in rally_segments)
    total_break = sum(s["end"] - s["start"] for s in break_segments)

    print(f"\nResults:")
    print(f"  Rally segments: {len(rally_segments)} "
          f"({total_rally:.0f}s total)")
    print(f"  Break segments: {len(break_segments)} "
          f"({total_break:.0f}s total)")

    # Save segments JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "video": str(video_path),
        "parameters": {
            "window_sec": args.window,
            "step_sec": args.step,
            "threshold": args.threshold,
            "smooth_window": args.smooth_window,
            "min_rally_sec": args.min_rally,
            "min_break_sec": args.min_break,
        },
        "segments": segments,
        "summary": {
            "rally_count": len(rally_segments),
            "break_count": len(break_segments),
            "rally_total_sec": round(total_rally, 1),
            "break_total_sec": round(total_break, 1),
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Segments saved to {output_path}")

    # Optional FFmpeg trim
    if args.trim:
        trim_path = args.trim_output or str(
            output_path.with_suffix("")) + "_rallies.mp4"
        trim_to_rallies(str(video_path), segments, trim_path)


if __name__ == "__main__":
    main()
