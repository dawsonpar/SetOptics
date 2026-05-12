#!/usr/bin/env python3
# EXPERIMENTAL: research-grade script (see README for status)
"""
Hybrid multi-method rally detection system.

Combines signal-based detector, VideoMAE, and LLM vision models
using weighted ensemble fusion.

Usage:
    python scripts/ensemble_rally_detector.py --video <path> --output <path>
    python scripts/ensemble_rally_detector.py --video <path> --output <path> --mode fast
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class EnsembleConfig:
    """Configuration for ensemble rally detection."""

    # Method selection
    use_signal: bool = True
    use_videomae: bool = True
    use_llm: bool = False  # Disabled by default (user can enable)

    # Fusion weights (must sum to 1.0) - indoor default
    signal_weight: float = 0.4
    videomae_weight: float = 0.4
    llm_weight: float = 0.2

    # Threshold for fused score
    fusion_threshold: float = 0.5

    # LLM provider (if enabled)
    llm_provider: str = "google"
    llm_model: str = "gemini-3-flash-preview"

    # Post-processing
    min_rally_sec: float = 3.0
    min_break_sec: float = 5.0
    merge_gap_sec: float = 2.0

    # Domain-specific (auto-detect or override)
    domain: str = "auto"  # "auto", "indoor", or "beach"

    def __post_init__(self):
        """Validate configuration."""
        # Normalize weights based on enabled methods
        total_weight = 0.0
        if self.use_signal:
            total_weight += self.signal_weight
        else:
            self.signal_weight = 0.0

        if self.use_videomae:
            total_weight += self.videomae_weight
        else:
            self.videomae_weight = 0.0

        if self.use_llm:
            total_weight += self.llm_weight
        else:
            self.llm_weight = 0.0

        if total_weight == 0:
            raise ValueError("At least one detection method must be enabled")

        # Normalize weights to sum to 1.0
        if abs(total_weight - 1.0) > 0.01:
            self.signal_weight /= total_weight
            self.videomae_weight /= total_weight
            self.llm_weight /= total_weight


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def detect_domain(video_path: str) -> str:
    """
    Auto-detect if video is indoor or beach volleyball.

    For now, use simple heuristic based on filename.
    Future: Could use audio spectrum analysis or LLM single-frame classification.
    """
    video_name = Path(video_path).stem.lower()
    if "beach" in video_name:
        return "beach"
    elif "indoor" in video_name:
        return "indoor"
    else:
        # Default to indoor if can't determine
        return "indoor"


def get_domain_config(domain: str, mode: str = "accurate") -> EnsembleConfig:
    """
    Get domain-specific ensemble configuration.

    Args:
        domain: "indoor" or "beach"
        mode: "fast" (signal+videomae only) or "accurate" (all methods)
    """
    if mode == "fast":
        # Fast mode: signal + VideoMAE only
        if domain == "beach":
            return EnsembleConfig(
                use_signal=True,
                use_videomae=True,
                use_llm=False,
                signal_weight=0.5,
                videomae_weight=0.5,
                llm_weight=0.0,
                domain=domain,
            )
        else:  # indoor
            return EnsembleConfig(
                use_signal=True,
                use_videomae=True,
                use_llm=False,
                signal_weight=0.4,
                videomae_weight=0.6,
                llm_weight=0.0,
                domain=domain,
            )
    else:  # accurate mode
        # Accurate mode: all three methods (LLM is free with Gemini)
        if domain == "beach":
            # Beach: rely heavily on LLM (signal/VideoMAE both fail)
            return EnsembleConfig(
                use_signal=True,
                use_videomae=True,
                use_llm=True,
                signal_weight=0.1,
                videomae_weight=0.1,
                llm_weight=0.8,
                domain=domain,
            )
        else:  # indoor
            # Indoor: VideoMAE strongest (90.5% recall), LLM validates
            return EnsembleConfig(
                use_signal=True,
                use_videomae=True,
                use_llm=True,
                signal_weight=0.3,
                videomae_weight=0.4,
                llm_weight=0.3,
                domain=domain,
            )


def run_signal_detector(video_path: str, temp_dir: Path) -> dict:
    """
    Run signal-based rally detector.

    Returns:
        dict with 'segments' and 'signals' (including per-second scores)
    """
    print("Running signal detector...")
    t0 = time.time()

    output_path = temp_dir / "signal_output.json"

    # Use absolute path to script
    script_dir = Path(__file__).parent
    signal_script = script_dir / "signal_rally_detector.py"

    cmd = [
        sys.executable,
        str(signal_script),
        "--video",
        video_path,
        "--output",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Signal detector failed: {result.stderr}")

    with open(output_path) as f:
        data = json.load(f)

    print(f"  Signal detector completed in {time.time() - t0:.1f}s")
    return data


def run_videomae_detector(video_path: str, temp_dir: Path) -> dict:
    """
    Run VideoMAE rally detector.

    Returns:
        dict with 'segments' and per-window probabilities
    """
    print("Running VideoMAE detector...")
    t0 = time.time()

    output_path = temp_dir / "videomae_output.json"

    # Use absolute paths
    script_dir = Path(__file__).parent
    videomae_script = script_dir / "infer_rally_detector.py"
    model_path = script_dir.parent.parent / "models" / "rally_detector.pth"

    cmd = [
        sys.executable,
        str(videomae_script),
        "--video",
        video_path,
        "--model",
        str(model_path),
        "--output",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"VideoMAE detector failed: {result.stderr}")

    with open(output_path) as f:
        data = json.load(f)

    print(f"  VideoMAE detector completed in {time.time() - t0:.1f}s")
    return data


def run_llm_annotator(video_path: str, temp_dir: Path, config: EnsembleConfig) -> dict:
    """
    Run LLM-based rally annotation.

    Returns:
        dict with 'segments'
    """
    print(f"Running LLM annotator ({config.llm_model})...")
    t0 = time.time()

    output_path = temp_dir / "llm_output.json"

    # Check if LLM annotation already exists
    annotation_path = (
        Path(video_path).parent / f"{Path(video_path).stem}_annotations_corrected.json"
    )
    if annotation_path.exists():
        print(f"  Using existing LLM annotation: {annotation_path}")
        with open(annotation_path) as f:
            data = json.load(f)
        print(f"  LLM annotation loaded in {time.time() - t0:.1f}s")
        return data

    # Use absolute path to script
    script_dir = Path(__file__).parent
    tools_dir = script_dir.parent.parent / "tools" / "annotation"
    annotate_script = tools_dir / "annotate_fast.py"

    # Run LLM annotator
    cmd = [
        sys.executable,
        str(annotate_script),
        video_path,
        "--output-dir",
        str(temp_dir / "llm_batches"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LLM annotator failed: {result.stderr}")

    # Load combined results
    combined_path = temp_dir / "llm_batches" / "combined.json"
    if not combined_path.exists():
        raise RuntimeError("LLM annotator did not produce combined.json")

    with open(combined_path) as f:
        data = json.load(f)

    print(f"  LLM annotator completed in {time.time() - t0:.1f}s")
    return data


def normalize_signal_to_timeline(signal_data: dict, duration_sec: float) -> np.ndarray:
    """
    Convert signal detector output to per-second score array.

    Args:
        signal_data: Output from signal detector
        duration_sec: Video duration in seconds

    Returns:
        np.ndarray of shape (duration_sec,) with scores [0-1]
    """
    if "signals" not in signal_data or "fused" not in signal_data["signals"]:
        # No signals available, return zeros
        return np.zeros(int(duration_sec))

    fused = np.array(signal_data["signals"]["fused"])

    # Pad or truncate to match duration
    target_len = int(duration_sec)
    if len(fused) < target_len:
        fused = np.pad(fused, (0, target_len - len(fused)), constant_values=0)
    elif len(fused) > target_len:
        fused = fused[:target_len]

    # Ensure values are in [0, 1] range
    fused = np.clip(fused, 0, 1)

    return fused


def normalize_videomae_to_timeline(
    videomae_data: dict, duration_sec: float
) -> np.ndarray:
    """
    Convert VideoMAE segments to per-second score array.

    VideoMAE outputs segments with labels. We need to convert this to
    a per-second probability array.

    For now, use simple approach:
    - rally segments get score 1.0
    - break segments get score 0.0
    - interpolate at boundaries

    Args:
        videomae_data: Output from VideoMAE detector
        duration_sec: Video duration in seconds

    Returns:
        np.ndarray of shape (duration_sec,) with scores [0-1]
    """
    target_len = int(duration_sec)
    scores = np.zeros(target_len)

    if "segments" not in videomae_data:
        return scores

    for seg in videomae_data["segments"]:
        if seg["label"] == "rally":
            start_idx = int(seg["start"])
            end_idx = int(min(seg["end"], duration_sec))
            scores[start_idx:end_idx] = 1.0

    return scores


def normalize_llm_to_timeline(llm_data: dict, duration_sec: float) -> np.ndarray:
    """
    Convert LLM annotation segments to per-second score array.

    Args:
        llm_data: Output from LLM annotator
        duration_sec: Video duration in seconds

    Returns:
        np.ndarray of shape (duration_sec,) with scores [0-1]
    """
    target_len = int(duration_sec)
    scores = np.zeros(target_len)

    if "segments" not in llm_data:
        return scores

    for seg in llm_data["segments"]:
        if seg["type"] == "in-play":
            start_idx = int(seg["start_ms"] / 1000)
            end_idx = int(min(seg["end_ms"] / 1000, duration_sec))
            # LLM segments assumed to have confidence 1.0
            confidence = seg.get("confidence", 1.0)
            scores[start_idx:end_idx] = confidence

    return scores


def fuse_scores(
    signal_score: np.ndarray,
    videomae_score: np.ndarray,
    llm_score: np.ndarray,
    config: EnsembleConfig,
) -> np.ndarray:
    """
    Weighted fusion of all method scores.

    Args:
        signal_score: Per-second scores from signal detector
        videomae_score: Per-second scores from VideoMAE
        llm_score: Per-second scores from LLM
        config: Ensemble configuration with weights

    Returns:
        np.ndarray of fused scores [0-1]
    """
    fused = (
        config.signal_weight * signal_score
        + config.videomae_weight * videomae_score
        + config.llm_weight * llm_score
    )

    return np.clip(fused, 0, 1)


def scores_to_segments(
    scores: np.ndarray, config: EnsembleConfig, fps: float = 60.0
) -> list[dict]:
    """
    Convert per-second fused scores to segments.

    Args:
        scores: Per-second fused scores
        config: Ensemble configuration
        fps: Video frame rate

    Returns:
        List of segment dicts with type, start_ms, end_ms
    """
    # Threshold scores
    binary = (scores >= config.fusion_threshold).astype(int)

    # Find transitions
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Build initial segments
    segments = []
    for start_sec, end_sec in zip(starts, ends):
        duration = end_sec - start_sec
        if binary[start_sec] == 1 and duration >= config.min_rally_sec:
            segments.append(
                {
                    "type": "in-play",
                    "start_ms": int(start_sec * 1000),
                    "end_ms": int(end_sec * 1000),
                    "confidence": float(np.mean(scores[start_sec:end_sec])),
                }
            )

    # Merge close segments
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap_sec = (seg["start_ms"] - prev["end_ms"]) / 1000
            if gap_sec <= config.merge_gap_sec:
                # Merge
                prev["end_ms"] = seg["end_ms"]
                prev["confidence"] = (prev["confidence"] + seg["confidence"]) / 2
            else:
                merged.append(seg)
        segments = merged

    # Add breaks between rallies
    final_segments = []
    rally_number = 0
    prev_end_ms = 0

    for seg in segments:
        # Add break before this rally
        if seg["start_ms"] > prev_end_ms:
            break_duration = (seg["start_ms"] - prev_end_ms) / 1000
            if break_duration >= config.min_break_sec:
                final_segments.append(
                    {
                        "segment_id": len(final_segments) + 1,
                        "type": "break",
                        "start_ms": prev_end_ms,
                        "end_ms": seg["start_ms"],
                        "confidence": 1.0,
                        "rally_number": None,
                    }
                )

        # Add rally
        rally_number += 1
        final_segments.append(
            {
                "segment_id": len(final_segments) + 1,
                "type": "in-play",
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "confidence": seg["confidence"],
                "rally_number": rally_number,
            }
        )
        prev_end_ms = seg["end_ms"]

    # Add final break if needed
    video_duration_ms = int(len(scores) * 1000)
    if prev_end_ms < video_duration_ms:
        break_duration = (video_duration_ms - prev_end_ms) / 1000
        if break_duration >= config.min_break_sec:
            final_segments.append(
                {
                    "segment_id": len(final_segments) + 1,
                    "type": "break",
                    "start_ms": prev_end_ms,
                    "end_ms": video_duration_ms,
                    "confidence": 1.0,
                    "rally_number": None,
                }
            )

    return final_segments


def run_ensemble_pipeline(
    video_path: str, config: EnsembleConfig, temp_dir: Path
) -> dict:
    """
    Run complete ensemble pipeline.

    Args:
        video_path: Path to input video
        config: Ensemble configuration
        temp_dir: Temporary directory for intermediate outputs

    Returns:
        dict with ensemble results
    """
    print(f"\nRunning ensemble pipeline: {video_path}")
    print(f"  Mode: {config.domain}")
    print(
        f"  Methods: signal={config.use_signal}, videomae={config.use_videomae}, llm={config.use_llm}"
    )
    print(
        f"  Weights: signal={config.signal_weight:.2f}, videomae={config.videomae_weight:.2f}, llm={config.llm_weight:.2f}"
    )

    t_start = time.time()

    # Get video metadata
    duration_sec = get_video_duration(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"  Duration: {duration_sec:.1f}s, FPS: {fps:.1f}")

    # Run detectors in parallel (or sequentially for now)
    signal_data = None
    videomae_data = None
    llm_data = None

    if config.use_signal:
        signal_data = run_signal_detector(video_path, temp_dir)

    if config.use_videomae:
        videomae_data = run_videomae_detector(video_path, temp_dir)

    if config.use_llm:
        llm_data = run_llm_annotator(video_path, temp_dir, config)

    # Normalize to common timeline
    print("\nNormalizing to common timeline...")
    signal_score = (
        normalize_signal_to_timeline(signal_data, duration_sec)
        if signal_data
        else np.zeros(int(duration_sec))
    )
    videomae_score = (
        normalize_videomae_to_timeline(videomae_data, duration_sec)
        if videomae_data
        else np.zeros(int(duration_sec))
    )
    llm_score = (
        normalize_llm_to_timeline(llm_data, duration_sec)
        if llm_data
        else np.zeros(int(duration_sec))
    )

    print(
        f"  Timeline: {len(signal_score)}s (signal), {len(videomae_score)}s (videomae), {len(llm_score)}s (llm)"
    )

    # Fuse scores
    print("\nFusing scores...")
    fused_score = fuse_scores(signal_score, videomae_score, llm_score, config)

    # Convert to segments
    print("\nGenerating segments...")
    segments = scores_to_segments(fused_score, config, fps)

    rally_segments = [s for s in segments if s["type"] == "in-play"]
    break_segments = [s for s in segments if s["type"] == "break"]

    print(f"\nResults:")
    print(f"  Rally segments: {len(rally_segments)}")
    print(f"  Break segments: {len(break_segments)}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    return {
        "video_metadata": {
            "path": str(Path(video_path).resolve()),
            "fps": fps,
            "duration_seconds": duration_sec,
        },
        "config": asdict(config),
        "segments": segments,
        "summary": {
            "rally_count": len(rally_segments),
            "break_count": len(break_segments),
            "rally_total_sec": round(
                sum((s["end_ms"] - s["start_ms"]) / 1000 for s in rally_segments), 1
            ),
            "break_total_sec": round(
                sum((s["end_ms"] - s["start_ms"]) / 1000 for s in break_segments), 1
            ),
        },
        "scores": {
            "signal": signal_score.tolist(),
            "videomae": videomae_score.tolist(),
            "llm": llm_score.tolist(),
            "fused": fused_score.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble rally detection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument(
        "--mode",
        choices=["fast", "accurate"],
        default="accurate",
        help="Detection mode (fast=signal+videomae, accurate=all methods with LLM)",
    )
    parser.add_argument(
        "--domain",
        choices=["auto", "indoor", "beach"],
        default="auto",
        help="Video domain (auto-detect by default)",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory for intermediate outputs",
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Detect domain
    domain = detect_domain(str(video_path)) if args.domain == "auto" else args.domain

    # Get configuration
    config = get_domain_config(domain, args.mode)

    # Create temp directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = Path("/tmp") / f"ensemble_{video_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        result = run_ensemble_pipeline(str(video_path), config, temp_dir)

        # Save output
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nOutput saved to {output_path}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
