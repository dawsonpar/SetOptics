# EXPERIMENTAL: research-grade script (see README for status)
"""
Visualize signal-based rally detection signals overlaid with ground truth.

Plots audio signals (whistle energy, onset strength, silence), visual signals
(motion energy, motion variance), the fused score, and ground truth rally
boundaries from corrected annotation JSON.

Usage:
    python scripts/signal_rally_visualize.py \
        --signals ../data/processed/beach-game-001_signal.json \
        --ground-truth ../data/rally-gt/beach-game-001_annotations_corrected.json \
        --output ../data/processed/beach-game-001_signals.png

    # Or run the detector and visualize in one step:
    python scripts/signal_rally_visualize.py \
        --video ../data/rally-gt/beach-game-001.mp4 \
        --ground-truth ../data/rally-gt/beach-game-001_annotations_corrected.json \
        --output ../data/processed/beach-game-001_signals.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str) -> list[dict]:
    """Load corrected annotation JSON, return list of segments."""
    with open(gt_path) as f:
        data = json.load(f)
    return data["segments"]


def get_rally_spans(segments: list[dict]) -> list[tuple[float, float]]:
    """Extract (start_sec, end_sec) for in-play segments."""
    spans = []
    for seg in segments:
        if seg["type"] == "in-play":
            spans.append((seg["start_ms"] / 1000, seg["end_ms"] / 1000))
    return spans


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_rally_shading(ax: plt.Axes, rally_spans: list[tuple[float, float]],
                      color: str = "green", alpha: float = 0.15,
                      label: str = "Ground truth rally"):
    """Add vertical shading for ground truth rally spans."""
    labeled = False
    for start, end in rally_spans:
        ax.axvspan(start, end, color=color, alpha=alpha,
                   label=label if not labeled else None)
        labeled = True


def add_predicted_shading(ax: plt.Axes, segments: list[dict],
                          color: str = "blue", alpha: float = 0.1,
                          label: str = "Predicted rally"):
    """Add vertical shading for predicted rally segments."""
    labeled = False
    for seg in segments:
        if seg["type"] == "in-play":
            start = seg["start_ms"] / 1000
            end = seg["end_ms"] / 1000
            ax.axvspan(start, end, color=color, alpha=alpha,
                       label=label if not labeled else None)
            labeled = True


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_signals(signals_data: dict, gt_path: str | None = None,
                 output_path: str | None = None, show: bool = False):
    """Create a multi-panel signal visualization.

    Args:
        signals_data: Output from signal_rally_detector.run_pipeline()
        gt_path: Path to corrected annotation JSON (optional)
        output_path: Save plot to this path (optional)
        show: Display plot interactively
    """
    audio = signals_data["signals"]["audio"]
    visual = signals_data["signals"]["visual"]
    fused = np.array(signals_data["signals"]["fused"])
    segments = signals_data["segments"]

    has_audio = audio is not None
    has_deriv = ("audio_deriv" in (audio or {})
                 or "visual_deriv" in visual)
    n_panels = 4 + (2 if has_audio else 0) + (1 if has_deriv else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 3 * n_panels),
                             sharex=True)

    # Load ground truth if provided
    rally_spans = []
    if gt_path:
        gt_segments = load_ground_truth(gt_path)
        rally_spans = get_rally_spans(gt_segments)

    panel_idx = 0

    # --- Audio panels ---
    if has_audio:
        whistle = np.array(audio["whistle_energy"])
        onset = np.array(audio["onset_strength"])
        silence = np.array(audio["silence"])
        peaks = np.array(audio["whistle_peaks"])
        audio_score = np.array(audio["audio_score"])
        t_audio = np.arange(len(whistle))

        # Panel 1: Whistle energy + peaks
        ax = axes[panel_idx]
        ax.plot(t_audio, whistle, color="orange", linewidth=0.8,
                label="Whistle energy (2-4 kHz)")
        if len(peaks) > 0:
            ax.plot(peaks, whistle[peaks], "rv", markersize=8,
                    label=f"Whistle peaks ({len(peaks)})")
        add_rally_shading(ax, rally_spans)
        add_predicted_shading(ax, segments)
        ax.set_ylabel("Whistle Energy")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Audio: Whistle Band Energy (2-4 kHz)")
        panel_idx += 1

        # Panel 2: Onset strength + silence
        ax = axes[panel_idx]
        ax.plot(t_audio[:len(onset)], onset, color="blue", linewidth=0.8,
                label="Onset strength")
        ax.fill_between(t_audio[:len(silence)], 0,
                        silence * np.max(onset) if np.max(onset) > 0
                        else silence,
                        color="red", alpha=0.2, label="Silence")
        add_rally_shading(ax, rally_spans)
        add_predicted_shading(ax, segments)
        ax.set_ylabel("Onset / Silence")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Audio: Onset Strength & Silence Indicator")
        panel_idx += 1

    # --- Visual panels ---
    motion = np.array(visual["motion_energy"])
    variance = np.array(visual["motion_variance"])
    visual_score = np.array(visual["visual_score"])
    t_visual = np.arange(len(motion))

    # Panel: Motion energy + variance
    ax = axes[panel_idx]
    ax.plot(t_visual, motion, color="purple", linewidth=0.8,
            label="Motion energy (mean flow)")
    if len(variance) > 0:
        ax2 = ax.twinx()
        ax2.plot(t_visual[:len(variance)], variance, color="gray",
                 linewidth=0.5, alpha=0.6, label="Motion variance")
        ax2.set_ylabel("Variance", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
    add_rally_shading(ax, rally_spans)
    add_predicted_shading(ax, segments)
    ax.set_ylabel("Motion Energy")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Visual: Optical Flow Motion Energy & Variance")
    panel_idx += 1

    # Panel: Visual score
    ax = axes[panel_idx]
    ax.plot(t_visual[:len(visual_score)], visual_score, color="teal",
            linewidth=0.8, label="Visual score (normalized)")
    add_rally_shading(ax, rally_spans)
    add_predicted_shading(ax, segments)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Visual Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Visual: Normalized Score")
    panel_idx += 1

    # Panel: Derivatives (audio + visual)
    if has_deriv:
        ax = axes[panel_idx]
        t_max = 0
        if "visual_deriv" in visual:
            vd = np.array(visual["visual_deriv"])
            t_vd = np.arange(len(vd))
            ax.plot(t_vd, vd, color="purple", linewidth=0.8, alpha=0.7,
                    label="Visual derivative")
            t_max = max(t_max, len(vd))
        if has_audio and "audio_deriv" in audio:
            ad = np.array(audio["audio_deriv"])
            t_ad = np.arange(len(ad))
            ax.plot(t_ad, ad, color="orange", linewidth=0.8, alpha=0.7,
                    label="Audio derivative")
            t_max = max(t_max, len(ad))
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        add_rally_shading(ax, rally_spans)
        add_predicted_shading(ax, segments)
        ax.set_ylabel("Derivative")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Smoothed Derivatives (positive = rising activity)")
        panel_idx += 1

    # Panel: Fused score + threshold
    ax = axes[panel_idx]
    t_fused = np.arange(len(fused))
    config = signals_data.get("config", {})
    threshold = config.get("fusion_threshold", 0.5)

    ax.plot(t_fused, fused, color="black", linewidth=1.0,
            label="Fused score")
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.0,
               label=f"Threshold ({threshold})")
    ax.fill_between(t_fused, threshold, fused,
                    where=fused >= threshold, color="green", alpha=0.3,
                    label="Above threshold")
    add_rally_shading(ax, rally_spans)
    ax.set_ylabel("Fused Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (seconds)")
    ax.legend(loc="upper right", fontsize=8)

    weights = signals_data.get("config_effective_weights", {})
    aw = weights.get("audio_weight", "?")
    vw = weights.get("visual_weight", "?")
    ax.set_title(f"Fused Score (audio={aw}, visual={vw})")

    # Overall title
    video_path = signals_data.get("video_metadata", {}).get("path", "")
    video_name = Path(video_path).name if video_path else "Unknown"
    summary = signals_data.get("summary", {})
    rally_count = summary.get("rally_count", "?")
    gt_rally_count = len(rally_spans) if rally_spans else "N/A"
    fig.suptitle(
        f"{video_name} | Predicted: {rally_count} rallies | "
        f"Ground truth: {gt_rally_count} rallies",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize signal rally detection signals"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signals", type=str,
                       help="Path to signal detector output JSON")
    group.add_argument("--video", type=str,
                       help="Path to video (runs detector first)")

    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to corrected annotation JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Save plot to file (PNG/PDF)")
    parser.add_argument("--show", action="store_true",
                        help="Display plot interactively")

    # Detector config overrides (only used with --video)
    parser.add_argument("--fusion-threshold", type=float, default=0.5)
    parser.add_argument("--audio-weight", type=float, default=0.4)
    parser.add_argument("--visual-weight", type=float, default=0.6)
    args = parser.parse_args()

    if args.signals:
        with open(args.signals) as f:
            signals_data = json.load(f)
    else:
        # Run detector on video
        from signal_rally_detector import SignalConfig, run_pipeline

        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)

        config = SignalConfig(
            fusion_threshold=args.fusion_threshold,
            audio_weight=args.audio_weight,
            visual_weight=args.visual_weight,
        )
        signals_data = run_pipeline(str(video_path), config)

    if not args.output and not args.show:
        print("Warning: Neither --output nor --show specified. "
              "Defaulting to --show.")
        args.show = True

    plot_signals(signals_data, args.ground_truth, args.output, args.show)


if __name__ == "__main__":
    main()
