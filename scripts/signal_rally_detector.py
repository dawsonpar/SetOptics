# EXPERIMENTAL: research-grade script (see README for status)
"""
Signal-based rally detection using audio and visual features.

Domain-agnostic alternative to VideoMAE: extracts audio (whistle energy,
onset strength, silence) and visual (optical flow motion) signals, computes
smoothed derivatives to detect rally transitions, and uses leaky integration
to convert edge detections into sustained rally scores.

Usage:
    python scripts/signal_rally_detector.py \
        --video ../data/rally-gt/beach-game-001.mp4 \
        --output ../data/processed/beach-game-001_signal.json
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import librosa
import numpy as np
from scipy.signal import butter, find_peaks, sosfilt


@dataclass
class SignalConfig:
    """All tunable parameters for the signal rally detector."""
    # Audio extraction
    whistle_band_low: float = 2000
    whistle_band_high: float = 4000
    whistle_peak_distance: int = 5
    silence_threshold_db: float = -40
    audio_sample_rate: int = 22050
    # Visual extraction
    visual_sample_fps: float = 2.0
    flow_resize_height: int = 480
    motion_percentile_low: float = 10
    motion_percentile_high: float = 90
    # Derivative parameters
    deriv_window: int = 5          # seconds for smoothed derivative
    deriv_decay: float = 0.88      # leaky integrator decay per second
    sustained_window: int = 10     # rolling mean window for level signal
    # Fusion weights (tuned for indoor volleyball)
    audio_weight: float = 0.5
    visual_weight: float = 0.5
    deriv_weight: float = 0.6      # weight for derivative-based signal
    level_weight: float = 0.4      # weight for sustained-level signal
    fusion_threshold: float = 0.45
    # Post-processing
    whistle_snap_window_sec: float = 2.0
    min_rally_sec: float = 3.0
    min_break_sec: float = 5.0
    merge_gap_sec: float = 2.0


def get_video_info(video_path: str) -> dict:
    """Get video duration, fps, and whether audio exists."""
    container = av.open(video_path)
    v_stream = container.streams.video[0]
    fps = float(v_stream.average_rate)
    total_frames = v_stream.frames
    if total_frames == 0:
        duration = float(v_stream.duration * v_stream.time_base)
    else:
        duration = total_frames / fps
    has_audio = len(container.streams.audio) > 0
    container.close()
    return {"path": video_path, "fps": fps,
            "duration_seconds": duration, "has_audio": has_audio}


def _extract_audio(video_path: str, sample_rate: int) -> np.ndarray:
    """Decode audio via PyAV, return float32 mono at sample_rate."""
    container = av.open(video_path)
    if len(container.streams.audio) == 0:
        container.close()
        return np.array([], dtype=np.float32)
    resampler = av.AudioResampler(
        format="s16", layout="mono", rate=sample_rate)
    samples = []
    for frame in container.decode(container.streams.audio[0]):
        for r in resampler.resample(frame):
            samples.append(r.to_ndarray().flatten())
    container.close()
    if not samples:
        return np.array([], dtype=np.float32)
    return np.concatenate(samples).astype(np.float32) / 32768.0


def _rms_per_second(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute RMS energy per 1-second window."""
    n = int(np.floor(len(signal) / sr))
    energy = np.zeros(n)
    for i in range(n):
        energy[i] = np.sqrt(np.mean(signal[i * sr:(i + 1) * sr] ** 2))
    return energy


def _onset_strength_per_second(audio: np.ndarray, sr: int) -> np.ndarray:
    """Onset strength envelope resampled to 1-second resolution."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_sr = sr / 512
    n = int(np.floor(len(audio) / sr))
    result = np.zeros(n)
    for i in range(n):
        s, e = int(i * onset_sr), min(int((i + 1) * onset_sr), len(onset_env))
        if s < e:
            result[i] = np.mean(onset_env[s:e])
    return result


def _normalize_percentile(arr: np.ndarray, p_lo: float, p_hi: float
                          ) -> np.ndarray:
    """Normalize array to 0-1 using percentile range."""
    if len(arr) == 0:
        return arr
    lo, hi = np.percentile(arr, p_lo), np.percentile(arr, p_hi)
    return np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)


def _smoothed_deriv(arr: np.ndarray, window: int) -> np.ndarray:
    """mean(next W sec) - mean(prev W sec). Positive = rising signal."""
    n = len(arr)
    deriv = np.zeros(n)
    for i in range(window, n - window):
        deriv[i] = np.mean(arr[i:i + window]) - np.mean(arr[i - window:i])
    return deriv


def _leaky_integrate(signal: np.ndarray, decay: float) -> np.ndarray:
    """state[t] = max(input[t], state[t-1] * decay). Spreads spikes over time."""
    n = len(signal)
    state = np.zeros(n)
    for i in range(n):
        prev = state[i - 1] * decay if i > 0 else 0
        state[i] = max(signal[i], prev)
    return state


def _sustained_level(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over W seconds - smooths out brief spikes."""
    n = len(arr)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        out[i] = np.mean(arr[max(0, i - half):min(n, i + half + 1)])
    return out


def compute_audio_signals(video_path: str, config: SignalConfig) -> dict | None:
    """Extract all audio signals. Returns None if no audio stream."""
    audio = _extract_audio(video_path, config.audio_sample_rate)
    if len(audio) == 0:
        return None
    sr = config.audio_sample_rate

    # Whistle energy: bandpass 2-4 kHz then RMS
    nyq = sr / 2.0
    sos = butter(4, [config.whistle_band_low / nyq,
                     config.whistle_band_high / nyq], btype="band", output="sos")
    whistle = _rms_per_second(sosfilt(sos, audio), sr)

    onset = _onset_strength_per_second(audio, sr)
    rms = _rms_per_second(audio, sr)
    silence = (20 * np.log10(rms + 1e-10) < config.silence_threshold_db
               ).astype(float)

    # Align lengths
    n = min(len(whistle), len(onset), len(silence))
    whistle, onset, silence = whistle[:n], onset[:n], silence[:n]

    # Whistle peaks (boundary candidates, not part of score)
    w_thresh = np.percentile(whistle, 90) if n > 0 else 0
    peaks, _ = find_peaks(whistle, height=w_thresh,
                          distance=config.whistle_peak_distance)

    # Audio score: onset * (1 - silence), normalized
    audio_score = onset * (1.0 - silence)
    if np.max(audio_score) > 0:
        audio_score = _normalize_percentile(audio_score, 5, 95)

    # Derivative: positive = audio activity rising
    audio_deriv = _smoothed_deriv(audio_score, config.deriv_window)

    return {"whistle_energy": whistle, "onset_strength": onset,
            "silence": silence, "whistle_peaks": peaks,
            "audio_score": audio_score, "audio_deriv": audio_deriv,
            "duration_sec": len(audio) / sr}


def _decode_gray_frames(video_path: str, target_fps: float,
                        resize_height: int) -> list[np.ndarray]:
    """Decode video at target_fps, return resized grayscale frames."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    interval = max(1, int(round(float(stream.average_rate) / target_fps)))
    frames = []
    for idx, frame in enumerate(container.decode(stream)):
        if idx % interval == 0:
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            new_w = int(w * resize_height / h)
            gray = cv2.cvtColor(
                cv2.resize(img, (new_w, resize_height)), cv2.COLOR_RGB2GRAY)
            frames.append(gray)
    container.close()
    return frames


def compute_visual_signals(video_path: str, config: SignalConfig) -> dict:
    """Extract motion energy and variance via Farneback optical flow."""
    frames = _decode_gray_frames(
        video_path, config.visual_sample_fps, config.flow_resize_height)
    empty = {"motion_energy": np.array([]), "motion_variance": np.array([]),
             "visual_score": np.array([]), "visual_deriv": np.array([])}
    if len(frames) < 2:
        return empty

    magnitudes, variances = [], []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i - 1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(np.mean(mag))
        variances.append(np.var(mag))

    magnitudes, variances = np.array(magnitudes), np.array(variances)

    fps = int(config.visual_sample_fps)
    n_sec = len(magnitudes) // fps
    motion_energy = np.array([np.mean(magnitudes[i*fps:(i+1)*fps])
                              for i in range(n_sec)])
    motion_variance = np.array([np.mean(variances[i*fps:(i+1)*fps])
                                for i in range(n_sec)])

    visual_score = _normalize_percentile(
        motion_energy, config.motion_percentile_low,
        config.motion_percentile_high)

    # Derivative: positive = motion activity rising
    visual_deriv = _smoothed_deriv(visual_score, config.deriv_window)

    return {"motion_energy": motion_energy, "motion_variance": motion_variance,
            "visual_score": visual_score, "visual_deriv": visual_deriv}


def fuse_scores(audio_signals: dict | None, visual_signals: dict,
                config: SignalConfig) -> np.ndarray:
    """Fuse audio + visual using derivative-based and level-based signals.

    Two components are combined:
    1. Derivative signal: smoothed derivative → clip positive → leaky integrate
       Captures rally transitions (the "sudden spike from low" pattern).
    2. Level signal: sustained (rolling mean) absolute score.
       Captures ongoing activity during established rallies.
    """
    vs = visual_signals["visual_score"]
    vd = visual_signals["visual_deriv"]

    if audio_signals is None or len(audio_signals["audio_score"]) == 0:
        # No audio: visual only
        pos_vd = np.clip(vd, 0, None)
        deriv_sig = _leaky_integrate(
            _normalize_percentile(pos_vd, 0, 95), config.deriv_decay)
        level_sig = _sustained_level(vs, config.sustained_window)
        return (config.deriv_weight * deriv_sig
                + config.level_weight * level_sig)

    a_s = audio_signals["audio_score"]
    a_d = audio_signals["audio_deriv"]

    # Align lengths
    n = min(len(a_s), len(vs), len(a_d), len(vd))
    a_s, vs = a_s[:n], vs[:n]
    a_d, vd = a_d[:n], vd[:n]
    aw, vw = config.audio_weight, config.visual_weight

    # Derivative component: clip positive, normalize, leaky integrate
    combined_deriv = aw * np.clip(a_d, 0, None) + vw * np.clip(vd, 0, None)
    deriv_sig = _leaky_integrate(
        _normalize_percentile(combined_deriv, 0, 95), config.deriv_decay)

    # Level component: sustained rolling mean of absolute scores
    combined_level = aw * a_s + vw * vs
    level_sig = _sustained_level(combined_level, config.sustained_window)

    fused = config.deriv_weight * deriv_sig + config.level_weight * level_sig
    return fused


def generate_candidates(fused_score: np.ndarray,
                        whistle_peaks: np.ndarray | None,
                        config: SignalConfig) -> list[dict]:
    """Threshold fused score, apply post-processing, return segments."""
    if len(fused_score) == 0:
        return []

    above = fused_score >= config.fusion_threshold
    raw, in_rally, start = [], False, 0
    for i in range(len(above)):
        if above[i] and not in_rally:
            start, in_rally = i, True
        elif not above[i] and in_rally:
            raw.append([float(start), float(i)])
            in_rally = False
    if in_rally:
        raw.append([float(start), float(len(above))])

    # Whistle boundary snapping
    if whistle_peaks is not None and len(whistle_peaks) > 0:
        snap = config.whistle_snap_window_sec
        for seg in raw:
            for peak in whistle_peaks:
                p = float(peak)
                if abs(seg[0] - p) <= snap:
                    seg[0] = p
                if abs(seg[1] - p) <= snap:
                    seg[1] = p

    # Merge close rallies, filter short ones
    merged = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] < config.merge_gap_sec:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg[:])
    rallies = [s for s in merged
               if s[1] - s[0] >= config.min_rally_sec]

    return _build_timeline(rallies, float(len(fused_score)),
                           fused_score, config)


def _build_timeline(rally_segs: list[list[float]], duration: float,
                    fused: np.ndarray, config: SignalConfig) -> list[dict]:
    """Build alternating break/rally timeline with confidence scores."""
    def make_seg(sid, stype, s, e):
        si, ei = max(0, int(s)), min(len(fused), int(e))
        if si < ei:
            mean = float(np.mean(fused[si:ei]))
            conf = mean if stype == "in-play" else 1.0 - mean
        else:
            conf = 0.5
        return {"segment_id": sid, "type": stype,
                "start_ms": int(s * 1000), "end_ms": int(e * 1000),
                "confidence": round(conf, 3)}

    full, sid, prev = [], 1, 0.0
    for s, e in rally_segs:
        if s > prev:
            gap = s - prev
            if gap >= config.min_break_sec or sid == 1:
                full.append(make_seg(sid, "break", prev, s))
                sid += 1
            elif full and full[-1]["type"] == "in-play":
                full[-1]["end_ms"] = int(s * 1000)
        full.append(make_seg(sid, "in-play", s, e))
        sid += 1
        prev = e
    if prev < duration:
        full.append(make_seg(sid, "break", prev, duration))

    rally_num = 1
    for seg in full:
        if seg["type"] == "in-play":
            seg["rally_number"] = rally_num
            rally_num += 1
        else:
            seg["rally_number"] = None
    return full


def run_pipeline(video_path: str, config: SignalConfig) -> dict:
    """Run full signal rally detection pipeline."""
    print(f"Processing: {video_path}")
    info = get_video_info(video_path)
    print(f"  {info['duration_seconds']:.1f}s, {info['fps']:.1f} fps, "
          f"audio={'yes' if info['has_audio'] else 'no'}")

    t0 = time.time()
    audio_signals = None
    if info["has_audio"]:
        audio_signals = compute_audio_signals(video_path, config)
        if audio_signals:
            print(f"  Audio: {len(audio_signals['audio_score'])}s, "
                  f"{len(audio_signals['whistle_peaks'])} whistle peaks "
                  f"[{time.time() - t0:.1f}s]")

    t1 = time.time()
    visual_signals = compute_visual_signals(video_path, config)
    print(f"  Visual: {len(visual_signals['visual_score'])}s "
          f"[{time.time() - t1:.1f}s]")

    aw = 0.0 if audio_signals is None else config.audio_weight
    vw = 1.0 if audio_signals is None else config.visual_weight
    fused = fuse_scores(audio_signals, visual_signals, config)
    peaks = audio_signals["whistle_peaks"] if audio_signals else None
    segments = generate_candidates(fused, peaks, config)

    rallies = [s for s in segments if s["type"] == "in-play"]
    breaks = [s for s in segments if s["type"] == "break"]
    r_ms = sum(s["end_ms"] - s["start_ms"] for s in rallies)
    b_ms = sum(s["end_ms"] - s["start_ms"] for s in breaks)
    print(f"  Result: {len(rallies)} rallies ({r_ms / 1000:.0f}s), "
          f"{len(breaks)} breaks ({b_ms / 1000:.0f}s) "
          f"[{time.time() - t0:.1f}s total]")

    return {
        "video_metadata": {
            "path": str(Path(video_path).resolve()),
            "fps": info["fps"], "duration_seconds": info["duration_seconds"],
        },
        "config": asdict(config),
        "config_effective_weights": {"audio_weight": aw, "visual_weight": vw},
        "segments": segments,
        "summary": {
            "rally_count": len(rallies), "break_count": len(breaks),
            "rally_total_sec": round(r_ms / 1000, 1),
            "break_total_sec": round(b_ms / 1000, 1),
        },
        "signals": {
            **_serialize_signals(audio_signals, visual_signals),
            "fused": fused.tolist() if len(fused) > 0 else [],
        },
    }


def _serialize_signals(audio: dict | None, visual: dict) -> dict:
    """Convert numpy arrays to lists for JSON serialization."""
    a = None
    if audio is not None:
        a = {k: audio[k].tolist() for k in
             ("whistle_energy", "onset_strength", "silence",
              "whistle_peaks", "audio_score", "audio_deriv")}
    v = {k: visual[k].tolist() for k in
         ("motion_energy", "motion_variance", "visual_score", "visual_deriv")}
    return {"audio": a, "visual": v}


def _build_config_from_args(args) -> SignalConfig:
    return SignalConfig(
        whistle_band_low=args.whistle_band_low,
        whistle_band_high=args.whistle_band_high,
        silence_threshold_db=args.silence_threshold_db,
        visual_sample_fps=args.visual_fps,
        flow_resize_height=args.flow_resize_height,
        deriv_window=args.deriv_window,
        deriv_decay=args.deriv_decay,
        sustained_window=args.sustained_window,
        audio_weight=args.audio_weight,
        visual_weight=args.visual_weight,
        deriv_weight=args.deriv_weight,
        level_weight=args.level_weight,
        fusion_threshold=args.fusion_threshold,
        whistle_snap_window_sec=args.whistle_snap,
        min_rally_sec=args.min_rally,
        min_break_sec=args.min_break,
        merge_gap_sec=args.merge_gap,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Signal-based rally detection")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    for name, tp, default in [
        ("whistle-band-low", float, 2000), ("whistle-band-high", float, 4000),
        ("silence-threshold-db", float, -40), ("visual-fps", float, 2.0),
        ("flow-resize-height", int, 480), ("deriv-window", int, 5),
        ("deriv-decay", float, 0.88), ("sustained-window", int, 10),
        ("audio-weight", float, 0.5), ("visual-weight", float, 0.5),
        ("deriv-weight", float, 0.6), ("level-weight", float, 0.4),
        ("fusion-threshold", float, 0.45), ("whistle-snap", float, 2.0),
        ("min-rally", float, 3.0), ("min-break", float, 5.0),
        ("merge-gap", float, 2.0),
    ]:
        p.add_argument(f"--{name}", type=tp, default=default)
    args = p.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    result = run_pipeline(str(video_path), _build_config_from_args(args))
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nOutput saved to {out}")
