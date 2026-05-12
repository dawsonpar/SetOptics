"""
Audio whistle detection for rally boundary hints.

Extracts referee whistle timestamps from volleyball video audio using
bandpass filtering (2-4 kHz) and peak detection. Logic extracted from
backend/scripts/signal_rally_detector.py.

Dependencies: numpy, scipy (no librosa needed)
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfilt


# Default parameters (from SignalConfig in signal_rally_detector.py)
SAMPLE_RATE = 22050
WHISTLE_BAND_LOW = 2000
WHISTLE_BAND_HIGH = 4000
WHISTLE_PEAK_DISTANCE = 5  # seconds between peaks


def _extract_audio_from_video(video_path: str | Path) -> np.ndarray:
    """Extract mono audio from video using ffmpeg, return as float32 array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            tmp_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed: {result.stderr}"
            )

        sr, data = wavfile.read(tmp_path)
        # Normalize int16 to float32 [-1, 1]
        audio = data.astype(np.float32) / 32768.0
        return audio
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _rms_per_second(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute RMS energy per 1-second window."""
    n_seconds = len(signal) // sr
    if n_seconds == 0:
        return np.array([])
    rms = np.zeros(n_seconds)
    for i in range(n_seconds):
        chunk = signal[i * sr:(i + 1) * sr]
        rms[i] = np.sqrt(np.mean(chunk ** 2))
    return rms


def detect_whistle_timestamps(
    video_path: str | Path,
    band_low: float = WHISTLE_BAND_LOW,
    band_high: float = WHISTLE_BAND_HIGH,
    peak_distance: int = WHISTLE_PEAK_DISTANCE,
    sr: int = SAMPLE_RATE,
    percentile: float = 90,
) -> list[float]:
    """Detect referee whistle timestamps in a volleyball video.

    Applies a bandpass filter in the whistle frequency range (2-4 kHz),
    computes per-second RMS energy, and finds peaks above the given
    percentile threshold.

    Args:
        video_path: Path to video file.
        band_low: Lower bandpass frequency in Hz.
        band_high: Upper bandpass frequency in Hz.
        peak_distance: Minimum seconds between whistle peaks.
        sr: Audio sample rate.
        percentile: Energy percentile threshold (higher = fewer, more
            confident detections). Default 90.

    Returns:
        List of timestamps (in seconds) where whistles were detected.
    """
    audio = _extract_audio_from_video(video_path)
    if len(audio) == 0:
        return []

    # Bandpass filter for whistle frequency range
    nyq = sr / 2.0
    sos = butter(
        4, [band_low / nyq, band_high / nyq], btype="band", output="sos"
    )
    filtered = sosfilt(sos, audio)

    # Per-second RMS of filtered signal
    whistle_energy = _rms_per_second(filtered, sr)
    if len(whistle_energy) == 0:
        return []

    threshold = np.percentile(whistle_energy, percentile)
    peaks, _ = find_peaks(
        whistle_energy, height=threshold, distance=peak_distance
    )

    return [float(p) for p in peaks]
