"""Abstract base class for rally detection strategies."""

from abc import ABC, abstractmethod
from pathlib import Path


class RallyDetectorBase(ABC):
    """Interface for rally detection implementations.

    All detectors return segments in canonical format with start_ms/end_ms.
    New detection methods (VideoMAE, ensemble) implement this interface
    and are selected via configuration.
    """

    @abstractmethod
    def detect(self, video_path: Path) -> list[dict]:
        """Detect rally segments in a video.

        Args:
            video_path: Path to the video file.

        Returns:
            List of segment dicts with keys:
                - type: "in-play" or "break"
                - start_ms: Start time in milliseconds
                - end_ms: End time in milliseconds
                - confidence: Float 0.0-1.0
                - rally_number: Int for in-play, None for break
        """
