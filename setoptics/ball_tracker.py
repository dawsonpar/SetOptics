"""
Ball Tracking Service using BoT-SORT

Tracks volleyball across frames using multi-object tracking algorithms.
"""

import cv2
import numpy as np
from supervision import Detections, ByteTrack
from typing import List, Dict, Optional
from pathlib import Path
from ultralytics import YOLO


class BallTracker:
    """
    Tracks volleyball across video frames using BoT-SORT algorithm.

    Attributes:
        detector: YOLOv26n ball detector
        tracker: BoT-SORT multi-object tracker
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize detector and tracker."""
        if model_path is None:
            # Default to <repo root>/models/volleyball_yolo26n.pt
            project_root = Path(__file__).resolve().parent.parent
            model_path = str(project_root / 'models' / 'volleyball_yolo26n.pt')

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Ball detection model not found at {model_path}. "
                "Please ensure volleyball_yolo26n.pt is in the models/ directory."
            )

        self.detector = YOLO(model_path)
        # ByteTrack is a precursor to BoT-SORT, supervision uses this naming
        self.tracker = ByteTrack(
            track_activation_threshold=0.25,  # Lower for ball detection
            lost_track_buffer=30,  # Keep track for 30 frames after loss
            minimum_matching_threshold=0.7,  # IoU threshold for matching
            frame_rate=30  # Assumed FPS
        )

    def track_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Track ball throughout video.

        Args:
            video_path: Path to video file
            confidence_threshold: Minimum detection confidence

        Returns:
            List of tracking results per frame:
            [{
                'frame': int,
                'track_id': int or None,
                'bbox': [x1, y1, x2, y2] or None,
                'confidence': float or None
            }]
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect ball
            detections_yolo = self.detector(frame, conf=confidence_threshold)[0]

            # Convert to Supervision format
            detections = Detections.from_ultralytics(detections_yolo)

            # Update tracker
            tracked = self.tracker.update_with_detections(detections)

            # Extract result
            if len(tracked) > 0:
                # Assume single ball (take first detection)
                results.append({
                    'frame': frame_num,
                    'track_id': int(tracked.tracker_id[0]),
                    'bbox': tracked.xyxy[0].tolist(),
                    'confidence': float(tracked.confidence[0])
                })
            else:
                # No detection this frame
                results.append({
                    'frame': frame_num,
                    'track_id': None,
                    'bbox': None,
                    'confidence': None
                })

            frame_num += 1

        cap.release()
        return results

    def smooth_trajectory(
        self,
        tracking_results: List[Dict],
        smoothing_factor: float = 5.0
    ) -> List[Dict]:
        """
        Smooth ball trajectory using cubic spline interpolation.

        Args:
            tracking_results: Raw tracking results from track_video()
            smoothing_factor: Spline smoothing parameter (higher = smoother)

        Returns:
            Smoothed tracking results with same format
        """
        from scipy.interpolate import UnivariateSpline

        # Extract valid detections
        valid = [r for r in tracking_results if r['bbox'] is not None]
        if len(valid) < 4:  # Need at least 4 points for cubic spline
            return tracking_results

        frames = np.array([r['frame'] for r in valid])
        x_coords = np.array([r['bbox'][0] for r in valid])  # x1
        y_coords = np.array([r['bbox'][1] for r in valid])  # y1

        # Fit splines
        x_spline = UnivariateSpline(frames, x_coords, s=smoothing_factor)
        y_spline = UnivariateSpline(frames, y_coords, s=smoothing_factor)

        # Apply smoothing to all valid frames
        smoothed = [r.copy() for r in tracking_results]
        for i, result in enumerate(smoothed):
            if result['bbox'] is not None:
                frame = result['frame']
                bbox = result['bbox'].copy()
                bbox[0] = float(x_spline(frame))  # Smooth x1
                bbox[1] = float(y_spline(frame))  # Smooth y1
                # Keep width/height unchanged
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox[2] = bbox[0] + width
                bbox[3] = bbox[1] + height
                smoothed[i]['bbox'] = bbox

        return smoothed
