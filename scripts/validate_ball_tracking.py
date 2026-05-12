# EXPERIMENTAL: research-grade script (see README for status)
"""
Validate ball tracking using collection-of-sets annotations.

Validates that ball tracking maintains single ID per video segment
and has smooth trajectories without jumps.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from setoptics.ball_tracker import BallTracker


def parse_annotations(annotation_file: str) -> list[dict]:
    """Parse collection-of-sets-annotations.txt format."""
    segments = []
    with open(annotation_file) as f:
        for line in f:
            if not line.strip():
                continue
            # Format: "video 1: frame 0 - frame 77"
            parts = line.strip().split(':')
            video_num = int(parts[0].split()[1])
            frame_range = parts[1].strip()
            start = int(frame_range.split('-')[0].split()[1])
            end = int(frame_range.split('-')[1].split()[1])

            segments.append({
                'video': video_num,
                'start_frame': start,
                'end_frame': end
            })
    return segments


def validate_segment_tracking(
    segment: dict,
    tracking_results: list[dict]
) -> dict:
    """
    Validate tracking for a single segment.

    Returns:
        {
            'video': int,
            'track_ids_found': List[int],
            'id_switches': int,
            'position_jumps': int,
            'detection_rate': float
        }
    """
    start = segment['start_frame']
    end = segment['end_frame']

    # Extract results for this segment
    segment_results = [
        r for r in tracking_results
        if start <= r['frame'] <= end
    ]

    # Count unique track IDs
    track_ids = [
        r['track_id'] for r in segment_results
        if r['track_id'] is not None
    ]
    unique_ids = list(set(track_ids))
    id_switches = len(unique_ids) - 1 if unique_ids else 0

    # Detect position jumps (>50px movement between frames)
    jumps = 0
    for i in range(1, len(segment_results)):
        prev = segment_results[i-1]
        curr = segment_results[i]

        if prev['bbox'] and curr['bbox']:
            prev_center = np.array([
                (prev['bbox'][0] + prev['bbox'][2]) / 2,
                (prev['bbox'][1] + prev['bbox'][3]) / 2
            ])
            curr_center = np.array([
                (curr['bbox'][0] + curr['bbox'][2]) / 2,
                (curr['bbox'][1] + curr['bbox'][3]) / 2
            ])

            distance = np.linalg.norm(curr_center - prev_center)
            if distance > 50:  # 50px threshold
                jumps += 1

    # Detection rate
    detected_frames = sum(
        1 for r in segment_results if r['bbox'] is not None
    )
    detection_rate = (
        detected_frames / len(segment_results) if segment_results else 0
    )

    return {
        'video': segment['video'],
        'track_ids_found': unique_ids,
        'id_switches': id_switches,
        'position_jumps': jumps,
        'detection_rate': detection_rate,
        'total_frames': len(segment_results)
    }


def main():
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / 'data/videos/collection-of-sets.mp4'
    annotation_file = project_root / 'data/videos/collection-of-sets-annotations.txt'
    output_file = project_root / 'data/processed/ball_tracking_validation.json'

    # Parse segments
    print("Loading annotations...")
    segments = parse_annotations(str(annotation_file))
    print(f"Found {len(segments)} video segments")

    # Run tracking
    print("\nRunning ball tracking...")
    tracker = BallTracker()
    tracking_results = tracker.track_video(str(video_path))

    # Optional: Apply smoothing
    print("Smoothing trajectories...")
    smoothed_results = tracker.smooth_trajectory(tracking_results)

    # Validate each segment
    print("\nValidating per segment:")
    print("-" * 60)

    validation_results = []
    for segment in segments:
        result = validate_segment_tracking(segment, smoothed_results)
        validation_results.append(result)

        print(
            f"Video {result['video']:2d}: "
            f"IDs={result['track_ids_found']} "
            f"(switches={result['id_switches']}) | "
            f"Jumps={result['position_jumps']} | "
            f"Detection={result['detection_rate']:.1%} "
            f"({result['total_frames']} frames)"
        )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_switches = sum(r['id_switches'] for r in validation_results)
    total_jumps = sum(r['position_jumps'] for r in validation_results)
    avg_detection_rate = np.mean(
        [r['detection_rate'] for r in validation_results]
    )

    print(f"Total ID switches: {total_switches} (target: <5% of segments)")
    print(f"Total position jumps: {total_jumps}")
    print(f"Average detection rate: {avg_detection_rate:.1%}")

    # Save results
    output_data = {
        'segments': validation_results,
        'summary': {
            'total_id_switches': total_switches,
            'total_position_jumps': total_jumps,
            'average_detection_rate': float(avg_detection_rate)
        }
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_file), 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)

    id_switch_rate = total_switches / len(segments)
    print(f"✓ ID switch rate: {id_switch_rate:.1%} (target: <5%)")
    print(f"✓ Detection rate: {avg_detection_rate:.1%} (target: >80%)")

    if id_switch_rate < 0.05 and avg_detection_rate > 0.8:
        print("\n✅ VALIDATION PASSED")
    else:
        print("\n⚠️  VALIDATION NEEDS IMPROVEMENT")


if __name__ == "__main__":
    main()
