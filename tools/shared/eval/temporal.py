"""
Shared temporal evaluation metrics for rally/segment detection.

Provides IoU-based matching, precision/recall/F1, boundary accuracy,
and aggregation across multiple videos. Used by signal, ensemble,
and LLM annotation evaluators.

Usage:
    from shared.eval.temporal import (
        Interval, load_intervals, evaluate, print_metrics,
        aggregate_metrics,
    )
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Interval:
    """A time interval in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_intervals(
    path: str,
    type_filter: str = "in-play",
) -> list[Interval]:
    """Load intervals from a JSON file with a 'segments' array.

    Expects segments with start_ms/end_ms (milliseconds) and a 'type'
    field. Filters to segments matching type_filter.
    """
    with open(path) as f:
        data = json.load(f)

    intervals = []
    for seg in data["segments"]:
        if seg["type"] != type_filter:
            continue
        if "start_ms" in seg:
            intervals.append(Interval(
                start=seg["start_ms"] / 1000,
                end=seg["end_ms"] / 1000,
            ))
    return intervals


def temporal_iou(a: Interval, b: Interval) -> float:
    """Compute intersection-over-union of two time intervals."""
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)
    intersection = max(0, overlap_end - overlap_start)
    union = (a.duration + b.duration) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_matrix(
    predicted: list[Interval],
    ground_truth: list[Interval],
) -> np.ndarray:
    """Compute pairwise IoU matrix (predicted x ground_truth)."""
    matrix = np.zeros((len(predicted), len(ground_truth)))
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            matrix[i, j] = temporal_iou(p, g)
    return matrix


def center_of(interval: Interval) -> float:
    """Return the temporal center of an interval."""
    return (interval.start + interval.end) / 2


def match_rallies_by_center(
    predicted: list[Interval],
    ground_truth: list[Interval],
    tolerance_sec: float = 0.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy center-based matching of predicted to ground truth rallies.

    A predicted segment matches a ground truth segment if the predicted
    center falls within the GT interval (expanded by tolerance_sec on
    each side). Ties are broken by distance from predicted center to
    GT center (closest first).

    Returns:
        matches: list of (pred_idx, gt_idx) pairs
        unmatched_pred: false positive indices
        unmatched_gt: missed rally indices
    """
    if not predicted or not ground_truth:
        return [], list(range(len(predicted))), list(range(len(ground_truth)))

    # Build all candidate (pred_idx, gt_idx, distance) triples
    candidates = []
    for i, p in enumerate(predicted):
        c_pred = center_of(p)
        for j, g in enumerate(ground_truth):
            expanded_start = g.start - tolerance_sec
            expanded_end = g.end + tolerance_sec
            if expanded_start <= c_pred <= expanded_end:
                dist = abs(c_pred - center_of(g))
                candidates.append((i, j, dist))

    # Greedy: sort by distance, match closest first
    candidates.sort(key=lambda x: x[2])

    matches = []
    used_pred = set()
    used_gt = set()

    for p_idx, g_idx, _ in candidates:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        matches.append((p_idx, g_idx))
        used_pred.add(p_idx)
        used_gt.add(g_idx)

    unmatched_pred = [i for i in range(len(predicted)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_gt]

    return matches, unmatched_pred, unmatched_gt


def match_rallies_by_boundary_tolerance(
    predicted: list[Interval],
    ground_truth: list[Interval],
    start_tolerance_sec: float = 0.5,
    end_tolerance_before_sec: float = 0.5,
    end_tolerance_after_sec: float = 1.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy matching using asymmetric boundary-tolerance windows.

    A predicted rally matches a GT rally iff BOTH:
      - |pred.start - gt.start| <= start_tolerance_sec
      - (gt.end - end_tolerance_before_sec) <= pred.end
            <= (gt.end + end_tolerance_after_sec)

    The asymmetric end window reflects the product observation that
    viewers tolerate cuts that extend slightly past the rally's true
    end (`_after` > 0) but find early cuts jarring (`_before` is
    small). Start windows are symmetric.

    Greedy resolution when multiple pairs satisfy the rule: pick the
    pair with the smallest summed boundary error first.

    Returns:
        matches: list of (pred_idx, gt_idx)
        unmatched_pred: false positives
        unmatched_gt: missed rallies
    """
    if not predicted or not ground_truth:
        return [], list(range(len(predicted))), list(range(len(ground_truth)))

    candidates = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            start_delta = p.start - g.start
            end_delta = p.end - g.end
            if abs(start_delta) > start_tolerance_sec:
                continue
            if end_delta < -end_tolerance_before_sec:
                continue
            if end_delta > end_tolerance_after_sec:
                continue
            cost = abs(start_delta) + abs(end_delta)
            candidates.append((i, j, cost))

    candidates.sort(key=lambda x: x[2])

    matches: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for p_idx, g_idx, _ in candidates:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        matches.append((p_idx, g_idx))
        used_pred.add(p_idx)
        used_gt.add(g_idx)

    unmatched_pred = [i for i in range(len(predicted)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def match_rallies(
    predicted: list[Interval],
    ground_truth: list[Interval],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy matching of predicted to ground truth rallies.

    Returns:
        matches: list of (pred_idx, gt_idx) pairs
        unmatched_pred: false positive indices
        unmatched_gt: missed rally indices
    """
    if not predicted or not ground_truth:
        return [], list(range(len(predicted))), list(range(len(ground_truth)))

    iou_matrix = compute_iou_matrix(predicted, ground_truth)

    matches = []
    used_pred = set()
    used_gt = set()

    while True:
        if iou_matrix.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[max_idx]

        if max_iou < iou_threshold:
            break

        p_idx, g_idx = int(max_idx[0]), int(max_idx[1])
        matches.append((p_idx, g_idx))
        used_pred.add(p_idx)
        used_gt.add(g_idx)

        iou_matrix[p_idx, :] = 0
        iou_matrix[:, g_idx] = 0

    unmatched_pred = [i for i in range(len(predicted)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_gt]

    return matches, unmatched_pred, unmatched_gt


def evaluate(
    predicted: list[Interval],
    ground_truth: list[Interval],
    iou_threshold: float = 0.5,
    matching_mode: str = "iou",
    center_tolerance: float = 0.0,
    start_tolerance_sec: float = 0.5,
    end_tolerance_before_sec: float = 0.5,
    end_tolerance_after_sec: float = 1.5,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        matching_mode: one of
            - "iou": IoU-based matching at `iou_threshold` (default)
            - "center": center-based matching with `center_tolerance`
            - "boundary": asymmetric boundary-tolerance matching using
              `start_tolerance_sec` and `end_tolerance_{before,after}_sec`.
              Defaults match the SetOptics product tolerance spec
              (±0.5s start, -0.5s/+1.5s end).

    Returns dict with: rally_count_accuracy, precision, recall, f1,
    mean_iou, boundary MAEs, total_time_error, and detail lists for
    false positives/negatives.
    """
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    if n_gt > 0:
        count_accuracy = max(0, 1 - abs(n_pred - n_gt) / n_gt)
    else:
        count_accuracy = 1.0 if n_pred == 0 else 0.0

    if matching_mode == "center":
        matches, unmatched_pred, unmatched_gt = match_rallies_by_center(
            predicted, ground_truth, center_tolerance
        )
    elif matching_mode == "boundary":
        matches, unmatched_pred, unmatched_gt = match_rallies_by_boundary_tolerance(
            predicted,
            ground_truth,
            start_tolerance_sec=start_tolerance_sec,
            end_tolerance_before_sec=end_tolerance_before_sec,
            end_tolerance_after_sec=end_tolerance_after_sec,
        )
    else:
        matches, unmatched_pred, unmatched_gt = match_rallies(
            predicted, ground_truth, iou_threshold
        )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    ious = []
    start_errors = []
    end_errors = []
    for p_idx, g_idx in matches:
        ious.append(temporal_iou(predicted[p_idx], ground_truth[g_idx]))
        start_errors.append(
            abs(predicted[p_idx].start - ground_truth[g_idx].start))
        end_errors.append(
            abs(predicted[p_idx].end - ground_truth[g_idx].end))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mae_start = float(np.mean(start_errors)) if start_errors else 0.0
    mae_end = float(np.mean(end_errors)) if end_errors else 0.0

    total_pred = sum(r.duration for r in predicted)
    total_gt = sum(r.duration for r in ground_truth)
    if total_gt > 0:
        total_time_error = abs(total_pred - total_gt) / total_gt
    else:
        total_time_error = 0.0 if total_pred == 0 else 1.0

    fp_list = [
        {"index": i, "start": predicted[i].start, "end": predicted[i].end}
        for i in unmatched_pred
    ]
    fn_list = [
        {"index": i, "start": ground_truth[i].start,
         "end": ground_truth[i].end}
        for i in unmatched_gt
    ]

    return {
        "rally_count_predicted": n_pred,
        "rally_count_ground_truth": n_gt,
        "rally_count_accuracy": round(count_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "matched_rallies": tp,
        "false_positives": fp,
        "missed_rallies": fn,
        "boundary_mae_start_sec": round(mae_start, 2),
        "boundary_mae_end_sec": round(mae_end, 2),
        "total_rally_time_predicted_sec": round(total_pred, 1),
        "total_rally_time_gt_sec": round(total_gt, 1),
        "total_time_error": round(total_time_error, 4),
        "false_positive_details": fp_list,
        "false_negative_details": fn_list,
    }


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean metrics across multiple videos."""
    if not all_metrics:
        return {}

    keys = [
        "rally_count_accuracy", "precision", "recall", "f1",
        "mean_iou", "boundary_mae_start_sec", "boundary_mae_end_sec",
        "total_time_error",
    ]

    agg = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[key] = round(float(np.mean(values)), 4)

    agg["total_matched"] = sum(m["matched_rallies"] for m in all_metrics)
    agg["total_false_positives"] = sum(
        m["false_positives"] for m in all_metrics)
    agg["total_missed"] = sum(m["missed_rallies"] for m in all_metrics)
    agg["n_videos"] = len(all_metrics)

    return agg


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    print(f"  Rally count:    {metrics['rally_count_predicted']} predicted "
          f"vs {metrics['rally_count_ground_truth']} ground truth "
          f"(accuracy: {metrics['rally_count_accuracy']:.1%})")
    print(f"  Precision:      {metrics['precision']:.1%}")
    print(f"  Recall:         {metrics['recall']:.1%}")
    print(f"  F1:             {metrics['f1']:.1%}")
    print(f"  Mean IoU:       {metrics['mean_iou']:.3f}")
    print(f"  Matched:        {metrics['matched_rallies']} | "
          f"FP: {metrics['false_positives']} | "
          f"Missed: {metrics['missed_rallies']}")
    print(f"  Boundary MAE:   start={metrics['boundary_mae_start_sec']:.1f}s, "
          f"end={metrics['boundary_mae_end_sec']:.1f}s")
    print(f"  Rally time:     {metrics['total_rally_time_predicted_sec']:.0f}s "
          f"predicted vs {metrics['total_rally_time_gt_sec']:.0f}s actual "
          f"(error: {metrics['total_time_error']:.1%})")
