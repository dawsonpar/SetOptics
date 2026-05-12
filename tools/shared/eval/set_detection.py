"""
Shared point-event evaluation for set moment detection.

Set detection is a point-event problem (single contact timestamp)
rather than interval-based like rally detection. This module provides
matching logic and metrics appropriate for scalar timestamps.

Usage:
    from shared.eval.set_detection import (
        SetMoment, load_set_annotations, evaluate_set_detection,
        aggregate_set_metrics, print_set_metrics,
    )
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SetMoment:
    """A single detected or annotated set moment."""

    set_moment_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    confidence: float = 1.0


def load_set_annotations(path: str | Path) -> list[SetMoment]:
    """Load ground truth set moments from annotation JSON.

    Expected format:
        {"sets": [{"set_moment_seconds": float,
                   "clip_start_seconds": float,
                   "clip_end_seconds": float}, ...]}
    """
    with open(path) as f:
        data = json.load(f)

    moments = []
    for s in data.get("sets", []):
        moments.append(SetMoment(
            set_moment_seconds=float(s["set_moment_seconds"]),
            clip_start_seconds=float(s["clip_start_seconds"]),
            clip_end_seconds=float(s["clip_end_seconds"]),
        ))
    return moments


def _greedy_match(
    predicted: list[SetMoment],
    ground_truth: list[SetMoment],
    tolerance_sec: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match predicted moments to GT moments by timestamp proximity.

    A prediction matches a GT moment if |pred - gt| <= tolerance_sec.
    Closest pair matched first; each moment matched at most once.

    Returns:
        matches: list of (pred_idx, gt_idx) pairs
        unmatched_pred: false positive indices
        unmatched_gt: false negative indices
    """
    candidates = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            dist = abs(p.set_moment_seconds - g.set_moment_seconds)
            if dist <= tolerance_sec:
                candidates.append((i, j, dist))

    candidates.sort(key=lambda x: x[2])

    matches = []
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


def evaluate_set_detection(
    predicted: list[SetMoment],
    ground_truth: list[SetMoment],
    moment_tolerance_sec: float = 1.5,
) -> dict:
    """Compute set detection metrics at a given tolerance.

    Args:
        predicted: Detector output moments.
        ground_truth: Human-annotated ground truth moments.
        moment_tolerance_sec: Max allowed |pred - gt| for a true positive.

    Returns dict with: precision, recall, f1, count_accuracy,
    moment_mae_sec, false_positive_details, false_negative_details.
    """
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    if n_gt > 0:
        count_accuracy = max(0.0, 1.0 - abs(n_pred - n_gt) / n_gt)
    else:
        count_accuracy = 1.0 if n_pred == 0 else 0.0

    matches, unmatched_pred, unmatched_gt = _greedy_match(
        predicted, ground_truth, moment_tolerance_sec,
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    moment_errors = [
        abs(
            predicted[p_idx].set_moment_seconds
            - ground_truth[g_idx].set_moment_seconds
        )
        for p_idx, g_idx in matches
    ]
    moment_mae = float(np.mean(moment_errors)) if moment_errors else 0.0

    fp_details = [
        {
            "index": i,
            "set_moment_seconds": predicted[i].set_moment_seconds,
            "confidence": predicted[i].confidence,
        }
        for i in unmatched_pred
    ]
    fn_details = [
        {
            "index": i,
            "set_moment_seconds": ground_truth[i].set_moment_seconds,
        }
        for i in unmatched_gt
    ]

    return {
        "tolerance_sec": moment_tolerance_sec,
        "count_predicted": n_pred,
        "count_ground_truth": n_gt,
        "count_accuracy": round(count_accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "moment_mae_sec": round(moment_mae, 3),
        "false_positive_details": fp_details,
        "false_negative_details": fn_details,
    }


def evaluate_at_tolerances(
    predicted: list[SetMoment],
    ground_truth: list[SetMoment],
    tolerances: list[float] | None = None,
) -> dict[float, dict]:
    """Evaluate at multiple tolerances.

    Args:
        tolerances: List of tolerance values in seconds.
                    Defaults to [0.5, 1.0, 1.5, 2.0].

    Returns:
        Dict mapping tolerance -> metrics dict.
    """
    if tolerances is None:
        tolerances = [0.5, 1.0, 1.5, 2.0]
    return {t: evaluate_set_detection(predicted, ground_truth, t)
            for t in tolerances}


def aggregate_set_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean metrics across multiple videos (at same tolerance)."""
    if not all_metrics:
        return {}

    keys = ["count_accuracy", "precision", "recall", "f1", "moment_mae_sec"]
    agg = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[key] = round(float(np.mean(values)), 4)

    agg["total_true_positives"] = sum(m["true_positives"] for m in all_metrics)
    agg["total_false_positives"] = sum(
        m["false_positives"] for m in all_metrics)
    agg["total_false_negatives"] = sum(
        m["false_negatives"] for m in all_metrics)
    agg["n_videos"] = len(all_metrics)

    return agg


def print_set_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print set detection metrics."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    tol = metrics.get("tolerance_sec", "?")
    print(f"  Tolerance:      ±{tol}s")
    print(
        f"  Set count:      {metrics['count_predicted']} predicted vs "
        f"{metrics['count_ground_truth']} ground truth "
        f"(accuracy: {metrics['count_accuracy']:.1%})"
    )
    print(f"  Precision:      {metrics['precision']:.1%}")
    print(f"  Recall:         {metrics['recall']:.1%}")
    print(f"  F1:             {metrics['f1']:.1%}")
    print(
        f"  Matched:        {metrics['true_positives']} | "
        f"FP: {metrics['false_positives']} | "
        f"Missed: {metrics['false_negatives']}"
    )
    print(f"  Moment MAE:     {metrics['moment_mae_sec']:.2f}s")
