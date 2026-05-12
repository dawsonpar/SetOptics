"""
Per-frame binary classification metrics for Gemini optimization experiments.

Provides accuracy, precision, recall, F1, confusion matrix, and
per-category breakdowns. Follows the same pattern as temporal.py:
dataclass for results, pure metric functions, loader, printer.

Usage:
    from shared.eval.frame_classification import (
        FrameResult, evaluate_frames, print_frame_metrics,
    )
"""

from dataclasses import dataclass, field


@dataclass
class FrameResult:
    """Classification result for a single frame."""

    frame_id: str
    ground_truth: str
    predicted: str
    category: str = ""
    confidence: float = 1.0


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_confusion_matrix(
    results: list[FrameResult],
    positive_label: str = "rally",
) -> ConfusionMatrix:
    """Build binary confusion matrix from frame results."""
    cm = ConfusionMatrix()
    for r in results:
        is_positive = r.ground_truth == positive_label
        predicted_positive = r.predicted == positive_label
        if is_positive and predicted_positive:
            cm.tp += 1
        elif not is_positive and predicted_positive:
            cm.fp += 1
        elif is_positive and not predicted_positive:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def evaluate_frames(
    results: list[FrameResult],
    positive_label: str = "rally",
    categories: list[str] | None = None,
) -> dict:
    """Compute classification metrics from frame results.

    Returns dict with overall metrics and optional per-category breakdown.
    """
    if not results:
        return {"error": "No results to evaluate"}

    cm = build_confusion_matrix(results, positive_label)

    total = len(results)
    correct = cm.tp + cm.tn
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "total_frames": total,
        "accuracy": round(accuracy, 4),
        "precision": round(cm.precision, 4),
        "recall": round(cm.recall, 4),
        "f1": round(cm.f1, 4),
        "confusion_matrix": {
            "tp": cm.tp,
            "fp": cm.fp,
            "fn": cm.fn,
            "tn": cm.tn,
        },
    }

    if categories is None:
        seen = {r.category for r in results if r.category}
        categories = sorted(seen) if seen else []

    if categories:
        per_category = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue
            cat_cm = build_confusion_matrix(cat_results, positive_label)
            cat_total = len(cat_results)
            cat_correct = cat_cm.tp + cat_cm.tn
            per_category[cat] = {
                "total_frames": cat_total,
                "accuracy": round(
                    cat_correct / cat_total if cat_total > 0 else 0.0, 4
                ),
                "precision": round(cat_cm.precision, 4),
                "recall": round(cat_cm.recall, 4),
                "f1": round(cat_cm.f1, 4),
                "confusion_matrix": {
                    "tp": cat_cm.tp,
                    "fp": cat_cm.fp,
                    "fn": cat_cm.fn,
                    "tn": cat_cm.tn,
                },
            }
        metrics["per_category"] = per_category

    # Misclassified frames for error analysis
    errors = []
    for r in results:
        if r.ground_truth != r.predicted:
            errors.append({
                "frame_id": r.frame_id,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "category": r.category,
                "confidence": r.confidence,
            })
    metrics["errors"] = errors
    metrics["error_rate"] = round(len(errors) / total if total > 0 else 0.0, 4)

    return metrics


def print_frame_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print frame classification metrics."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    print(f"  Total frames:   {metrics['total_frames']}")
    print(f"  Accuracy:       {metrics['accuracy']:.1%}")
    print(f"  Precision:      {metrics['precision']:.1%}")
    print(f"  Recall:         {metrics['recall']:.1%}")
    print(f"  F1:             {metrics['f1']:.1%}")

    cm = metrics["confusion_matrix"]
    print(f"  Confusion:      TP={cm['tp']}  FP={cm['fp']}  "
          f"FN={cm['fn']}  TN={cm['tn']}")

    if "per_category" in metrics:
        print(f"\n  Per-category breakdown:")
        for cat, cat_m in metrics["per_category"].items():
            print(f"    {cat:<12} acc={cat_m['accuracy']:.1%}  "
                  f"F1={cat_m['f1']:.1%}  "
                  f"(n={cat_m['total_frames']})")

    print(f"  Errors:         {len(metrics.get('errors', []))} "
          f"({metrics.get('error_rate', 0):.1%})")
