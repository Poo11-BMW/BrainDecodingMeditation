"""
Metric computation — comprehensive, honest, and interview-defensible.

Binary metrics: accuracy, balanced accuracy, macro F1, precision, recall,
                ROC-AUC, PR-AUC, sensitivity, specificity, confusion matrix.
Multiclass:     accuracy, balanced accuracy, macro F1, weighted F1,
                per-class precision/recall, confusion matrix.

Bootstrap 95% CI is computed for scalar metrics when enough samples exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 500,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a scalar classification metric."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not scores:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))


# ── Binary classification metrics ─────────────────────────────────────────────

def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """
    Full binary classification metric suite.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)  — integer labels {0, 1}
    y_pred : np.ndarray, shape (n,)  — predicted integer labels
    y_prob : np.ndarray, shape (n,) or (n, 2) — probability of positive class
    label_names : list[str] | None  — class names for display
    bootstrap : bool — compute 95% CI via bootstrap

    Returns
    -------
    dict of metric_name → value
    """
    acc  = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()

    # Sensitivity / Specificity from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) == 2 else (0, 0, 0, 0)
    sensitivity = float(tp / (tp + fn + 1e-12))
    specificity = float(tn / (tn + fp + 1e-12))

    result: dict[str, Any] = {
        "accuracy":         acc,
        "balanced_accuracy": bacc,
        "macro_f1":         f1,
        "macro_precision":  prec,
        "macro_recall":     rec,
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "confusion_matrix": cm,
        "n_samples":        int(len(y_true)),
    }

    if y_prob is not None:
        prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
            result["pr_auc"]  = float(average_precision_score(y_true, prob_pos))
        except Exception:
            result["roc_auc"] = float("nan")
            result["pr_auc"]  = float("nan")

    if bootstrap and len(y_true) >= 20:
        rng = np.random.default_rng(42)
        lo, hi = _bootstrap_ci(y_true, y_pred, accuracy_score, rng=rng)
        result["accuracy_ci95"] = [lo, hi]
        lo, hi = _bootstrap_ci(y_true, y_pred,
                               lambda a, b: f1_score(a, b, average="macro", zero_division=0),
                               rng=rng)
        result["macro_f1_ci95"] = [lo, hi]

    if label_names:
        result["label_names"] = label_names

    return result


# ── Multiclass classification metrics ─────────────────────────────────────────

def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Full multiclass metric suite."""
    acc   = float(accuracy_score(y_true, y_pred))
    bacc  = float(balanced_accuracy_score(y_true, y_pred))
    f1_m  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_w  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec  = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec   = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm    = confusion_matrix(y_true, y_pred).tolist()

    result: dict[str, Any] = {
        "accuracy":          acc,
        "balanced_accuracy": bacc,
        "macro_f1":          f1_m,
        "weighted_f1":       f1_w,
        "per_class_precision": prec,
        "per_class_recall":    rec,
        "confusion_matrix":  cm,
        "n_samples":         int(len(y_true)),
    }
    if label_names:
        result["label_names"] = label_names
    return result


# ── Aggregation across subjects / folds ───────────────────────────────────────

def aggregate_results(
    per_fold: list[dict[str, Any]],
    scalar_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute mean, std, median, min, max across subjects or folds.

    Parameters
    ----------
    per_fold : list[dict]
        One dict of metrics per subject or fold.
    scalar_keys : list[str] | None
        Which keys to aggregate. If None, aggregates all numeric scalars.
    """
    if not per_fold:
        return {}
    all_keys = scalar_keys or [
        k for k, v in per_fold[0].items()
        if isinstance(v, (int, float)) and not np.isnan(v)
    ]
    agg: dict[str, Any] = {}
    for k in all_keys:
        vals = [r[k] for r in per_fold if isinstance(r.get(k), (int, float)) and not np.isnan(r.get(k, float("nan")))]
        if not vals:
            continue
        agg[k] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "median": float(np.median(vals)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
            "n":      len(vals),
        }
    return agg


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Save a metrics dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics → %s", path)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a metrics JSON file."""
    with open(path) as f:
        return json.load(f)
