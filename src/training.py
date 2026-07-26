"""
Training orchestration — personalized and unseen-subject protocols.

Both protocols use leakage-free preprocessing (see preprocessing.py).
All results are collected as dicts and saved via evaluation.py.
"""

from __future__ import annotations

import copy
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import binary_metrics, multiclass_metrics
from src.models import build_models, train_and_time, inference_latency
from src.preprocessing import get_feature_cols, prepare_subject_splits, prepare_unseen_subject_fold

logger = logging.getLogger(__name__)


def run_personalized_evaluation(
    df: pd.DataFrame,
    label_col: str,
    model_cfg: Any,
    task_type: str = "binary",
    label_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Per-subject within-subject evaluation.

    The `split` column in `df` must already encode chronological
    train/val/test blocks (produced by splitting.py).

    Parameters
    ----------
    df : pd.DataFrame
        Full feature table with 'Subject', 'split', and label_col columns.
    label_col : str
        Column name for the classification target.
    model_cfg : ModelConfig
    task_type : str
        'binary' or 'multiclass'
    label_names : list[str] | None

    Returns
    -------
    per_subject_rows : list[dict]  — one dict per (subject, model)
    comparison_rows  : list[dict]  — model-level comparison (across all subjects)
    """
    feature_cols    = get_feature_cols(df)
    subjects        = sorted(df["Subject"].unique())
    models_template = build_models(model_cfg)

    per_subject_rows: list[dict[str, Any]] = []

    for sub in subjects:
        sub_df = df[df["Subject"] == sub].copy()
        prepared = prepare_subject_splits(sub_df, feature_cols, label_col)
        if prepared is None:
            continue

        X_train = prepared["X_train"]
        X_val   = prepared["X_val"]
        X_test  = prepared["X_test"]
        y_train = prepared["y_train"]
        y_val   = prepared["y_val"]
        y_test  = prepared["y_test"]

        for model_name, model_template in models_template.items():
            model = copy.deepcopy(model_template)
            # Combine train+val for final model fit (val was used for monitoring only)
            X_fit = np.vstack([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])

            fitted_model, train_time = train_and_time(model, X_fit, y_fit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = fitted_model.predict(X_test)
                y_prob = None
                if hasattr(fitted_model, "predict_proba"):
                    y_prob = fitted_model.predict_proba(X_test)

            if task_type == "binary":
                prob_1d = y_prob[:, 1] if (y_prob is not None and y_prob.ndim == 2) else y_prob
                metrics = binary_metrics(y_test, y_pred, prob_1d, label_names=label_names)
            else:
                metrics = multiclass_metrics(y_test, y_pred, label_names=label_names)

            lat = inference_latency(fitted_model, X_test)

            row = {
                "subject":       sub,
                "model":         model_name,
                "protocol":      "personalized",
                "task_type":     task_type,
                "n_train":       len(y_train),
                "n_val":         len(y_val),
                "n_test":        len(y_test),
                "train_time_s":  round(train_time, 4),
                "p50_latency_us": lat["p50_us"],
                "p95_latency_us": lat["p95_us"],
                **{k: v for k, v in metrics.items()
                   if not isinstance(v, (list, dict))},
            }
            per_subject_rows.append(row)
            logger.debug("  %s | %s | acc=%.3f", sub, model_name, metrics.get("accuracy", float("nan")))

    return per_subject_rows


def run_unseen_subject_evaluation(
    df: pd.DataFrame,
    label_col: str,
    model_cfg: Any,
    folds: list[tuple[list[str], list[str]]],
    task_type: str = "binary",
    label_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Leave-One-Subject-Out (LOSO) evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature table.
    folds : list of (train_subjects, test_subjects)
        From splitting.subject_grouped_folds().

    Returns
    -------
    fold_rows : list[dict]  — one dict per (fold, model)
    """
    feature_cols    = get_feature_cols(df)
    models_template = build_models(model_cfg)
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_subs, test_subs) in enumerate(folds):
        test_sub = test_subs[0] if len(test_subs) == 1 else ",".join(test_subs)
        train_df = df[df["Subject"].isin(train_subs)].copy()
        test_df  = df[df["Subject"].isin(test_subs)].copy()

        if len(train_df[label_col].unique()) < 2 or len(test_df) == 0:
            logger.warning("Fold %d: skipping (insufficient data).", fold_idx)
            continue

        for model_name, model_template in models_template.items():
            model    = copy.deepcopy(model_template)
            prepared = prepare_unseen_subject_fold(train_df, test_df, feature_cols, label_col)
            fitted_model, train_time = train_and_time(
                model, prepared["X_train"], prepared["y_train"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = fitted_model.predict(prepared["X_test"])
                y_prob = None
                if hasattr(fitted_model, "predict_proba"):
                    y_prob = fitted_model.predict_proba(prepared["X_test"])

            if task_type == "binary":
                prob_1d = y_prob[:, 1] if (y_prob is not None and y_prob.ndim == 2) else y_prob
                metrics = binary_metrics(prepared["y_test"], y_pred, prob_1d, label_names=label_names)
            else:
                metrics = multiclass_metrics(prepared["y_test"], y_pred, label_names=label_names)

            lat = inference_latency(fitted_model, prepared["X_test"])
            row = {
                "fold":          fold_idx,
                "test_subject":  test_sub,
                "model":         model_name,
                "protocol":      "unseen_subject",
                "task_type":     task_type,
                "n_train":       len(prepared["y_train"]),
                "n_test":        len(prepared["y_test"]),
                "train_time_s":  round(train_time, 4),
                "p50_latency_us": lat["p50_us"],
                "p95_latency_us": lat["p95_us"],
                **{k: v for k, v in metrics.items()
                   if not isinstance(v, (list, dict))},
            }
            fold_rows.append(row)
            logger.debug(
                "  Fold %d test=%s | %s | acc=%.3f",
                fold_idx, test_sub, model_name, metrics.get("accuracy", float("nan")),
            )

    return fold_rows
