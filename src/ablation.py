"""
Feature ablation experiments.

Runs personalized evaluation with:
  1. All features
  2. Each feature group independently
  3. All features minus one group at a time (leave-one-group-out)

Results are written to results/feature_ablation.csv.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import binary_metrics
from src.feature_extraction import get_feature_groups
from src.models import build_models, train_and_time
from src.preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)


def _evaluate_feature_subset(
    df: pd.DataFrame,
    selected_features: list[str],
    label_col: str,
    model_cfg: Any,
    model_name: str = "LightGBM",
) -> dict[str, float]:
    """
    Run personalized evaluation for a specific feature subset.
    Returns aggregate metrics across all subjects.
    """
    from src.preprocessing import get_feature_cols
    models_template = build_models(model_cfg)
    model_template = models_template.get(model_name)
    if model_template is None:
        # Fall back to first available
        model_template = next(iter(models_template.values()))

    subjects = sorted(df["Subject"].unique())
    accs = []
    f1s  = []

    for sub in subjects:
        sub_df = df[df["Subject"] == sub].copy()
        for sp_col in ("split",):
            if sp_col not in sub_df.columns:
                logger.warning("No 'split' column in df for ablation — skipping %s", sub)
                return {}

        train_df = sub_df[sub_df["split"] == "train"]
        test_df  = sub_df[sub_df["split"] == "test"]

        if len(train_df) < 5 or len(test_df) < 2:
            continue
        if len(train_df[label_col].unique()) < 2:
            continue

        valid_feats = [f for f in selected_features if f in sub_df.columns]
        if not valid_feats:
            continue

        X_train = train_df[valid_feats].replace([np.inf, -np.inf], np.nan).values
        X_test  = test_df[valid_feats].replace([np.inf, -np.inf], np.nan).values
        y_train = train_df[label_col].values
        y_test  = test_df[label_col].values

        pipe = build_preprocessing_pipeline()
        X_train = pipe.fit_transform(X_train)
        X_test  = pipe.transform(X_test)

        model = copy.deepcopy(model_template)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = binary_metrics(y_test, y_pred, bootstrap=False)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])

    if not accs:
        return {"mean_accuracy": float("nan"), "mean_macro_f1": float("nan"), "n_subjects": 0}
    return {
        "mean_accuracy":  float(np.mean(accs)),
        "std_accuracy":   float(np.std(accs)),
        "mean_macro_f1":  float(np.mean(f1s)),
        "std_macro_f1":   float(np.std(f1s)),
        "n_subjects":     len(accs),
    }


def run_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model_cfg: Any,
    model_name: str = "LightGBM",
) -> pd.DataFrame:
    """
    Run the full feature ablation study.

    Returns
    -------
    pd.DataFrame with columns:
        condition, features_used, n_features, mean_accuracy, std_accuracy,
        mean_macro_f1, std_macro_f1, n_subjects
    """
    groups = get_feature_groups(feature_cols)
    rows   = []

    # All features together
    logger.info("Ablation: all features (%d)", len(feature_cols))
    m = _evaluate_feature_subset(df, feature_cols, label_col, model_cfg, model_name)
    rows.append({"condition": "all_features", "features_used": "all", "n_features": len(feature_cols), **m})

    # Each group independently
    for group_name, feats in groups.items():
        logger.info("Ablation: only %s (%d features)", group_name, len(feats))
        m = _evaluate_feature_subset(df, feats, label_col, model_cfg, model_name)
        rows.append({
            "condition":    f"only_{group_name}",
            "features_used": group_name,
            "n_features":   len(feats),
            **m,
        })

    # All minus one group
    for group_name, feats in groups.items():
        remaining = [f for f in feature_cols if f not in feats]
        if not remaining:
            continue
        logger.info("Ablation: all_minus_%s (%d features)", group_name, len(remaining))
        m = _evaluate_feature_subset(df, remaining, label_col, model_cfg, model_name)
        rows.append({
            "condition":    f"all_minus_{group_name}",
            "features_used": f"all except {group_name}",
            "n_features":   len(remaining),
            **m,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    return result
