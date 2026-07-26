"""
Leakage-free preprocessing pipelines.

All imputers and scalers are fitted on training data only.
Validation and test data are transformed using training statistics.
sklearn.Pipeline is used so there is no risk of accidentally calling
fit_transform on the full dataset.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude metadata columns)."""
    exclude = {"Subject", "Task", "label", "binary", "split",
               "recording_id", "start_s", "end_s"}
    return [c for c in df.columns if c not in exclude]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline: median imputation → standard scaling.

    Calling pipeline.fit(X_train) fits both steps on training data only.
    pipeline.transform(X_test) applies training statistics to test data.
    This structure makes leakage physically impossible.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def prepare_subject_splits(
    sub_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> dict[str, Any] | None:
    """
    Prepare leakage-free train/val/test arrays for one subject.

    Assumes `sub_df` already has a `split` column ('train'/'val'/'test')
    produced by the chronological splitting step.

    Parameters
    ----------
    sub_df : pd.DataFrame
        Rows for one subject, containing feature_cols + label_col + 'split'.
    feature_cols : list[str]
        Feature column names.
    label_col : str
        Target column name.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test, pipeline
    or None if any split is empty or has fewer than 2 classes.
    """
    splits = {}
    for sp in ("train", "val", "test"):
        mask = sub_df["split"] == sp
        if mask.sum() == 0:
            logger.warning("Split '%s' is empty — skipping subject.", sp)
            return None
        splits[sp] = sub_df[mask]

    train_df = splits["train"]
    val_df   = splits["val"]
    test_df  = splits["test"]

    if len(train_df[label_col].unique()) < 2:
        logger.warning("Training split has fewer than 2 classes — skipping subject.")
        return None

    X_train = train_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).values
    X_val   = val_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).values
    X_test  = test_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).values

    y_train_raw = train_df[label_col].values
    y_val_raw   = val_df[label_col].values
    y_test_raw  = test_df[label_col].values

    # Encode string labels to integers (required by XGBoost; harmless for others)
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)   if set(y_val_raw).issubset(set(le.classes_)) else y_val_raw
    y_test  = le.transform(y_test_raw)  if set(y_test_raw).issubset(set(le.classes_)) else y_test_raw

    pipe = build_preprocessing_pipeline()
    X_train = pipe.fit_transform(X_train)   # fitted on train only
    X_val   = pipe.transform(X_val)         # training statistics applied
    X_test  = pipe.transform(X_test)        # training statistics applied

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "pipeline": pipe,
        "label_encoder": le,
    }


def prepare_unseen_subject_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> dict[str, Any]:
    """
    Prepare leakage-free arrays for one LOSO fold.

    Imputer and scaler are fitted on train_df only.
    test_df is transformed using training statistics.
    The test subject's data never influences preprocessing.
    """
    X_train = train_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).values
    X_test  = test_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).values
    y_train_raw = train_df[label_col].values
    y_test_raw  = test_df[label_col].values

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw) if set(y_test_raw).issubset(set(le.classes_)) else y_test_raw

    pipe = build_preprocessing_pipeline()
    X_train = pipe.fit_transform(X_train)
    X_test  = pipe.transform(X_test)

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "pipeline": pipe,
        "label_encoder": le,
    }
