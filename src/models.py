"""
Model definitions with deterministic seeds and shared hyperparameters.

All models are constructed via factory functions that accept a ModelConfig
so every experiment uses identical settings.
"""

from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def _try_import_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def build_models(model_cfg: Any) -> dict[str, Any]:
    """
    Return a dict of model_name → unfitted sklearn-compatible classifier.

    Parameters
    ----------
    model_cfg : ModelConfig
        Hyperparameter configuration.
    """
    seed = model_cfg.random_seed
    models: dict[str, Any] = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent", random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.rf_max_depth,
            min_samples_leaf=model_cfg.rf_min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.rf_max_depth,
            min_samples_leaf=model_cfg.rf_min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    XGBClassifier = _try_import_xgb()
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.xgb_max_depth,
            learning_rate=model_cfg.xgb_learning_rate,
            subsample=model_cfg.xgb_subsample,
            colsample_bytree=model_cfg.xgb_colsample_bytree,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=seed,
        )

    LGBMClassifier = _try_import_lgbm()
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.lgbm_learning_rate,
            num_leaves=model_cfg.lgbm_num_leaves,
            verbose=-1,
            random_state=seed,
        )

    return models


def train_and_time(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, float]:
    """Fit `model` and return (fitted_model, training_time_seconds)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def inference_latency(
    model: Any,
    X: np.ndarray,
    n_repeats: int = 200,
) -> dict[str, float]:
    """
    Measure per-sample inference latency (p50, p95) in microseconds.

    Parameters
    ----------
    model : fitted classifier
    X : np.ndarray, shape (n_samples, n_features)
    n_repeats : int
        Number of single-sample predictions to time.

    Returns
    -------
    dict with keys p50_us and p95_us
    """
    times_us = []
    for i in range(min(n_repeats, len(X))):
        sample = X[i: i + 1]
        t0 = time.perf_counter()
        model.predict(sample)
        times_us.append((time.perf_counter() - t0) * 1e6)
    return {
        "p50_us": float(np.percentile(times_us, 50)),
        "p95_us": float(np.percentile(times_us, 95)),
    }
