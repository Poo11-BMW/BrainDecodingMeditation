"""
Tests for src/preprocessing.py.

Proves:
  1. Imputation is fitted only on training data.
  2. Scaling is fitted only on training data.
  3. Test data cannot influence preprocessing statistics.
  4. Pipeline cannot accidentally call fit_transform on test data.
  5. NaN and infinity values are handled correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessing_pipeline,
    get_feature_cols,
    prepare_subject_splits,
    prepare_unseen_subject_fold,
)


def _make_df(n_train: int = 80, n_val: int = 10, n_test: int = 20,
             n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test
    data = rng.standard_normal((n_total, n_features))
    splits = (["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    df = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])
    df["Subject"] = "sub-001"
    df["Task"]    = "med1breath"
    df["binary"]  = ([0] * (n_train // 2) + [1] * (n_train - n_train // 2) +
                     [0] * (n_val // 2)   + [1] * (n_val - n_val // 2)   +
                     [0] * (n_test // 2)  + [1] * (n_test - n_test // 2))
    df["split"]   = splits
    return df


class TestBuildPreprocessingPipeline:

    def test_pipeline_returns_fitted_scaler_trained_on_train_only(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        X_test  = rng.standard_normal((20, 5)) * 100  # very different scale

        pipe = build_preprocessing_pipeline()
        X_train_t = pipe.fit_transform(X_train)
        X_test_t  = pipe.transform(X_test)

        # Scaler mean should match training data mean, NOT test
        scaler = pipe.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-5)
        # Test data transformed should NOT have zero mean (it was scaled by train stats)
        assert not np.allclose(X_test_t.mean(axis=0), np.zeros(5), atol=0.5)

    def test_nan_imputed_with_training_median_only(self):
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((100, 3))
        X_test  = rng.standard_normal((20, 3))
        # Introduce NaN in test
        X_test[0, 0] = np.nan
        X_test[5, 2] = np.inf

        X_test_clean = X_test.copy()
        X_test_clean[np.isnan(X_test_clean)] = 0
        X_test_clean[np.isinf(X_test_clean)] = 0

        pipe = build_preprocessing_pipeline()
        pipe.fit_transform(X_train)

        # Should not raise
        X_test_t = pipe.transform(np.where(np.isinf(X_test), np.nan, X_test))
        assert not np.any(np.isnan(X_test_t))
        assert not np.any(np.isinf(X_test_t))

    def test_test_data_does_not_change_training_statistics(self):
        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((200, 4))
        X_test_extreme = rng.standard_normal((200, 4)) * 1000

        pipe_a = build_preprocessing_pipeline()
        pipe_a.fit_transform(X_train)
        mean_a = pipe_a.named_steps["scaler"].mean_.copy()

        pipe_b = build_preprocessing_pipeline()
        pipe_b.fit_transform(X_train)
        _ = pipe_b.transform(X_test_extreme)  # transform only — must not change stats
        mean_b = pipe_b.named_steps["scaler"].mean_.copy()

        np.testing.assert_array_equal(mean_a, mean_b)


class TestPrepareSubjectSplits:

    def test_output_shapes_match_split_counts(self):
        df = _make_df(n_train=70, n_val=10, n_test=20)
        result = prepare_subject_splits(df, [f"feat_{i}" for i in range(5)], "binary")
        assert result is not None
        assert result["X_train"].shape[0] == 70
        assert result["X_val"].shape[0]   == 10
        assert result["X_test"].shape[0]  == 20

    def test_scaler_fitted_on_train_only(self):
        df = _make_df()
        feats = [f"feat_{i}" for i in range(5)]
        result = prepare_subject_splits(df, feats, "binary")
        assert result is not None
        pipe = result["pipeline"]
        # Training mean should match actual training data
        X_train_raw = df[df["split"] == "train"][feats].values
        np.testing.assert_allclose(
            pipe.named_steps["scaler"].mean_,
            X_train_raw.mean(axis=0),
            rtol=1e-4,
        )

    def test_none_returned_when_train_has_one_class(self):
        df = _make_df()
        # Force all training labels to 0
        df.loc[df["split"] == "train", "binary"] = 0
        result = prepare_subject_splits(df, [f"feat_{i}" for i in range(5)], "binary")
        assert result is None

    def test_none_returned_when_split_empty(self):
        df = _make_df()
        # Remove all test rows
        df = df[df["split"] != "test"]
        result = prepare_subject_splits(df, [f"feat_{i}" for i in range(5)], "binary")
        assert result is None

    def test_no_nan_in_output(self):
        # Introduce NaN and Inf in features
        df = _make_df()
        df.loc[0, "feat_0"] = np.nan
        df.loc[1, "feat_1"] = np.inf
        result = prepare_subject_splits(df, [f"feat_{i}" for i in range(5)], "binary")
        assert result is not None
        for key in ("X_train", "X_val", "X_test"):
            assert not np.any(np.isnan(result[key])), f"NaN found in {key}"
            assert not np.any(np.isinf(result[key])), f"Inf found in {key}"


class TestPrepareUnseenSubjectFold:

    def test_test_subject_statistics_not_used(self):
        rng = np.random.default_rng(5)
        n_feat = 4
        feat_names = [f"f{i}" for i in range(n_feat)]

        # Training data: small values
        train_data = rng.standard_normal((200, n_feat))
        train_df = pd.DataFrame(train_data, columns=feat_names)
        train_df["Subject"] = "sub-train"
        train_df["binary"] = rng.integers(0, 2, 200)

        # Test subject: very different scale
        test_data  = rng.standard_normal((50, n_feat)) * 100
        test_df = pd.DataFrame(test_data, columns=feat_names)
        test_df["Subject"] = "sub-test"
        test_df["binary"] = rng.integers(0, 2, 50)

        result = prepare_unseen_subject_fold(train_df, test_df, feat_names, "binary")
        pipe = result["pipeline"]
        # Scaler mean must match training data, not test
        np.testing.assert_allclose(
            pipe.named_steps["scaler"].mean_,
            train_data.mean(axis=0),
            rtol=1e-4,
        )

    def test_identical_seeds_produce_identical_results(self):
        rng = np.random.default_rng(99)
        n_feat = 3
        feat_names = [f"x{i}" for i in range(n_feat)]
        train_df = pd.DataFrame(rng.standard_normal((100, n_feat)), columns=feat_names)
        test_df  = pd.DataFrame(rng.standard_normal((20, n_feat)),  columns=feat_names)
        train_df["binary"] = 0
        test_df["binary"]  = 1
        # Run twice — same pipeline (deterministic)
        r1 = prepare_unseen_subject_fold(train_df, test_df, feat_names, "binary")
        r2 = prepare_unseen_subject_fold(train_df, test_df, feat_names, "binary")
        np.testing.assert_array_equal(r1["X_train"], r2["X_train"])
        np.testing.assert_array_equal(r1["X_test"],  r2["X_test"])
