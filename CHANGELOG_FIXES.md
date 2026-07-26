# CHANGELOG — Evaluation Fixes

**Date**: 2026-07-26  
**Scope**: Correctness, reproducibility, and scientific defensibility

---

## Root Causes Found

### 1. Overlapping-epoch leakage (CRITICAL)
**Where**: `run_pipeline.py:136-141`, `per_subject_model.py:52-57`  
**What**: Epochs were created with 50% overlap using `make_fixed_length_events(..., overlap=0.5)`. Adjacent epochs share 1 second of raw signal. When `StratifiedShuffleSplit` randomly assigns these to train and test, the model effectively sees test-set signal during training.  
**Effect**: Artificially inflated accuracy on the test set.  
**Fix**: `src/splitting.py` — raw recording time is split chronologically before any epoch is generated. Non-overlapping epochs are then created within each partition. A safety gap of one epoch length is enforced at partition boundaries.

### 2. Scaler leakage (CRITICAL)
**Where**: `per_subject_model.py:50`, `prove_it.py:253-255`  
**What**: `StandardScaler().fit_transform(X)` was called on the entire subject dataset before any train/test split. The scaler's mean and standard deviation included test-set statistics.  
**Fix**: `src/preprocessing.py` — `sklearn.Pipeline` wraps `SimpleImputer` + `StandardScaler`. The pipeline is `fit_transform`'d on training data only. Test data is `transform`'d using training statistics.

### 3. Imputer leakage (CRITICAL)
**Where**: `per_subject_model.py:39`, `prove_it.py:108`  
**What**: `X.fillna(X.median())` filled NaN values using the median computed over train + test combined.  
**Fix**: `SimpleImputer(strategy="median")` inside the `sklearn.Pipeline`, fitted on training data only.

### 4. No unseen-subject evaluation (CRITICAL — scientific misrepresentation)
**Where**: README, `per_subject_model.py`  
**What**: The headline "93% accuracy" was personalized within-subject performance. No evaluation on completely unseen subjects was performed. The README did not make this distinction clear.  
**Fix**: `scripts/run_unseen_subject_evaluation.py` — Leave-One-Subject-Out evaluation using `src/splitting.subject_grouped_folds`. Test subject's data is never used in preprocessing, feature selection, or model training. Both protocols are now clearly labelled in the README.

### 5. Calibration analysis — random epoch subsampling instead of chronological (HIGH)
**Where**: `prove_it.py:196-204`  
**What**: The calibration experiment sampled `n` random epochs from the training pool. Because `StratifiedShuffleSplit` randomises epoch order, this did not simulate "the first N seconds of recording" — it sampled from the full timeline.  
**Fix**: `src/training.py` calibration logic uses only the earliest `n` epochs chronologically from the training partition.

### 6. Cross-subject baseline uses one random 80/20 split (HIGH)
**Where**: `run_pipeline.py:244-253`  
**What**: One random split of 16 train / 4 test subjects gave a single point estimate with high variance and no confidence interval.  
**Fix**: LOSO with 20 folds (one per subject) gives 20 estimates, mean ± std reported.

### 7. Feature importance trained on all data (LOW — exploratory mislabelling)
**Where**: `prove_it.py:251-261`  
**What**: `rf_all.fit(Xs_all, y_all)` trained on the full dataset. This is valid for descriptive feature ranking but was not labelled as such.  
**Fix**: Feature importance is now computed from the training fold only, clearly labelled as exploratory.

---

## Files Changed

| Action | File |
|---|---|
| NEW | `src/__init__.py` |
| NEW | `src/config.py` |
| NEW | `src/data_loading.py` |
| NEW | `src/preprocessing.py` |
| NEW | `src/epoching.py` |
| NEW | `src/feature_extraction.py` |
| NEW | `src/splitting.py` |
| NEW | `src/models.py` |
| NEW | `src/training.py` |
| NEW | `src/evaluation.py` |
| NEW | `src/ablation.py` |
| NEW | `src/visualization.py` |
| NEW | `src/inference.py` |
| NEW | `scripts/build_features.py` |
| NEW | `scripts/run_personalized_evaluation.py` |
| NEW | `scripts/run_unseen_subject_evaluation.py` |
| NEW | `scripts/run_ablation.py` |
| NEW | `scripts/generate_report.py` |
| NEW | `configs/default.yaml` |
| NEW | `configs/test.yaml` |
| NEW | `tests/__init__.py` |
| NEW | `tests/conftest.py` |
| NEW | `tests/test_splitting.py` |
| NEW | `tests/test_preprocessing.py` |
| NEW | `tests/test_epoching.py` |
| NEW | `tests/test_features.py` |
| NEW | `tests/test_evaluation.py` |
| NEW | `.github/workflows/ci.yml` |
| NEW | `docs/evaluation_audit.md` |
| NEW | `CHANGELOG_FIXES.md` |
| REWRITTEN | `README.md` |
| REPLACED | `requirements.txt` (was `requirement.txt`, now version-pinned) |
| PRESERVED | `run_pipeline.py` (original, kept for reference) |
| PRESERVED | `per_subject_model.py` (original, kept for reference) |
| PRESERVED | `prove_it.py` (original, kept for reference) |
| PRESERVED | `brain/rich_features.csv` (feature data from original pipeline) |

---

## Leakage Issues Fixed

| # | Issue | Severity | Fixed? |
|---|---|---|---|
| 1 | Overlapping epochs across train/test | CRITICAL | ✅ Yes |
| 2 | Scaler fitted before split | CRITICAL | ✅ Yes |
| 3 | Imputer fitted before split | CRITICAL | ✅ Yes |
| 4 | No unseen-subject evaluation | CRITICAL | ✅ Yes |
| 5 | Calibration uses random not chronological subsampling | HIGH | ✅ Yes |
| 6 | Cross-subject uses single random split | HIGH | ✅ Yes |
| 7 | Feature importance on all data | LOW | ✅ Fixed (labelled correctly) |

---

## Tests Added

| Test file | Tests | What is proved |
|---|---|---|
| `test_splitting.py` | 12 | No time overlap across splits; epochs stay within boundaries; LOSO subject isolation |
| `test_preprocessing.py` | 9 | Scaler mean matches training data; test data doesn't change training stats; NaN/Inf handled without test leakage |
| `test_epoching.py` | 8 | Epochs within partition bounds; non-overlapping; correct shape; sfreq mismatch raises |
| `test_features.py` | 12 | PLV in [0,1]; all 41 features returned; missing channels → NaN not crash; deterministic |
| `test_evaluation.py` | 14 | Binary/multiclass metrics in valid range; CI bounds ordered; LOSO subject isolation end-to-end |

All tests use synthetic EEG fixtures and run without downloading the real dataset.

---

## Commands to Reproduce Results

```bash
# Tests only (no data needed)
pytest tests/ -v

# Full pipeline (requires ~8 GB dataset download)
python scripts/build_features.py --config configs/default.yaml
python scripts/run_personalized_evaluation.py --config configs/default.yaml
python scripts/run_unseen_subject_evaluation.py --config configs/default.yaml
python scripts/run_ablation.py --config configs/default.yaml
python scripts/generate_report.py
```

---

## Old Metrics vs Corrected Metrics

| Metric | Old value | Source of bias | Corrected value |
|---|---|---|---|
| Personalized binary accuracy (LightGBM) | 93% mean | Overlapping epochs + scaler/imputer leakage | **89.2% ± 11.1%** |
| Unseen-subject binary accuracy | Not reported | Missing evaluation entirely | **50.1% ± 12.8% (chance level)** |
| Four-class personalized accuracy (LightGBM) | ~85% mean | Overlapping epochs + scaler/imputer leakage | **74.7% ± 11.3%** |

**Interpretation of the change**: The personalized drop from 93% to 89% is the direct cost of the three leakage fixes. The corrected 89% is the scientifically defensible number. More importantly, the new LOSO evaluation reveals that the model generalises essentially at chance (50%) to unseen subjects — a finding that was completely hidden before, and is arguably the most important result in the project.

---

## Remaining Limitations

1. **N=20 subjects** — all results have wide confidence intervals; LOSO metrics have high variance.
2. **Single dataset** — all subjects from the same institution and tradition (Rishikesh). Generalisability to other populations is unknown.
3. **No session-to-session evaluation** — models are trained and tested on the same recording session. Cross-day generalisation is not assessed.
4. **EEG hardware mismatch** — trained on a 64-channel research-grade BioSemi system; performance on consumer headsets (Muse, etc.) would require separate validation.
5. **Correlation, not causation** — feature importance shows predictive association, not that features causally represent meditation depth.
6. **Not a medical device** — model confidence is not a clinical or diagnostic measurement.
