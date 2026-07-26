# Evaluation Audit — BrainDecodingMeditation

**Date**: 2026-07-26  
**Auditor**: Senior ML/EEG Research Engineer review  
**Scope**: `run_pipeline.py`, `per_subject_model.py`, `prove_it.py`, `Source Code.ipynb`, `brain/rich_features.csv`

---

## 1. Summary of Issues Found

| Severity | Issue | File(s) | Status |
|---|---|---|---|
| CRITICAL | Overlapping-epoch leakage in personalized split | `per_subject_model.py` | Fixed |
| CRITICAL | Scaler fitted on full subject dataset before splitting | `per_subject_model.py`, `prove_it.py` | Fixed |
| CRITICAL | Median imputation fitted on full subject dataset before splitting | `per_subject_model.py`, `prove_it.py` | Fixed |
| CRITICAL | No unseen-subject evaluation — only personalized accuracy reported | README, `per_subject_model.py` | Fixed |
| HIGH | Calibration analysis uses random epoch subsampling, not chronological | `prove_it.py` | Fixed |
| HIGH | Cross-subject baseline in `run_pipeline.py` uses subject-level 80/20 split without grouped k-fold | `run_pipeline.py` | Fixed |
| MEDIUM | No reproducibility: no config file, seeds scattered, no artifact saving | All | Fixed |
| MEDIUM | No confidence intervals or bootstrap on reported metrics | All | Fixed |
| MEDIUM | `93% accuracy` headline not clearly labelled as personalized within-subject | README | Fixed |
| LOW | `prove_it.py` fits a new model on all data (no held-out test) for feature importance | `prove_it.py` | Fixed |
| LOW | Correlation analysis (experience vs accuracy) does not account for multiple comparisons | `prove_it.py` | Noted |

---

## 2. Detailed Findings

### 2.1 Overlapping-epoch leakage (CRITICAL)

**Location**: `per_subject_model.py` lines 52–57; `run_pipeline.py` lines 136–141

**Description**:  
Epochs are created with 50% overlap (`overlap=0.5`) using MNE's `make_fixed_length_events`. A 2-second epoch starting at time T and a 2-second epoch starting at time T+1s share 1 second of identical raw EEG signal. When `StratifiedShuffleSplit` is applied to the resulting epoch list, adjacent overlapping epochs are randomly assigned to train and test. This means the test set contains epochs that share raw signal samples with training epochs.

**Effect**: The model has access (through overlapping signal) to information from the test partition during training. Reported accuracy is optimistically inflated.

**Fix**: Split the raw recording timeline chronologically before epoching. Each temporal partition is epoched independently with a safety gap equal to one epoch length at the boundary.

---

### 2.2 Scaler and imputer fitted before split (CRITICAL)

**Location**: `per_subject_model.py` lines 50–56

```python
# ORIGINAL — leaky
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X.values)          # sees all data including test
sss = StratifiedShuffleSplit(...)
tr, te = next(sss.split(X_sc, y))
```

**Description**:  
`StandardScaler.fit_transform` computes mean and standard deviation over the entire subject dataset (train + test combined). The test set therefore influences the normalisation statistics. The same applies to `X.fillna(X.median())` — the median is computed over all rows including test rows.

**Effect**: Leakage of test-set statistics into training preprocessing. Reported accuracy is optimistically biased.

**Fix**: Split indices first. Fit scaler and imputer on `X[train_idx]` only. Transform `X[test_idx]` using the fitted training statistics. Wrap both in a `sklearn.pipeline.Pipeline`.

---

### 2.3 No unseen-subject evaluation (CRITICAL)

**Location**: `per_subject_model.py`, README

**Description**:  
The reported "93% accuracy" is the mean per-subject accuracy when each subject's own earlier epochs are used to train a model that classifies that same subject's later epochs. This is **personalized within-subject accuracy**, not cross-subject or unseen-subject performance.

The README does not clearly state this distinction. A reader could reasonably interpret 93% as the model's ability to generalise to new people, which would be a false claim.

**Effect**: Scientific misrepresentation. The model may perform much worse on a person it has never seen.

**Fix**: Implement a separate Leave-One-Subject-Out evaluation. Report both clearly and distinctly.

---

### 2.4 Calibration analysis uses random subsampling, not chronological (HIGH)

**Location**: `prove_it.py` lines 196–204

```python
# ORIGINAL — leaky calibration
tr_idx = []
for cls in np.unique(y[tr_full]):
    cls_idx = tr_full[y[tr_full] == cls]
    n_cls   = min(n // 2, len(cls_idx))
    tr_idx.extend(np.random.choice(cls_idx, n_cls, replace=False))
```

**Description**:  
The calibration experiment randomly samples `n` epochs from the training pool. Because the epochs are already randomly shuffled by `StratifiedShuffleSplit`, this samples from across the entire recording timeline rather than simulating "the first N seconds of recording." In a real product, you would observe only the first N seconds, not a random N seconds.

**Effect**: The calibration curve is optimistic and not an accurate simulation of real-world deployment.

**Fix**: Use only the earliest `n` epochs (chronologically) from the training partition, keeping the test partition fixed.

---

### 2.5 Cross-subject baseline uses random 80/20 subject split (HIGH)

**Location**: `run_pipeline.py` lines 244–253

**Description**:  
The cross-subject model in `run_pipeline.py` splits 20 subjects into a fixed 80/20 random partition (16 train, 4 test). This gives one estimate with high variance and no confidence interval. There is no nested validation for hyperparameter selection.

**Fix**: Use GroupKFold or LeaveOneGroupOut with Subject as the group, giving 20 per-subject estimates for the unseen-subject protocol.

---

### 2.6 Feature importance fitted on all data (LOW)

**Location**: `prove_it.py` lines 251–261

**Description**:  
The feature importance analysis (`rf_all.fit(Xs_all, y_all)`) trains a Random Forest on the entire dataset with no held-out test set. This is acceptable for an exploratory analysis (the goal is feature ranking, not generalisation accuracy), but it should be explicitly labelled as such.

**Fix**: Label this clearly in code and output as exploratory/descriptive rather than a validated result. Compute feature importance from the training fold of a properly validated model instead.

---

## 3. What Is Correct in the Original Code

- Feature engineering is neuroscience-grounded and well-implemented (PLV, FAA, Hjorth, PermEntropy)
- The idea of per-subject normalisation is correct in principle
- Band definitions and channel region mappings are correct
- MNE preprocessing chain (filter → notch → resample → average reference) is standard and correct
- The insight that personalized models are needed for EEG BCI is scientifically valid

---

## 4. Changes Implemented

All issues above are fixed in the refactored codebase under `src/`. See `CHANGELOG_FIXES.md` for a complete list of changes. Key changes:

1. **`src/splitting.py`**: Chronological time-block splitting of raw recordings before epoching; safety gap enforcement; overlap leakage detection
2. **`src/preprocessing.py`**: `sklearn.Pipeline` enforcing train-only fit for imputer and scaler
3. **`src/evaluation.py`**: Two protocols (personalized + LOSO), full metric suite, bootstrap CIs
4. **`src/epoching.py`**: Epoch generation per partition, metadata tracking
5. **`scripts/run_personalized_evaluation.py`**: Clean per-subject evaluation
6. **`scripts/run_unseen_subject_evaluation.py`**: LOSO evaluation
7. **`configs/default.yaml`**: All hyperparameters centralised
8. **`tests/`**: 12 tests with synthetic EEG fixtures, all passing without requiring real data

---

## 5. Metrics That Require a Full Rerun

The following metrics require downloading the full EEG dataset (~8 GB from OpenNeuro ds003969) and running the complete pipeline:

- Corrected personalized binary accuracy (previously reported as 93% from a leaky split)
- Corrected unseen-subject (LOSO) binary accuracy (not previously reported)
- Calibration curve under chronological splitting
- Feature ablation table

All result files in `results/` are marked `PENDING_RERUN` until the full pipeline completes.
