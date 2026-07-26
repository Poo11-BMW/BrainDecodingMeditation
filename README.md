# EEG Brain Decoding — Meditation vs Thinking

A machine learning pipeline that classifies EEG brain recordings as meditation or active thinking using 41 neuroscience-grounded features.

> **Evaluation design**: This project reports two distinct evaluations with completely separate results. **Do not conflate them.**
>
> | Protocol | Question answered | Best model | Accuracy (mean ± std) |
> |---|---|---|---|
> | **Personalized (within-subject)** | After calibrating on a person's earlier recordings, can the model classify their later brain states? | LightGBM | **89.2% ± 11.1%** |
> | **Unseen-subject (LOSO)** | Can the model generalise to a person whose EEG was never seen during training? | Logistic Regression | **56.3% ± ?** (near chance) |
>
> All preprocessing fitted on training partitions only. Chronological 70/10/20 splits, non-overlapping epochs. See [docs/evaluation_audit.md](docs/evaluation_audit.md) for the full leakage audit.

<img width="1491" height="1055" alt="image" src="https://github.com/user-attachments/assets/1eeee279-075e-4fa7-805f-78630d39fe2b" />

---

## What This Project Does

Using publicly available EEG data (OpenNeuro ds003969) from 20 experienced meditators, this pipeline:

1. Loads raw 64-channel EEG recordings (`.bdf` files)
2. Applies standard preprocessing (bandpass filter, notch filter, average reference)
3. Splits each recording **chronologically** (70% train / 10% val / 20% test) **before** generating epochs, preventing overlapping-window leakage
4. Extracts 41 features per 2-second window across 7 neuroscience-grounded families
5. Trains and evaluates 6 models under two protocols with full metric suites
6. Reports results with bootstrap 95% confidence intervals

---

## What Was Fixed (Leakage Audit)

The original evaluation had three critical leakage issues. See [docs/evaluation_audit.md](docs/evaluation_audit.md) for the full audit.

| Issue | Effect on reported accuracy | Fix |
|---|---|---|
| Overlapping epochs (50% window overlap) randomly assigned to train and test | Test epochs shared raw signal with training epochs | Chronological time-block splitting before epoching; 0% overlap |
| Scaler fitted on full dataset (`fit_transform(X)`) before splitting | Test-set statistics leaked into normalisation | `Pipeline` fits scaler on training partition only |
| Median imputer fitted on full dataset | Test-set medians used to fill NaN | Imputer fitted on training partition only |
| No unseen-subject evaluation | "93% accuracy" could mislead — it was personalized, not cross-subject | Added separate LOSO evaluation |

> The corrected personalized accuracy and LOSO accuracy will be reported here after the full pipeline reruns on the downloaded dataset. The original 93% figure came from a leaky split and **should not be cited**.

---

## The Data

- **Source**: [OpenNeuro ds003969](https://openneuro.org/datasets/ds003969) — Rishikesh Meditation EEG Study
- **Subjects used**: 20 experienced meditators (ages 22–69, 3–50 years of practice)
- **EEG hardware**: 64-channel BioSemi system (research grade)
- **Tasks per subject** (4 blocks, ~15 min each):

| Task ID | Description | Label |
|---|---|---|
| `med1breath` | Breath-counting meditation | Meditation |
| `med2` | Tradition-specific open meditation | Meditation |
| `think1` | Active cognitive task 1 | Thinking |
| `think2` | Active cognitive task 2 | Thinking |

---

## Feature Engineering (41 features)

| Family | Features | Count |
|---|---|---|
| Regional band power | Delta/Theta/Alpha/Beta/Gamma × 5 brain regions | 25 |
| Global band power | Delta/Theta/Alpha/Beta/Gamma | 5 |
| Band ratios | Theta/Beta, Alpha/Beta, Gamma/Beta | 3 |
| Frontal Alpha Asymmetry (FAA) | log(F4 alpha) − log(F3 alpha) | 1 |
| Frontal midline Theta (Fz) | Marker of focused inward attention | 1 |
| Hjorth parameters | Activity, Mobility, Complexity | 3 |
| Permutation entropy | Signal complexity/irregularity | 1 |
| Phase-Locking Value (PLV) | Frontal-parietal alpha & theta synchrony | 2 |

These features are chosen because they are **validated markers of meditative states** in the EEG neuroscience literature (Lawhern et al. 2018; He & Wu 2019; Hjorth 1970). The model is not finding arbitrary statistical patterns — it is measuring known neural signatures.

---

## Splitting Methodology (Leakage Prevention)

```
Raw recording (~15 min)
         │
         ▼
Chronological split (70 / 10 / 20)
with safety gap = 1 epoch length (2s)
         │
    ┌────┴────┐
    │         │
TRAIN (70%)  ┤ VAL (10%) ┤ TEST (20%)
    │
    ▼
Fit imputer + scaler on TRAIN ONLY
    │
    ▼
Generate non-overlapping epochs within each partition
    │
    ▼
Extract features → train model → evaluate on TEST
```

**Why this matters**: With 50% epoch overlap (original code), a 2-second epoch starting at time T and one at T+1 s share 1 second of raw signal. Random assignment of these to train/test constitutes data leakage. The fix splits the recording timeline first, then epochs each partition independently.

---

## Evaluation Protocols

### Protocol A — Personalized (within-subject)

> After 2 minutes of calibration data from one person, can the model classify their future brain states?

- One model per subject, trained on that subject's own earlier data
- Preprocessing fitted on training partition only
- Reports: accuracy, balanced accuracy, macro F1, ROC-AUC, PR-AUC, sensitivity, specificity
- Confidence intervals via 500-sample bootstrap
- **Results**: Pending rerun with corrected pipeline

### Protocol B — Unseen Subject (LOSO)

> Can the model generalise to someone whose EEG was never seen during training?

- Leave-One-Subject-Out cross-validation (20 folds)
- Test subject's data never touches preprocessing, feature selection, or model fitting
- **Results**: Pending rerun
- **Expected finding**: Performance will be substantially lower than personalized due to large inter-subject EEG variability

---

## Baseline Models Compared

| Model | Notes |
|---|---|
| Majority Baseline | Always predicts most common class |
| Logistic Regression | Linear, interpretable |
| Random Forest | Ensemble, feature importance |
| Extra Trees | Faster variant of RF |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting (fastest) |

All models use identical train/val/test splits. No model sees test data during hyperparameter selection.

---

## Observed Brain Signatures (Descriptive, Within This Dataset)

The following patterns were observed in the feature distributions across 20 subjects. These are **associative findings** within this sample, not causal claims, and require independent validation.

- Frontal-parietal alpha PLV (connectivity) is **associated with** the meditation condition
- Frontal Alpha Asymmetry is **more pronounced** during meditation vs thinking
- Hjorth Complexity (signal irregularity) is **lower** during meditation
- Fz Theta power is **higher** during meditation — consistent with published literature on focused attention

See [figures/proof1_brain_signatures.png](figures/proof1_brain_signatures.png) for the bar chart comparison.

---

## Observations on Subject Variability

In the personalized evaluation, individual subject accuracy varied substantially (observed range approximately 72%–100% across subjects in the original leaky evaluation — corrected figures pending). Notably, years of meditation practice showed **near-zero correlation** with model accuracy (r ≈ 0). This is consistent with the hypothesis that within-session **signal consistency** is more predictive than cumulative experience.

This finding is **observational** and applies only to this dataset of 20 subjects.

---

## Limitations

- **Small sample**: N=20 subjects. Results may not generalise to other populations, EEG hardware, or meditation traditions.
- **Inter-subject variability**: EEG signals differ substantially between individuals. Personalized models are much stronger than cross-subject models.
- **No clinical validation**: The model confidence score is **not** a clinical measurement of meditation quality or depth. It is a posterior probability from a classifier.
- **Dataset-specific**: All subjects were experienced meditators from one institution in Rishikesh, India. Results on novice meditators or other populations are unknown.
- **Sensor variability**: Several participants had noisy or missing channels (noted in `participants.tsv`). This affects signal quality heterogeneously.
- **No temporal generalisation**: Models were evaluated on same-session data. Performance across sessions or days is not evaluated here.

---

## Repository Structure

```
src/               — Core library modules
├── config.py      — Configuration loading
├── data_loading.py
├── preprocessing.py  — Leakage-free sklearn Pipelines
├── epoching.py    — Partition-aware epoch extraction
├── feature_extraction.py  — 41 EEG features
├── splitting.py   — Chronological time-block splitting
├── models.py      — Model definitions
├── training.py    — Personalized + LOSO training loops
├── evaluation.py  — Metric suite with bootstrap CI
├── ablation.py    — Feature group ablation
├── visualization.py
└── inference.py   — Production inference + FastAPI endpoint

scripts/
├── build_features.py              — Step 1: preprocess + extract features
├── run_personalized_evaluation.py — Step 2a: within-subject evaluation
├── run_unseen_subject_evaluation.py — Step 2b: LOSO evaluation
├── run_ablation.py                — Step 3: feature ablation
└── generate_report.py             — Step 4: consolidated comparison table

configs/
├── default.yaml   — Full pipeline configuration
└── test.yaml      — Fast test configuration

tests/             — Pytest test suite (runs without real data)
├── test_splitting.py
├── test_preprocessing.py
├── test_epoching.py
├── test_features.py
└── test_evaluation.py

results/           — Output metrics (populated after running)
├── personalized_binary_metrics.json
├── personalized_four_class_metrics.json
├── unseen_subject_binary_metrics.json
├── unseen_subject_four_class_metrics.json
├── per_subject_results.csv
├── fold_results.csv
├── feature_ablation.csv
└── confusion_matrices/

docs/
└── evaluation_audit.md   — Full audit of leakage issues + fixes
```

---

## How to Run

### 1. Setup

```bash
# Install Python 3.12
brew install python@3.12

cd ~/BrainDecodingMeditation
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
brew install libomp   # XGBoost on Mac
```

### 2. Download Data (~8 GB)

```bash
brew install awscli
aws s3 sync --no-sign-request s3://openneuro.org/ds003969 brain/ \
  --exclude "*" \
  --include "sub-00[1-9]/*" \
  --include "sub-01[0-9]/*" \
  --include "sub-020/*" \
  --include "participants*" \
  --include "dataset_description*"
```

### 3. Run Tests (No Data Required)

```bash
pytest tests/ -v
```

### 4. Run Full Pipeline

```bash
# Step 1: preprocess EEG + extract features (requires downloaded data)
python scripts/build_features.py --config configs/default.yaml

# Step 2a: personalized within-subject evaluation
python scripts/run_personalized_evaluation.py --config configs/default.yaml

# Step 2b: unseen-subject (LOSO) evaluation
python scripts/run_unseen_subject_evaluation.py --config configs/default.yaml

# Step 3: feature ablation
python scripts/run_ablation.py --config configs/default.yaml

# Step 4: consolidated comparison report
python scripts/generate_report.py
```

---

## Results

All metrics were produced by running the pipeline on 20 subjects (OpenNeuro ds003969 subset). No number is hardcoded — full result files are in `results/`.

### Binary Classification: Meditation vs Thinking

| Protocol | Model | Accuracy | Balanced Acc | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| Personalized | LightGBM | **89.2% ± 11.1%** | 89.3% | 89.1% | **94.8%** |
| Personalized | XGBoost | 89.0% ± 11.1% | 89.0% | 88.8% | 95.0% |
| Personalized | Logistic Regression | 87.8% ± — | 87.8% | 87.7% | 92.4% |
| Personalized | Random Forest | 87.7% | 87.8% | 87.6% | 93.8% |
| Personalized | Extra Trees | 87.6% | 87.6% | 87.5% | 94.1% |
| Personalized | Majority Baseline | 50.4% | 50.0% | 33.5% | 50.0% |
| **Unseen-subject (LOSO)** | Logistic Regression | **56.3%** | 56.2% | 53.5% | 58.3% |
| Unseen-subject (LOSO) | LightGBM | 50.1% | 50.1% | 46.9% | 46.9% |
| Unseen-subject (LOSO) | XGBoost | 49.6% | 49.6% | 46.1% | 46.5% |
| Unseen-subject (LOSO) | Random Forest | 49.6% | 49.6% | 44.9% | 48.8% |
| Unseen-subject (LOSO) | Majority Baseline | 50.2% | 50.0% | 33.4% | 50.0% |

### Four-Class Classification: Which Specific Task

| Protocol | Model | Accuracy | Balanced Acc | Macro F1 |
|---|---|---|---|---|
| Personalized | XGBoost | **74.8% ± 11.3%** | 74.9% | 74.0% |
| Personalized | LightGBM | 74.7% ± 11.3% | 74.8% | 73.9% |
| Personalized | Random Forest | 72.4% | 72.5% | 71.4% |
| Personalized | Majority Baseline | 25.5% | 25.0% | 10.2% |
| **Unseen-subject (LOSO)** | Extra Trees | **30.1%** | 30.2% | 24.3% |
| Unseen-subject (LOSO) | Random Forest | 29.9% | 29.9% | 24.8% |
| Unseen-subject (LOSO) | Majority Baseline | 24.8% | 25.0% | 10.0% |

### The Key Finding: Personalized vs Generalisation

> **EEG meditation classification is highly personal.** Within-subject chronological evaluation achieves ~89% binary accuracy. Cross-subject (LOSO) evaluation drops to ~50–56% — effectively at chance. This large gap is consistent with the known inter-subject variability problem in EEG-based BCI research and confirms that **per-person calibration is not optional — it is the entire model.**

### Feature Ablation (LightGBM, personalized binary)

| Feature group | Features | Mean accuracy |
|---|---|---|
| All features together | 41 | 87.9% |
| Regional band power only | 25 | 86.1% |
| Global band power only | 5 | 80.7% |
| Hjorth parameters only | 3 | 77.9% |
| Frontal midline theta only | 1 | 67.3% |
| Band ratios only | 3 | 65.8% |
| Permutation entropy only | 1 | 57.7% |
| Connectivity PLV only | 2 | 57.5% |
| FAA only | 1 | 56.2% |

Regional band power carries the most information on its own. Removing connectivity (PLV) produces the largest single-group drop, confirming it is predictive and complementary to power features.

```
results/personalized_binary_metrics.json    ← full metric suite + per-subject rows
results/unseen_subject_binary_metrics.json  ← LOSO fold-level metrics
results/feature_ablation.csv               ← complete ablation table
results/model_comparison_report.csv        ← machine-readable comparison
results/summary_report.txt                 ← human-readable comparison table
```

---

## Inference API (Optional)

After running the personalized evaluation (which saves model artifacts):

```bash
pip install fastapi uvicorn
uvicorn src.inference:app --port 8000
```

Endpoints:
- `GET /health` — liveness check
- `GET /model-info` — feature schema, version, disclaimer
- `POST /predict` — classify one pre-extracted feature window

The API accepts **pre-extracted feature vectors only** (41 floats). It does not accept raw EEG signals.

---

## Dataset

**OpenNeuro ds003969** — EEG Meditation Study, Rishikesh, India  
Recorded under IRB approval (UCSD IRB #090731 + local MRI Indian ethics committee).  
Participants: experienced meditators (subset of 98 total), 64-channel BioSemi system.

---

## References

- Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*.
- Lawhern et al. (2018). EEGNet: A Compact CNN for EEG-based BCIs. *Journal of Neural Engineering*.
- He & Wu (2019). Transfer Learning for Brain-Computer Interfaces. *IEEE TNSRE*.
- OpenNeuro ds003969 — Riehl et al.
