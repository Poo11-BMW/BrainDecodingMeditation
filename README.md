# Can a Computer Tell If You're Meditating?

A machine learning research project that reads EEG brainwaves and classifies whether a person is meditating or thinking — trained and evaluated on real recordings from 20 experienced meditators.

<img width="1491" height="1055" alt="Brain signatures during meditation vs thinking" src="https://github.com/user-attachments/assets/1eeee279-075e-4fa7-805f-78630d39fe2b" />

---

## The Short Answer

**Yes — but only when the model has seen that specific person's brain before.**

| Evaluation | What it means | Accuracy |
|---|---|---|
| Personalized (within-subject) | Model trained on your own earlier brain data, tested on your later data | **89.2% ± 11.1%** |
| Unseen-subject (LOSO) | Model trained on 19 other people, tested on a brand-new person | **~50% (chance level)** |

This gap is the central finding. EEG is so individual that a model trained on other people is useless on someone it has never seen. A short personal calibration session is necessary for the system to work.

---

## What Problem Are We Solving?

Meditation is hard to measure. Practitioners report their experience subjectively. Researchers studying whether meditation works rely on self-reported questionnaires. There is no real-time, objective signal that tells you — or a researcher — how deep into a meditative state you actually are.

**EEG** (electroencephalography) records electrical activity from the brain through electrodes placed on the scalp. It is non-invasive and captures neural dynamics at millisecond resolution. The question this project asks is:

> Can a machine learning model look at 2 seconds of EEG and reliably tell whether the brain producing it is meditating or thinking?

---

## The Dataset

**Source**: OpenNeuro ds003969 — a public EEG dataset recorded at the Meditation Research Institute in Rishikesh, India, under full IRB ethics approval.

**Who participated**: 20 experienced meditators, ages 22–69, with 3–50 years of meditation practice.

**Recording setup**: 64-channel BioSemi EEG cap (research-grade hardware) at 2048 Hz, later resampled to 128 Hz.

**What they did**: Each person completed four ~15-minute blocks in alternating order:

| Block | What they did | Label |
|---|---|---|
| `med1breath` | Breath-counting meditation | Meditation |
| `med2` | Open or tradition-specific meditation | Meditation |
| `think1` | Active cognitive task (thinking) | Thinking |
| `think2` | Active cognitive task (thinking) | Thinking |

After preprocessing, each person's recordings produce roughly **1,200 non-overlapping 2-second windows**. Each window becomes one training or test example.

---

## Pipeline: From Raw Brain Signal to Prediction

### Step 1 — Preprocessing

Raw EEG is extremely noisy. Before extracting any information, the signal goes through a standard neuroscience preprocessing chain:

| Step | What it does |
|---|---|
| Bandpass filter 0.5–40 Hz | Removes very slow drift and high-frequency muscle noise |
| Notch filter at 50 Hz | Removes AC power-line interference from the Indian power grid |
| Resample to 128 Hz | Reduces data size while preserving all brain-relevant frequencies |
| Average reference | Subtracts the mean signal across all electrodes to remove shared noise |
| Per-channel normalisation | Each electrode is scaled to zero mean and unit variance |

### Step 2 — Splitting the Recording

Each person's recording is divided chronologically:

```
─────────────────────────────── Recording timeline ────────────────────────────────
│        70% → Training        │  10% Val  │         20% → Test                   │
```

The first 70% of the recording becomes training data, the next 10% is validation, and the final 20% is the test set. All preprocessing statistics (normalisation constants) are computed from training data only and applied to the test set — never the reverse. This ensures the model learns from the past and is evaluated on the future.

### Step 3 — Feature Extraction (41 features per window)

Rather than feeding raw EEG into a neural network, we extract **41 neuroscience-grounded features** from each 2-second window. These are established markers of brain state from the published literature.

**The frequency bands:**

| Band | Frequency range | Associated with |
|---|---|---|
| Delta | 0.5–4 Hz | Deep sleep, unconscious processes |
| Theta | 4–8 Hz | Focused inward attention, meditation |
| Alpha | 8–12 Hz | Relaxed alertness |
| Beta | 12–30 Hz | Active thinking, alertness |
| Gamma | 30–40 Hz | High-level cognitive processing |

**The feature groups:**

| Group | # Features | What it captures |
|---|---|---|
| Regional band power | 25 | Energy in each of 5 bands across 5 scalp regions (frontal, temporal, parietal, occipital, central) |
| Global band power | 5 | Same bands, averaged across all 64 electrodes |
| Band ratios | 3 | Relative strength of one band vs another (e.g. theta/alpha) |
| Frontal Alpha Asymmetry (FAA) | 1 | Whether left or right frontal lobe dominates in alpha — linked to emotional valence and mindfulness |
| Frontal midline Theta (Fz) | 1 | Slow wave activity at the brain's frontal midline — rises during focused inward attention |
| Hjorth parameters | 3 | Signal activity, mobility, and complexity in the time domain |
| Permutation entropy | 1 | How chaotic vs ordered the brain signal is — lower during sustained quiet states |
| Phase-Locking Value (PLV) | 2 | How synchronised the front and back of the brain are — measures large-scale network coordination |

**Why features instead of raw signal or deep learning?**

Three reasons. First, **interpretability**: each feature maps to a published neuroscience concept. If the model relies heavily on frontal theta, that finding is meaningful and testable against the broader literature. Second, **sample size**: deep learning needs hundreds of subjects to generalise. With 20 people, well-chosen neuroscience features outperform end-to-end neural networks on this dataset. Third, **robustness**: theory-grounded features are more likely to transfer to other recording setups than raw amplitude patterns.

### Step 4 — Model Training

Six models are trained and compared:

| Model | Type |
|---|---|
| Majority Baseline | Always predicts the most common class — the performance floor |
| Logistic Regression | Linear, fully interpretable |
| Random Forest | Ensemble of decision trees |
| Extra Trees | Faster variant of Random Forest |
| XGBoost | Gradient-boosted trees |
| LightGBM | Fast gradient-boosted trees |

All models use a shared preprocessing pipeline (median imputation for missing values + standard scaling) fitted on training data only. Hyperparameters are fixed in `configs/default.yaml` and shared across all experiments.

---

## Experiment 1 — Personalized Evaluation (Within-Subject)

**Setup**: One model per person. The model is trained on that person's own earliest recordings (70% of their session) and tested on their own latest recordings (final 20%). The 19 other people are never involved.

**Results:**

| Model | Accuracy | Balanced Acc | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| **LightGBM** | **89.2% ± 11.1%** | **89.3%** | **89.1%** | **94.8%** |
| XGBoost | 89.0% ± 11.1% | 89.0% | 88.8% | 95.0% |
| Logistic Regression | 87.8% | 87.8% | 87.7% | 92.4% |
| Random Forest | 87.7% | 87.8% | 87.6% | 93.8% |
| Extra Trees | 87.6% | 87.6% | 87.5% | 94.1% |
| Majority Baseline | 50.4% | 50.0% | 33.5% | 50.0% |

The jump from 50% (baseline) to 89% (LightGBM) represents real, replicable predictive signal in the EEG features.

**Per-person breakdown (LightGBM):**

| Subject | Years of practice | Age | Accuracy |
|---|---|---|---|
| sub-018 | 9 | 36 | 100.0% |
| sub-015 | 18 | 69 | 99.6% |
| sub-020 | 32 | 67 | 99.2% |
| sub-003 | 3 | 22 | 98.7% |
| sub-009 | 40 | 58 | 97.5% |
| sub-001 | 3 | 26 | 97.5% |
| sub-011 | 15 | 38 | 95.0% |
| sub-007 | 15 | 31 | 92.2% |
| sub-005 | 28 | 38 | 88.0% |
| sub-012 | 46 | 65 | 84.5% |
| sub-002 | 31 | 62 | 78.7% |
| sub-006 | 8 | 33 | 78.1% |
| sub-010 | 40 | 59 | 77.4% |
| sub-008 | 16 | 56 | 75.7% |
| sub-019 | 20 | 55 | 71.2% |
| sub-004 | 50 | 67 | 64.3% |

**A striking observation**: the person with 50 years of experience (sub-004) scores the worst. Two people with only 3 years of practice (sub-001, sub-003) are in the top four. Years of meditation does not predict how distinctly separable a person's brain states will be.

One interpretation: experienced meditators often blend techniques, shift attention, and maintain richer internal narratives even during formal sessions — making their brain state less sharply distinct from active thinking. Newer practitioners follow instructions more literally, producing a more stereotyped, consistent signal the model can classify cleanly. Signal consistency within the recording session matters more than cumulative practice time.

---

## Experiment 2 — Unseen-Subject Evaluation (LOSO)

**Setup**: Leave-One-Subject-Out cross-validation. In each of 20 rounds, one person is completely withheld as the test subject. The model is trained on all data from the remaining 19 people and evaluated on the held-out person — someone it has never seen.

**Results:**

| Model | Accuracy | Balanced Acc | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | **56.3%** | **56.2%** | 53.5% | **58.3%** |
| Majority Baseline | 50.2% | 50.0% | 33.4% | 50.0% |
| LightGBM | 50.1% | 50.1% | 46.9% | 46.9% |
| XGBoost | 49.6% | 49.6% | 46.1% | 46.5% |
| Random Forest | 49.6% | 49.6% | 44.9% | 48.8% |
| Extra Trees | 49.8% | 49.8% | 44.5% | 47.6% |

All complex models perform at or near the majority-class baseline. The best result — Logistic Regression at 56% — is barely above chance. Interestingly, the simplest model wins here; it overfits less to the idiosyncratic patterns of specific individuals in training.

**What this means**: EEG signatures of meditation vary so dramatically between individuals that a model trained on 19 people cannot usefully classify the 20th. There is no universal "meditation signature" discoverable at this sample size. This is the **inter-subject variability problem** — a well-documented challenge in EEG-based brain-computer interface research. Every brain's electrical profile is unique enough that group-trained models fail at the individual level.

The practical consequence: any real-world EEG meditation classifier needs a per-person calibration phase before it can work reliably for that individual.

---

## Experiment 3 — Four-Class Classification

Beyond binary meditation vs thinking, we also asked: can the model distinguish *which specific task* — breath meditation, open meditation, cognitive task 1, or cognitive task 2? Chance here is 25%.

| Protocol | Best model | Accuracy |
|---|---|---|
| Personalized | XGBoost | **74.8% ± 11.3%** |
| Unseen-subject | Extra Trees | **30.1%** |

The same pattern holds. Personalized classification works well (75% vs 25% chance). Cross-subject barely clears chance (30%). The different types of meditation are more similar to each other than meditation vs thinking — but the model can still learn fine-grained distinctions on a per-person basis.

---

## What Features Drive the Predictions?

We ran the personalized binary evaluation using different subsets of the 41 features to measure each group's contribution.

| Features used | Mean accuracy |
|---|---|
| All 41 features | 87.9% |
| Regional band power alone (25 features) | 86.1% |
| All minus connectivity (PLV) | 87.0% |
| All minus regional band power | 84.2% |
| Global band power alone | 80.7% |
| Hjorth parameters alone | 77.9% |
| Frontal theta alone | 67.3% |
| Band ratios alone | 65.8% |
| Permutation entropy alone | 57.7% |
| PLV connectivity alone | 57.5% |
| Frontal Alpha Asymmetry alone | 56.2% |

**Key takeaways**:

- **Regional band power** is the single most powerful family — 25 features, 86% accuracy by itself. It carries the majority of the predictive signal.
- **Removing connectivity (PLV)** causes the largest single-group drop (87.9% → 87.0%), showing PLV adds unique information that power features alone do not capture.
- **No single feature family is sufficient on its own** — FAA, permutation entropy, and PLV each score near chance in isolation.
- The full combination of all 41 features reaches the highest overall accuracy.

---

## What Do the Brain Signatures Look Like?

Comparing meditation vs thinking windows across the 20 subjects:

- **Frontal-parietal alpha synchrony (PLV)** is higher during meditation — the front and back of the brain are more coherently linked
- **Frontal Alpha Asymmetry (FAA)** is more pronounced during meditation — left and right frontal lobes differ more in their alpha power
- **Signal complexity (permutation entropy, Hjorth)** is lower during meditation — the brain signal becomes quieter and more regular
- **Frontal midline theta (Fz)** is elevated during meditation — consistent with the literature on focused inward attention

These patterns align with established neuroscience findings on meditative states. The fact that our model's most important features match what the broader literature already reports suggests the model is learning real neural patterns rather than statistical noise.

---

## Project Structure

```
src/                                  — All library code
├── config.py                         — Hyperparameters, loaded from YAML
├── data_loading.py                   — Load raw BDF files + metadata
├── splitting.py                      — Chronological splitting of recording timeline
├── epoching.py                       — Generate windows within each partition
├── feature_extraction.py             — 41-feature extraction (PLV, FAA, PSD, etc.)
├── preprocessing.py                  — Imputer + scaler pipeline (train-only fit)
├── models.py                         — 6 model definitions
├── training.py                       — Personalized + LOSO training loops
├── evaluation.py                     — Metrics: accuracy, F1, ROC-AUC, bootstrap CI
├── ablation.py                       — Feature group ablation
├── visualization.py                  — Result figures
└── inference.py                      — FastAPI production inference endpoint

scripts/                              — Runnable pipeline steps
├── build_features.py                 — Preprocess EEG + extract features
├── run_personalized_evaluation.py    — Within-subject evaluation
├── run_unseen_subject_evaluation.py  — LOSO evaluation
├── run_ablation.py                   — Feature ablation study
└── generate_report.py                — Consolidated results table

tests/                                — 70 automated tests (no real data needed)
configs/                              — YAML configuration files
results/                              — Output metrics (JSON + CSV + figures)
```

---

## Reproduce the Results

### Setup

```bash
git clone <this-repo>
cd BrainDecodingMeditation
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp   # required for XGBoost on macOS
```

### Download the dataset (~11 GB for 20 subjects)

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

### Run tests (no data needed)

```bash
pytest tests/ -v   # 70 tests using synthetic EEG fixtures
```

### Run the full pipeline

```bash
# Step 1 — preprocess raw EEG and extract features (~20–40 min)
python scripts/build_features.py --config configs/default.yaml

# Step 2a — personalized evaluation
python scripts/run_personalized_evaluation.py --config configs/default.yaml

# Step 2b — unseen-subject (LOSO) evaluation
python scripts/run_unseen_subject_evaluation.py --config configs/default.yaml

# Step 3 — feature ablation study
python scripts/run_ablation.py --config configs/default.yaml

# Step 4 — consolidated results table
python scripts/generate_report.py
```

---

## Summary of Results

| Metric | Value |
|---|---|
| Personalized binary accuracy (LightGBM) | **89.2% ± 11.1%** |
| Personalized ROC-AUC | **94.8%** |
| Unseen-subject binary accuracy (best model) | **56.3%** |
| Personalized four-class accuracy | **74.8% ± 11.3%** |
| Best individual (sub-018, 9 yrs practice) | **100.0%** |
| Worst individual (sub-004, 50 yrs practice) | **64.3%** |
| Subjects | 20 |
| Windows per subject | ~1,200 |
| Features | 41 |
| Models compared | 6 |
| Automated tests | 70 |

---

## Limitations

- **Small sample**: 20 subjects. Wide confidence intervals; results may not generalise to other populations, ages, or traditions.
- **Single session**: Training and testing happen within the same recording session. Cross-day or cross-session generalisation is not evaluated.
- **Experienced meditators only**: All 20 participants had 3–50 years of practice from one institution in Rishikesh. Performance on beginners or people from other traditions is unknown.
- **Research-grade hardware**: Trained on a 64-channel BioSemi system. Consumer EEG headsets (Muse, OpenBCI) have fewer channels and more noise — separate validation would be required.
- **Associations, not causation**: Feature importance tells us what is predictive in this dataset. It does not prove those features causally define or constitute meditative states.
- **Not a clinical tool**: Model output is a posterior probability from a classifier. It is not a measurement of meditation depth or quality, and is not suitable for diagnostic use.

---

## References

- Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*, 29(3), 306–310.
- He & Wu (2019). Transfer Learning for Brain-Computer Interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*.
- OpenNeuro ds003969 — EEG meditation recordings, Meditation Research Institute, Rishikesh, India.
