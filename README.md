# 🧠 EEG Brain Decoding: Classifying Mental States Using Machine Learning

This project explores how machine learning can be used to decode and classify different mental states — such as **meditation vs cognitive thinking** — using EEG (electroencephalography) data. EEG captures the electrical activity of the brain, and by carefully preprocessing and analyzing this data, we aim to distinguish between brain states with high accuracy.

---

## 📦 Dataset

The dataset consists of EEG recordings (real-time data) from **40 subjects**, each performing four distinct tasks:

- 🧘‍♂️ `med1breath` – Breath-focused meditation  
- 🧘 `med2` – Passive meditation  
- 🤔 `think1` – Logical thinking  
- 💭 `think2` – Mental visualization

The raw EEG data is stored in `.bdf` format and undergoes several preprocessing steps.

---

## 🔧 Preprocessing Pipeline

1. **Noise Removal**
   - Bandpass filter: 0.5–40 Hz
   - Notch filter: 50 Hz (to remove electrical interference)
   - Multiprocessing for efficient parallel execution

2. **Artifact Correction**
   - Independent Component Analysis (ICA) removes **eye-blink artifacts**
   - Uses automatically detected EOG (eye movement) channels

3. **Segmentation & Normalization**
   - Signals are normalized and **segmented into 2-second overlapping epochs**

---

## 📊 Feature Extraction

We compute **Power Spectral Density (PSD)** features for each EEG epoch to measure the brain’s activity across common frequency bands:

| Band   | Frequency Range | Associated State         |
|--------|------------------|--------------------------|
| Delta  | 0.5–4 Hz         | Deep sleep, unconscious  |
| Theta  | 4–8 Hz           | Meditation, drowsiness   |
| Alpha  | 8–12 Hz          | Calm, relaxed awareness  |
| Beta   | 12–30 Hz         | Focus, active thinking   |
| Gamma  | 30–50 Hz         | High-level cognition     |

We also engineered **Theta/Beta** and **Alpha/Beta** ratios — known indicators in cognitive and mental health research.

---

## 🤖 Machine Learning Models & Performance

Several machine learning models were trained using the extracted features:

| Model                 | Accuracy (%) |
|----------------------|--------------|
| 🏆 Random Forest      | **86.49%**   |
| 🚀 XGBoost            | 85.81%       |
| 🌱 Gradient Boosting  | 83.33%       |
| 🧠 SVM (RBF Kernel)   | 82.43%       |
| 📉 KNN                | 37.84%       |
| 📉 Logistic Regression| 25.68%       |

The **Random Forest model performed the best**, clearly indicating that nonlinear ensemble models are well-suited for EEG-based classification tasks. Simpler models like KNN and Logistic Regression failed to generalize effectively, likely due to the complexity and non-linearity of EEG signals.

---

## 💡 Project Impact

This end-to-end pipeline — from raw EEG to fully trained machine learning models — showcases how neuroscience and artificial intelligence can work together to decode the mind. It opens up pathways for:

- 🧠 Real-time **brain-computer interfaces**
- 📈 Personalized **neurofeedback systems**
- 🧘‍♀️ Monitoring **meditative states** in wellness applications
- 🧪 Cognitive state analysis in clinical research

With further enhancement using deep learning, raw EEG sequence modeling, or multimodal input (e.g., heart rate, eye tracking), this system could evolve into a robust cognitive decoding platform.

---

