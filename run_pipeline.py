"""
EEG Brain Decoding Pipeline — Rich Features
Features: regional PSD, frontal alpha asymmetry, band ratios,
          Hjorth parameters, permutation entropy, frontal-parietal PLV
"""

import os, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")
from scipy.signal import hilbert, butter, filtfilt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), "brain")
TASKS     = ["med1breath", "med2", "think1", "think2"]
BANDS     = {"Delta":(0.5,4), "Theta":(4,8), "Alpha":(8,12), "Beta":(12,30), "Gamma":(30,40)}
SFREQ     = 128
EPOCH_LEN = 2.0

# Brain regions — code will match against actual channel names
REGION_DEFS = {
    "frontal":   ["Fp1","Fp2","AF3","AF4","AF7","AF8","F7","F3","Fz","F4","F8"],
    "central":   ["FC5","FC1","FCz","FC2","FC6","C3","Cz","C4"],
    "temporal":  ["T7","T8","TP7","TP8"],
    "parietal":  ["CP5","CP1","CPz","CP2","CP6","P7","P3","Pz","P4","P8"],
    "occipital": ["PO7","PO3","POz","PO4","PO8","O1","Oz","O2"],
}

# ─────────────────────────────────────────────
# FEATURE HELPERS
# ─────────────────────────────────────────────

def bandpass(data, lo, hi, fs=SFREQ):
    """Bandpass filter (n_channels, n_times) data."""
    nyq = fs / 2.0
    lo  = max(lo, 0.1)
    hi  = min(hi, nyq - 0.1)
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)

def compute_plv(sig_a, sig_b):
    """PLV between two (n_ch_a, T) and (n_ch_b, T) arrays.
    Returns scalar PLV between mean phases of each group."""
    h_a   = hilbert(sig_a.mean(axis=0))
    h_b   = hilbert(sig_b.mean(axis=0))
    phase = np.angle(h_a) - np.angle(h_b)
    return float(np.abs(np.mean(np.exp(1j * phase))))

def hjorth(data):
    """Hjorth Activity, Mobility, Complexity — mean over channels."""
    dx   = np.diff(data, axis=-1)
    ddx  = np.diff(dx,   axis=-1)
    act  = np.var(data, axis=-1).mean()
    mob  = np.sqrt(np.var(dx, axis=-1) / (np.var(data, axis=-1) + 1e-12)).mean()
    comp = (np.sqrt(np.var(ddx, axis=-1) / (np.var(dx, axis=-1) + 1e-12)) /
            (np.sqrt(np.var(dx, axis=-1) / (np.var(data, axis=-1) + 1e-12)) + 1e-12)).mean()
    return float(act), float(mob), float(comp)

def perm_entropy(x, m=4, delay=1):
    """Permutation entropy of 1-D signal x — mean over channels."""
    results = []
    for ch in x:
        n = len(ch)
        patterns = {}
        for i in range(n - (m - 1) * delay):
            pat = tuple(np.argsort(ch[i: i + m * delay: delay]))
            patterns[pat] = patterns.get(pat, 0) + 1
        counts = np.array(list(patterns.values()), dtype=float)
        probs  = counts / counts.sum()
        results.append(-np.sum(probs * np.log2(probs + 1e-12)))
    return float(np.mean(results))

def band_power(psd_ch_freq, freqs, lo, hi):
    """Mean power in [lo, hi] Hz across all channels."""
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    return float(psd_ch_freq[:, idx].mean()) if len(idx) else np.nan

def regional_band_power(psd_ch_freq, freqs, ch_indices):
    """Mean power per band for a set of channel indices."""
    if len(ch_indices) == 0:
        return {b: np.nan for b in BANDS}
    sub = psd_ch_freq[ch_indices]
    return {b: band_power(sub, freqs, lo, hi) for b, (lo, hi) in BANDS.items()}


# ─────────────────────────────────────────────
# STEP 1 — PREPROCESSING + FEATURE EXTRACTION
# ─────────────────────────────────────────────
subjects = sorted([d for d in os.listdir(BASE) if d.startswith("sub-")])
print(f"\n{'='*60}")
print(f"  Found {len(subjects)} subjects: {subjects[0]} → {subjects[-1]}")
print(f"{'='*60}\n")
print("STEP 1/2 — Preprocessing + Feature Extraction")
print("-"*60)

rows = []
for si, subject in enumerate(subjects):
    for task in TASKS:
        bdf = os.path.join(BASE, subject, "eeg", f"{subject}_task-{task}_eeg.bdf")
        if not os.path.exists(bdf):
            print(f"  ⚠️  Missing: {subject}/{task}")
            continue

        print(f"  [{si+1:02d}/{len(subjects)}] {subject}/{task} ...", end=" ", flush=True)

        # ── Load & preprocess ──
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)
        raw.filter(0.5, 40, method="fir", verbose=False)
        raw.notch_filter(50, method="fir", verbose=False)
        raw.resample(SFREQ, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)

        # ── Build region→index map from actual channel names ──
        ch_names = [c.upper() for c in raw.ch_names]
        region_idx = {}
        for reg, clist in REGION_DEFS.items():
            idx = [ch_names.index(c.upper()) for c in clist if c.upper() in ch_names]
            region_idx[reg] = idx

        # Channels for FAA (Frontal Alpha Asymmetry): F4 (right) vs F3 (left)
        f3_idx = ch_names.index("F3") if "F3" in ch_names else None
        f4_idx = ch_names.index("F4") if "F4" in ch_names else None
        fz_idx = ch_names.index("FZ") if "FZ" in ch_names else None

        # ── Per-channel z-score ──
        data = raw.get_data()
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        raw._data = data

        # ── Epoch ──
        events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_LEN, overlap=0.5)
        epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=EPOCH_LEN,
                            baseline=None, preload=True, verbose=False)
        if len(epochs) == 0:
            print("no epochs — skip")
            del raw, epochs; gc.collect(); continue

        # ── Batch PSD ──
        psd_obj  = epochs.compute_psd(method="welch", fmin=0.5, fmax=40,
                                       n_fft=256, verbose=False)
        psd_vals = psd_obj.get_data()   # (n_epochs, n_ch, n_freqs)
        freqs    = psd_obj.freqs
        ep_data  = epochs.get_data()    # (n_epochs, n_ch, n_times)

        # ── Per-epoch features ──
        for ep in range(len(epochs)):
            psd = psd_vals[ep]      # (n_ch, n_freqs)
            sig = ep_data[ep]       # (n_ch, n_times)
            feat = {"Subject": subject, "Task": task}

            # 1. Regional PSD (5 bands × 5 regions = 25 features)
            for reg, idx in region_idx.items():
                rbp = regional_band_power(psd, freqs, idx)
                for band, val in rbp.items():
                    feat[f"{reg}_{band}"] = val

            # 2. Global band power & ratios (8 features)
            g = {b: band_power(psd, freqs, lo, hi) for b, (lo, hi) in BANDS.items()}
            feat.update({f"global_{b}": v for b, v in g.items()})
            feat["ratio_ThetaBeta"]  = g["Theta"]  / (g["Beta"]  + 1e-12)
            feat["ratio_AlphaBeta"]  = g["Alpha"]  / (g["Beta"]  + 1e-12)
            feat["ratio_GammaBeta"]  = g["Gamma"]  / (g["Beta"]  + 1e-12)

            # 3. Frontal Alpha Asymmetry (F4 - F3 log alpha)
            if f3_idx is not None and f4_idx is not None:
                idx_a = np.where((freqs >= 8) & (freqs <= 12))[0]
                f3_alpha = float(psd[f3_idx, idx_a].mean()) + 1e-12
                f4_alpha = float(psd[f4_idx, idx_a].mean()) + 1e-12
                feat["FAA"] = np.log(f4_alpha) - np.log(f3_alpha)
            else:
                feat["FAA"] = np.nan

            # 4. Frontal Midline Theta (Fz theta) — marker of focused attention
            if fz_idx is not None:
                idx_t = np.where((freqs >= 4) & (freqs <= 8))[0]
                feat["Fz_Theta"] = float(psd[fz_idx, idx_t].mean())
            else:
                feat["Fz_Theta"] = np.nan

            # 5. Hjorth parameters (3 features)
            act, mob, comp = hjorth(sig)
            feat["Hjorth_Activity"]   = act
            feat["Hjorth_Mobility"]   = mob
            feat["Hjorth_Complexity"] = comp

            # 6. Permutation Entropy — mean over channels (1 feature)
            feat["PermEntropy"] = perm_entropy(sig[:, ::4])   # downsample for speed

            # 7. PLV: frontal-parietal Alpha & Theta (2 features)
            f_idx = region_idx["frontal"]
            p_idx = region_idx["parietal"]
            if len(f_idx) > 0 and len(p_idx) > 0:
                alpha_sig = bandpass(sig, 8, 12)
                theta_sig = bandpass(sig, 4, 8)
                feat["PLV_FP_Alpha"] = compute_plv(alpha_sig[f_idx], alpha_sig[p_idx])
                feat["PLV_FP_Theta"] = compute_plv(theta_sig[f_idx], theta_sig[p_idx])
            else:
                feat["PLV_FP_Alpha"] = np.nan
                feat["PLV_FP_Theta"] = np.nan

            rows.append(feat)

        print(f"✅  {len(epochs)} epochs | {len(feat)-2} features/epoch")
        del raw, epochs, psd_obj, psd_vals, ep_data; gc.collect()

df = pd.DataFrame(rows)
csv_out = os.path.join(BASE, "rich_features.csv")
df.to_csv(csv_out, index=False)
print(f"\n  Saved → {csv_out}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

# ─────────────────────────────────────────────
# STEP 2 — MACHINE LEARNING
# ─────────────────────────────────────────────
print("STEP 2/2 — Model Training & Evaluation")
print("-"*60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

le = LabelEncoder()
df["label"] = le.fit_transform(df["Task"])

feature_cols = [c for c in df.columns if c not in ("Subject", "Task", "label")]
X = df[feature_cols].copy()
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
y = df["label"].values

# Subject-level split (no data leakage)
subjects_arr  = df["Subject"].values
unique_subs   = np.unique(subjects_arr)
np.random.seed(42)
perm = np.random.permutation(len(unique_subs))
split         = int(len(unique_subs) * 0.8)
train_subs    = set(unique_subs[perm[:split]])
test_subs     = set(unique_subs[perm[split:]])

train_idx = np.where([s in train_subs for s in subjects_arr])[0]
test_idx  = np.where([s in test_subs  for s in subjects_arr])[0]

X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
y_train, y_test = y[train_idx], y[test_idx]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Train: {len(train_subs)} subjects / {len(y_train):,} epochs")
print(f"  Test:  {len(test_subs)} subjects / {len(y_test):,} epochs")
print(f"  Features: {len(feature_cols)}\n")

models = {
    "Random Forest":     RandomForestClassifier(n_estimators=300, max_depth=20,
                                                 min_samples_leaf=5, random_state=42, n_jobs=-1),
    "XGBoost":           XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        eval_metric="mlogloss", verbosity=0, random_state=42),
    "LightGBM":          LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                         num_leaves=63, verbose=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                     random_state=42),
    "SVM (RBF)":         SVC(C=10, gamma="scale", kernel="rbf"),
}

print(f"  {'Model':<22} {'Accuracy':>10}  {'F1 macro':>10}")
print(f"  {'-'*22} {'-'*10}  {'-'*10}")

best_name, best_acc, best_pred = None, 0, None
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred, average="macro")
    print(f"  {name:<22} {acc*100:>9.2f}%  {f1:>10.4f}")
    if acc > best_acc:
        best_acc, best_name, best_pred = acc, name, pred

print(f"\n  🏆 Best: {best_name}  →  {best_acc*100:.2f}%\n")

# Detailed report
print(f"  Classification Report ({best_name})")
print("  " + "-"*50)
for line in classification_report(y_test, best_pred,
                                   target_names=le.classes_, digits=3).splitlines():
    print("  " + line)

# Confusion matrix
print(f"\n  Confusion Matrix  (Rows=Actual, Cols=Predicted)")
cm = confusion_matrix(y_test, best_pred)
print("  " + "".join(f"{c:>13}" for c in le.classes_))
for i, row in enumerate(cm):
    print(f"  {le.classes_[i]:<12}" + "".join(f"{v:>13}" for v in row))

# Feature importance (if RF or tree model)
best_model = list(models.values())[list(models.keys()).index(best_name)]
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols)
    top = imp.nlargest(10)
    print(f"\n  Top 10 Features ({best_name})")
    print("  " + "-"*40)
    for feat_name, val in top.items():
        bar = "█" * int(val * 200)
        print(f"  {feat_name:<28} {val:.4f}  {bar}")

print(f"\n{'='*60}")
print("  Pipeline complete ✅")
print(f"{'='*60}\n")
