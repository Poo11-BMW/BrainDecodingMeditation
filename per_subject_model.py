"""
Per-Subject EEG Classification
================================
Strategy : for each subject, train on 80% of their epochs, test on 20%.
           This mimics real BCI calibration (you record from a person, train,
           then classify their future brain states).

Also runs binary (meditation vs thinking) which is a cleaner problem.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

BASE = os.path.join(os.path.dirname(__file__), "brain")
CSV  = os.path.join(BASE, "rich_features.csv")

df = pd.read_csv(CSV)
feature_cols = [c for c in df.columns if c not in ("Subject","Task","label")]
feature_cols = [c for c in feature_cols if c in df.columns]

df["binary"] = df["Task"].apply(
    lambda t: "meditation" if t in ("med1breath","med2") else "thinking"
)

# ─────────────────────────────────────────────────────────────
# Helper — train/test one subject
# ─────────────────────────────────────────────────────────────
def run_subject(sub_df, label_col, model_fn):
    X = sub_df[feature_cols].copy()
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(sub_df[label_col])

    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None

    # Per-subject normalisation (critical — removes individual DC offset)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.values)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X_sc, y))

    model = model_fn()
    model.fit(X_sc[tr], y[tr])
    pred  = model.predict(X_sc[te])

    acc = accuracy_score(y[te], pred)
    f1  = f1_score(y[te], pred, average="macro")
    return acc, f1

subjects = sorted(df["Subject"].unique())

# ─────────────────────────────────────────────────────────────
# Models to compare
# ─────────────────────────────────────────────────────────────
model_fns = {
    "Random Forest":  lambda: RandomForestClassifier(n_estimators=300, max_depth=20,
                                                      min_samples_leaf=2, random_state=42,
                                                      n_jobs=-1),
    "Extra Trees":    lambda: ExtraTreesClassifier(n_estimators=300, max_depth=20,
                                                    min_samples_leaf=2, random_state=42,
                                                    n_jobs=-1),
    "XGBoost":        lambda: XGBClassifier(n_estimators=300, max_depth=7,
                                             learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8,
                                             eval_metric="mlogloss", verbosity=0,
                                             random_state=42),
    "LightGBM":       lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              num_leaves=63, verbose=-1, random_state=42),
}

# ─────────────────────────────────────────────────────────────
# RUN — 4-class
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  4-CLASS: med1breath / med2 / think1 / think2")
print(f"{'='*65}")

for mname, mfn in model_fns.items():
    accs, f1s = [], []
    for sub in subjects:
        sub_df = df[df["Subject"] == sub]
        acc, f1 = run_subject(sub_df, "Task", mfn)
        if acc is not None:
            accs.append(acc); f1s.append(f1)

    mean_acc = np.mean(accs)*100
    std_acc  = np.std(accs)*100
    print(f"  {mname:<18}  {mean_acc:.2f}% ± {std_acc:.1f}%   "
          f"F1={np.mean(f1s):.3f}   "
          f"[min {min(accs)*100:.0f}% / max {max(accs)*100:.0f}%]")

# ─────────────────────────────────────────────────────────────
# RUN — Binary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  BINARY: meditation vs thinking")
print(f"{'='*65}")

for mname, mfn in model_fns.items():
    accs, f1s = [], []
    for sub in subjects:
        sub_df = df[df["Subject"] == sub]
        acc, f1 = run_subject(sub_df, "binary", mfn)
        if acc is not None:
            accs.append(acc); f1s.append(f1)

    mean_acc = np.mean(accs)*100
    std_acc  = np.std(accs)*100
    print(f"  {mname:<18}  {mean_acc:.2f}% ± {std_acc:.1f}%   "
          f"F1={np.mean(f1s):.3f}   "
          f"[min {min(accs)*100:.0f}% / max {max(accs)*100:.0f}%]")

# ─────────────────────────────────────────────────────────────
# BEST MODEL DETAIL — Binary LightGBM, per subject breakdown
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  PER-SUBJECT BREAKDOWN  (Binary, LightGBM)")
print(f"{'='*65}")
print(f"  {'Subject':<12} {'Accuracy':>10}  {'F1':>8}  Bar")
print(f"  {'-'*12} {'-'*10}  {'-'*8}  {'-'*20}")

accs_detail = []
for sub in subjects:
    sub_df = df[df["Subject"] == sub]
    X = sub_df[feature_cols].copy()
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    le = LabelEncoder()
    y  = le.fit_transform(sub_df["binary"])
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.values)
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X_sc, y))
    model  = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=63, verbose=-1, random_state=42)
    model.fit(X_sc[tr], y[tr])
    pred   = model.predict(X_sc[te])
    acc    = accuracy_score(y[te], pred)
    f1     = f1_score(y[te], pred, average="macro")
    accs_detail.append(acc)
    bar    = "█" * int(acc * 30)
    print(f"  {sub:<12} {acc*100:>9.1f}%  {f1:>8.3f}  {bar}")

print(f"\n  {'MEAN':<12} {np.mean(accs_detail)*100:>9.1f}%")
print(f"  {'STD':<12} {np.std(accs_detail)*100:>9.1f}%")

# ─────────────────────────────────────────────────────────────
# FULL REPORT — best model on one subject as example
# ─────────────────────────────────────────────────────────────
best_sub = subjects[np.argmax(accs_detail)]
sub_df   = df[df["Subject"] == best_sub]
X = sub_df[feature_cols].copy()
X.replace([np.inf,-np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
le_b  = LabelEncoder()
y_b   = le_b.fit_transform(sub_df["binary"])
scaler2 = StandardScaler()
X_sc2   = scaler2.fit_transform(X.values)
sss2    = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr2, te2 = next(sss2.split(X_sc2, y_b))
mdl2    = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                          num_leaves=63, verbose=-1, random_state=42)
mdl2.fit(X_sc2[tr2], y_b[tr2])
pred2 = mdl2.predict(X_sc2[te2])

print(f"\n  Example report — {best_sub} (best subject)")
print("  " + "-"*45)
for line in classification_report(y_b[te2], pred2,
                                   target_names=le_b.classes_,
                                   digits=3).splitlines():
    print("  " + line)

print(f"\n{'='*65}")
print("  Done ✅")
print(f"{'='*65}\n")
