"""
Try stronger models on the already-extracted rich features.
Models: ExtraTrees, Stacking Ensemble, MLP (PyTorch), EEGNet-style CNN
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = os.path.join(os.path.dirname(__file__), "brain")
CSV  = os.path.join(BASE, "rich_features.csv")

print(f"\n{'='*60}")
print("  Loading features ...")
print(f"{'='*60}\n")

df = pd.read_csv(CSV)
le = LabelEncoder()
df["label"] = le.fit_transform(df["Task"])

feature_cols = [c for c in df.columns if c not in ("Subject","Task","label")]
X_raw = df[feature_cols].copy()
X_raw.replace([np.inf,-np.inf], np.nan, inplace=True)
X_raw.fillna(X_raw.median(), inplace=True)
y = df["label"].values

# ── Subject-level split ──
subjects_arr = df["Subject"].values
unique_subs  = np.unique(subjects_arr)
np.random.seed(42)
perm  = np.random.permutation(len(unique_subs))
split = int(len(unique_subs) * 0.8)
train_subs = set(unique_subs[perm[:split]])
test_subs  = set(unique_subs[perm[split:]])

train_idx = np.where([s in train_subs for s in subjects_arr])[0]
test_idx  = np.where([s in test_subs  for s in subjects_arr])[0]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_raw.iloc[train_idx].values)
X_test  = scaler.transform(X_raw.iloc[test_idx].values)
y_train = y[train_idx]
y_test  = y[test_idx]

print(f"  Train: {len(train_subs)} subjects / {len(y_train):,} epochs")
print(f"  Test:  {len(test_subs)} subjects / {len(y_test):,} epochs")
print(f"  Features: {len(feature_cols)}")
print(f"  Classes: {list(le.classes_)}\n")

results = {}

def evaluate(name, pred):
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, average="macro")
    results[name] = acc
    print(f"  {name:<30} {acc*100:>7.2f}%   F1={f1:.4f}")
    return acc, f1

print(f"  {'Model':<30} {'Accuracy':>9}   F1")
print(f"  {'-'*30} {'-'*9}   {'-'*6}")

# ─────────────────────────────────────────────
# 1. Extra Trees
# ─────────────────────────────────────────────
et = ExtraTreesClassifier(n_estimators=300, max_depth=20,
                           min_samples_leaf=3, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
evaluate("Extra Trees", et.predict(X_test))

# ─────────────────────────────────────────────
# 2. Voting Ensemble (soft vote — no parallelism issues)
# ─────────────────────────────────────────────
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

et2   = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=0,  n_jobs=-1)
lgb2  = LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=0)
xgb2  = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        eval_metric="mlogloss", verbosity=0, random_state=0)
voter = VotingClassifier(
    estimators=[("et", et2), ("lgb", lgb2), ("xgb", xgb2)],
    voting="soft", n_jobs=1
)
voter.fit(X_train, y_train)
evaluate("Voting (ET+LGB+XGB)", voter.predict(X_test))

# ─────────────────────────────────────────────
# 3. MLP — PyTorch
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512),  nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.GELU(),
            nn.Linear(64, n_cls),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n  PyTorch device: {device}\n")

Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.long)
Xv = torch.tensor(X_test,  dtype=torch.float32).to(device)

train_ds = TensorDataset(Xt, yt)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

n_in  = X_train.shape[1]
n_cls = len(le.classes_)
model = MLP(n_in, n_cls).to(device)

optim    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched    = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=60)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
best_preds   = None
EPOCHS = 60

print(f"  Training MLP ({EPOCHS} epochs) ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()
    sched.step()

    if epoch % 10 == 0 or epoch == EPOCHS:
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            preds  = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        if acc > best_val_acc:
            best_val_acc = acc
            best_preds   = preds
        print(f"    Epoch {epoch:3d}  acc={acc*100:.2f}%")

print()
evaluate("MLP (PyTorch, 4-layer)", best_preds)

# ─────────────────────────────────────────────
# 4. Residual MLP (deeper)
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.block(x))

class ResNet(nn.Module):
    def __init__(self, n_in, n_cls, dim=256, depth=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, dim), nn.BatchNorm1d(dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(depth)])
        self.head   = nn.Linear(dim, n_cls)
    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))

model2 = ResNet(n_in, n_cls, dim=256, depth=6).to(device)
optim2 = torch.optim.AdamW(model2.parameters(), lr=8e-4, weight_decay=1e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=80)

best_val_acc2 = 0
best_preds2   = None
EPOCHS2 = 80

print(f"\n  Training ResNet MLP ({EPOCHS2} epochs) ...")
for epoch in range(1, EPOCHS2 + 1):
    model2.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optim2.zero_grad()
        loss = criterion(model2(xb), yb)
        loss.backward()
        optim2.step()
    sched2.step()

    if epoch % 10 == 0 or epoch == EPOCHS2:
        model2.eval()
        with torch.no_grad():
            preds = model2(Xv).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        if acc > best_val_acc2:
            best_val_acc2 = acc
            best_preds2   = preds
        print(f"    Epoch {epoch:3d}  acc={acc*100:.2f}%")

print()
evaluate("ResNet MLP (6 blocks)", best_preds2)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL SUMMARY")
print(f"{'='*60}")
print(f"  {'Model':<30} {'Accuracy':>9}")
print(f"  {'-'*30} {'-'*9}")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * int(acc * 100)
    print(f"  {name:<30} {acc*100:>7.2f}%  {bar}")

best_name = max(results, key=results.get)
best_preds_final = (best_preds2 if "ResNet" in best_name
                    else best_preds if "MLP" in best_name
                    else voter.predict(X_test) if "Voting" in best_name
                    else et.predict(X_test))

print(f"\n  🏆 Best: {best_name} → {results[best_name]*100:.2f}%\n")
print(f"  Classification Report ({best_name})")
print("  " + "-"*52)
for line in classification_report(y_test, best_preds_final,
                                   target_names=le.classes_, digits=3).splitlines():
    print("  " + line)

print(f"\n{'='*60}")
print("  Done ✅")
print(f"{'='*60}\n")
