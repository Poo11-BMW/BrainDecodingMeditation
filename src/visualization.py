"""
Publication-quality figures for EEG meditation classification results.
All matplotlib figures save to results/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot accuracy, balanced accuracy, macro F1, and ROC-AUC vs calibration time.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Columns: calibration_seconds, accuracy, balanced_accuracy, macro_f1,
                 roc_auc, accuracy_std, etc.
    output_dir : Path
    """
    plt = _get_plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Calibration Duration vs Classification Performance\n"
        "(mean ± std across subjects, chronological splits)",
        fontsize=13, fontweight="bold",
    )
    metrics = [
        ("accuracy",         "Accuracy"),
        ("balanced_accuracy","Balanced Accuracy"),
        ("macro_f1",         "Macro F1"),
        ("roc_auc",          "ROC-AUC"),
    ]
    for ax, (col, label) in zip(axes.flat, metrics):
        if col not in calibration_df.columns:
            ax.set_visible(False)
            continue
        t = calibration_df["calibration_seconds"]
        m = calibration_df[col]
        s = calibration_df.get(f"{col}_std", pd.Series([0] * len(m)))
        ax.plot(t, m, "o-", color="#1E88E5", lw=2, ms=7)
        ax.fill_between(t, m - s, m + s, alpha=0.2, color="#1E88E5")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chance")
        ax.set_xlabel("Calibration time (seconds)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = Path(output_dir) / "calibration_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved → %s", out)


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    protocol: str = "personalized",
) -> None:
    """Bar chart comparing models across accuracy, balanced accuracy, macro F1."""
    plt = _get_plt()
    models = results_df["model"].unique()
    metrics_to_plot = ["accuracy", "balanced_accuracy", "macro_f1"]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 5))
    colors = ["#1E88E5", "#FFC107", "#4CAF50"]
    for i, metric in enumerate(metrics_to_plot):
        if metric not in results_df.columns:
            continue
        means = [results_df[results_df["model"] == m][metric].mean() for m in models]
        ax.bar(x + i * width, means, width, label=metric.replace("_", " ").title(), color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {protocol.replace('_',' ').title()} Protocol")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = Path(output_dir) / f"model_comparison_{protocol}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved → %s", out)


def plot_per_subject_accuracy(
    per_subject_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "LightGBM",
    metric: str = "accuracy",
) -> None:
    """Horizontal bar chart of per-subject accuracy for one model."""
    plt = _get_plt()
    sub_df = per_subject_df[per_subject_df["model"] == model_name].copy()
    if sub_df.empty:
        logger.warning("No rows for model '%s' in per_subject_df", model_name)
        return
    sub_df = sub_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(sub_df) * 0.35)))
    colors = ["#4CAF50" if v >= 0.85 else "#FFC107" if v >= 0.70 else "#F44336"
              for v in sub_df[metric]]
    ax.barh(sub_df["subject"], sub_df[metric], color=colors, alpha=0.85)
    ax.axvline(sub_df[metric].mean(), color="navy", ls="--", lw=1.5,
               label=f"Mean = {sub_df[metric].mean():.2f}")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-Subject {metric.replace('_',' ').title()} — {model_name}\n(personalized chronological evaluation)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / f"per_subject_{metric}_{model_name.lower().replace(' ','_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved → %s", out)


def plot_ablation(
    ablation_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Bar chart of mean accuracy by ablation condition."""
    plt = _get_plt()
    df = ablation_df.copy()
    # Show only_* and all row for readability
    df_only = df[df["condition"].str.startswith("only_") | (df["condition"] == "all_features")]
    if df_only.empty:
        df_only = df
    df_only = df_only.sort_values("mean_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_only) * 0.5)))
    ax.barh(df_only["condition"], df_only["mean_accuracy"],
            xerr=df_only.get("std_accuracy", pd.Series([0]*len(df_only))),
            color="#1E88E5", alpha=0.85, capsize=4)
    ax.set_xlabel("Mean Accuracy (personalized)")
    ax.set_title("Feature Ablation — Individual Group Performance")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "feature_ablation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved → %s", out)
