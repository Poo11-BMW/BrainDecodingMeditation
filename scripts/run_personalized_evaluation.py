"""
Personalized within-subject evaluation.

DESIGN: Each subject's recording is split chronologically (70/10/20).
        Preprocessing (imputer + scaler) is fitted on the training split
        of each subject independently.  Test data never influences any
        preprocessing or model step.

Usage:
    python scripts/run_personalized_evaluation.py --config configs/default.yaml

Outputs:
    results/personalized_binary_metrics.json
    results/personalized_four_class_metrics.json
    results/per_subject_results.csv
    results/figures/per_subject_accuracy_*.png
    results/figures/model_comparison_personalized.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loading import load_rich_features
from src.evaluation import aggregate_results, save_metrics
from src.training import run_personalized_evaluation
from src.visualization import plot_model_comparison, plot_per_subject_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def _add_binary_label(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    df["binary"] = df["Task"].apply(
        lambda t: 0 if t in cfg.meditation_tasks else 1
    )
    return df


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    df = load_rich_features(cfg.data_dir)

    # Validate that the split column exists (produced by build_features.py)
    if "split" not in df.columns:
        logger.error(
            "'split' column missing. Re-run build_features.py with the new pipeline.\n"
            "  python scripts/build_features.py --config %s --overwrite", args.config
        )
        sys.exit(1)

    df = _add_binary_label(df, cfg)
    results_dir = Path(cfg.results_dir)
    figures_dir = results_dir / "figures"

    # ── Binary: meditation vs thinking ──────────────────────────────────────
    logger.info("=== Personalized Binary (meditation vs thinking) ===")
    binary_rows = run_personalized_evaluation(
        df,
        label_col="binary",
        model_cfg=cfg.model,
        task_type="binary",
        label_names=["meditation", "thinking"],
    )
    binary_df = pd.DataFrame(binary_rows)

    agg_binary = aggregate_results(
        binary_rows,
        scalar_keys=["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc",
                     "sensitivity", "specificity"],
    )
    payload_binary = {
        "protocol":    "personalized",
        "task_type":   "binary",
        "aggregate":   agg_binary,
        "per_fold":    binary_rows,
        "note":        "Personalized within-subject evaluation. Each subject's own data "
                       "split chronologically 70/10/20. Preprocessing fitted on train only.",
    }
    save_metrics(payload_binary, results_dir / "personalized_binary_metrics.json")

    # ── Four-class: which specific task ─────────────────────────────────────
    logger.info("=== Personalized Four-Class (which task) ===")
    four_rows = run_personalized_evaluation(
        df,
        label_col="Task",
        model_cfg=cfg.model,
        task_type="multiclass",
        label_names=cfg.tasks,
    )
    four_df = pd.DataFrame(four_rows)

    agg_four = aggregate_results(
        four_rows,
        scalar_keys=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    payload_four = {
        "protocol":  "personalized",
        "task_type": "four_class",
        "aggregate": agg_four,
        "per_fold":  four_rows,
        "note":      "Four-class personalized evaluation (med1breath / med2 / think1 / think2).",
    }
    save_metrics(payload_four, results_dir / "personalized_four_class_metrics.json")

    # ── Combine and save per-subject CSV ────────────────────────────────────
    all_rows = pd.concat([binary_df, four_df], ignore_index=True)
    all_rows.to_csv(results_dir / "per_subject_results.csv", index=False)
    logger.info("Saved per_subject_results.csv (%d rows)", len(all_rows))

    # ── Figures ─────────────────────────────────────────────────────────────
    try:
        plot_model_comparison(binary_df, figures_dir, protocol="personalized")
        plot_per_subject_accuracy(binary_df, figures_dir, model_name="LightGBM")
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # ── Console summary ─────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("PERSONALIZED BINARY RESULTS (mean across subjects)")
    logger.info("%s", "-" * 60)
    for metric, stats in agg_binary.items():
        logger.info("  %-25s %.3f ± %.3f", metric, stats["mean"], stats["std"])

    if binary_df is not None and len(binary_df) > 0:
        best_model = (
            binary_df.groupby("model")["accuracy"].mean().idxmax()
        )
        logger.info("\nBest model: %s", best_model)
        model_summary = binary_df[binary_df["model"] == best_model].groupby("subject")["accuracy"].first()
        logger.info("Per-subject accuracy:\n%s", model_summary.to_string())


if __name__ == "__main__":
    main()
