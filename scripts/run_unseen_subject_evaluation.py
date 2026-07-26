"""
Unseen-subject (Leave-One-Subject-Out) evaluation.

DESIGN: For each fold, one subject is held out entirely.
        The remaining subjects form the training set.
        Imputer and scaler are fitted on training subjects only.
        The test subject's data NEVER influences any preprocessing step.
        This answers: "Can the model generalise to a new person?"

Usage:
    python scripts/run_unseen_subject_evaluation.py --config configs/default.yaml

Outputs:
    results/unseen_subject_binary_metrics.json
    results/unseen_subject_four_class_metrics.json
    results/fold_results.csv
    results/figures/model_comparison_unseen_subject.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loading import load_rich_features
from src.evaluation import aggregate_results, save_metrics
from src.splitting import subject_grouped_folds
from src.training import run_unseen_subject_evaluation
from src.visualization import plot_model_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--n-folds", type=int, default=None,
        help="Number of folds. Default=None uses true LOSO (one fold per subject).",
    )
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
    df = _add_binary_label(df, cfg)

    subjects = sorted(df["Subject"].unique())
    logger.info("Subjects available: %d", len(subjects))

    if len(subjects) < 3:
        logger.error(
            "LOSO requires at least 3 subjects. Found %d. "
            "Download the full dataset first.", len(subjects)
        )
        sys.exit(1)

    folds = subject_grouped_folds(subjects, n_splits=args.n_folds)
    logger.info("Running LOSO with %d folds.", len(folds))

    results_dir = Path(cfg.results_dir)
    figures_dir = results_dir / "figures"

    # ── Binary ────────────────────────────────────────────────────────────
    logger.info("=== Unseen-Subject Binary (meditation vs thinking) ===")
    binary_rows = run_unseen_subject_evaluation(
        df,
        label_col="binary",
        model_cfg=cfg.model,
        folds=folds,
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
        "protocol":  "unseen_subject_loso",
        "task_type": "binary",
        "n_folds":   len(folds),
        "aggregate": agg_binary,
        "per_fold":  binary_rows,
        "note":      (
            "Leave-One-Subject-Out evaluation. Test subject data never used in "
            "preprocessing, feature selection, or model fitting. "
            "This measures generalisation to new individuals."
        ),
    }
    save_metrics(payload_binary, results_dir / "unseen_subject_binary_metrics.json")

    # ── Four-class ────────────────────────────────────────────────────────
    logger.info("=== Unseen-Subject Four-Class ===")
    four_rows = run_unseen_subject_evaluation(
        df,
        label_col="Task",
        model_cfg=cfg.model,
        folds=folds,
        task_type="multiclass",
        label_names=cfg.tasks,
    )
    four_df = pd.DataFrame(four_rows)

    agg_four = aggregate_results(
        four_rows,
        scalar_keys=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    payload_four = {
        "protocol":  "unseen_subject_loso",
        "task_type": "four_class",
        "n_folds":   len(folds),
        "aggregate": agg_four,
        "per_fold":  four_rows,
        "note":      "LOSO four-class evaluation.",
    }
    save_metrics(payload_four, results_dir / "unseen_subject_four_class_metrics.json")

    # ── Save fold-level CSV ───────────────────────────────────────────────
    all_rows = pd.concat([binary_df, four_df], ignore_index=True)
    all_rows.to_csv(results_dir / "fold_results.csv", index=False)
    logger.info("Saved fold_results.csv (%d rows)", len(all_rows))

    try:
        plot_model_comparison(binary_df, figures_dir, protocol="unseen_subject")
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # ── Console summary ───────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("UNSEEN-SUBJECT BINARY RESULTS (mean across LOSO folds)")
    logger.info("%s", "-" * 60)
    for metric, stats in agg_binary.items():
        logger.info("  %-25s %.3f ± %.3f", metric, stats["mean"], stats["std"])

    logger.info(
        "\nIMPORTANT: Unseen-subject accuracy is typically significantly lower "
        "than personalized accuracy due to large inter-subject EEG variability."
    )


if __name__ == "__main__":
    main()
