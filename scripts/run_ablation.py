"""
Feature ablation study.

Runs personalized evaluation with:
  1. All 41 features
  2. Each feature family independently
  3. All features minus one family at a time

Output: results/feature_ablation.csv

Usage:
    python scripts/run_ablation.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ablation import run_ablation
from src.config import load_config
from src.data_loading import load_rich_features
from src.preprocessing import get_feature_cols
from src.visualization import plot_ablation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="LightGBM", help="Model name to use for ablation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    df = load_rich_features(cfg.data_dir)
    if "split" not in df.columns:
        logger.error("'split' column missing. Re-run build_features.py first.")
        sys.exit(1)

    df["binary"] = df["Task"].apply(
        lambda t: 0 if t in cfg.meditation_tasks else 1
    )

    feature_cols = get_feature_cols(df)
    logger.info("Running ablation on %d features across %d subjects", len(feature_cols), df["Subject"].nunique())

    ablation_df = run_ablation(
        df=df,
        feature_cols=feature_cols,
        label_col="binary",
        model_cfg=cfg.model,
        model_name=args.model,
    )

    out = Path(cfg.results_dir) / "feature_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out, index=False)
    logger.info("Saved → %s", out)

    try:
        plot_ablation(ablation_df, Path(cfg.results_dir) / "figures")
    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    logger.info("\n%s", "=" * 65)
    logger.info("FEATURE ABLATION RESULTS (binary, %s)", args.model)
    logger.info("%s", "-" * 65)
    print(ablation_df[["condition", "n_features", "mean_accuracy", "std_accuracy", "mean_macro_f1"]].to_string(index=False))

    # Interpretation (descriptive, no causal claims)
    best_solo = ablation_df[ablation_df["condition"].str.startswith("only_")].iloc[0]
    logger.info(
        "\nMost predictive single group: %s (mean_acc=%.3f)",
        best_solo["condition"], best_solo["mean_accuracy"],
    )
    logger.info(
        "Note: ablation measures predictive association within this dataset. "
        "Causation and generalisability require independent validation."
    )


if __name__ == "__main__":
    main()
