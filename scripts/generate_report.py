"""
Generate a consolidated plain-text + CSV model comparison report
from the results/ JSON files produced by the evaluation scripts.

Usage:
    python scripts/generate_report.py

Output: results/model_comparison_report.csv  (machine-readable)
        results/summary_report.txt            (human-readable)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


RESULTS_DIR = Path("results")
RESULT_FILES = {
    "personalized_binary":       RESULTS_DIR / "personalized_binary_metrics.json",
    "personalized_four_class":   RESULTS_DIR / "personalized_four_class_metrics.json",
    "unseen_subject_binary":     RESULTS_DIR / "unseen_subject_binary_metrics.json",
    "unseen_subject_four_class": RESULTS_DIR / "unseen_subject_four_class_metrics.json",
}

SCALAR_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]


def load_result(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("Missing: %s — run the corresponding evaluation script first.", path)
        return None
    with open(path) as f:
        return json.load(f)


def extract_model_rows(data: dict, protocol: str, task_type: str) -> list[dict]:
    """Extract per-model aggregate statistics from a result dict."""
    per_fold = data.get("per_fold", [])
    if not per_fold:
        return []
    df = pd.DataFrame(per_fold)
    rows = []
    for model, grp in df.groupby("model"):
        row = {"model": model, "protocol": protocol, "task_type": task_type}
        for m in SCALAR_METRICS:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[m + "_mean"] = round(float(vals.mean()), 4) if len(vals) else None
                row[m + "_std"]  = round(float(vals.std()), 4)  if len(vals) else None
        for col in ("train_time_s", "p50_latency_us", "p95_latency_us"):
            if col in grp.columns:
                row[col] = round(float(grp[col].mean()), 4)
        rows.append(row)
    return rows


def main() -> None:
    all_rows = []
    for key, path in RESULT_FILES.items():
        data = load_result(path)
        if data is None:
            continue
        protocol  = data.get("protocol", key)
        task_type = data.get("task_type", "")
        rows = extract_model_rows(data, protocol, task_type)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No result files found. Run evaluation scripts first.")
        sys.exit(1)

    report_df = pd.DataFrame(all_rows)
    report_df = report_df.sort_values(["protocol", "task_type", "accuracy_mean"], ascending=[True, True, False])

    csv_out = RESULTS_DIR / "model_comparison_report.csv"
    report_df.to_csv(csv_out, index=False)
    logger.info("Saved → %s", csv_out)

    # ── Human-readable text report ──────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("EEG MEDITATION CLASSIFIER — MODEL COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("IMPORTANT: Two evaluation protocols are reported separately.")
    lines.append("  'personalized'     = same subject, chronological 70/10/20 split")
    lines.append("  'unseen_subject'   = Leave-One-Subject-Out (new person, not in training)")
    lines.append("")
    lines.append("These measure DIFFERENT things. Do not conflate them.")
    lines.append("")

    for (protocol, task_type), grp in report_df.groupby(["protocol", "task_type"]):
        lines.append(f"{'─'*70}")
        lines.append(f"Protocol : {protocol}")
        lines.append(f"Task     : {task_type}")
        lines.append(f"{'─'*70}")
        header = f"  {'Model':<22} {'Acc':>8} {'BalAcc':>8} {'F1':>8} {'AUC':>8}  {'TrainTime(s)':>14}"
        lines.append(header)
        lines.append("  " + "-" * 65)
        for _, row in grp.iterrows():
            acc  = f"{row['accuracy_mean']:.3f}" if pd.notna(row.get("accuracy_mean")) else "  N/A "
            bacc = f"{row['balanced_accuracy_mean']:.3f}" if pd.notna(row.get("balanced_accuracy_mean")) else "  N/A "
            f1   = f"{row['macro_f1_mean']:.3f}"   if pd.notna(row.get("macro_f1_mean"))   else "  N/A "
            auc  = f"{row['roc_auc_mean']:.3f}"    if pd.notna(row.get("roc_auc_mean"))    else "  N/A "
            tt   = f"{row['train_time_s']:.2f}"    if pd.notna(row.get("train_time_s"))    else "  N/A "
            lines.append(f"  {row['model']:<22} {acc:>8} {bacc:>8} {f1:>8} {auc:>8}  {tt:>14}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Limitations:")
    lines.append("  - N=20 subjects; results may not generalise to other populations.")
    lines.append("  - Inter-subject EEG variability is large; personalized >> unseen.")
    lines.append("  - No clinical validation. Confidence scores ≠ meditation depth.")
    lines.append("=" * 70)

    txt_out = RESULTS_DIR / "summary_report.txt"
    with open(txt_out, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved → %s", txt_out)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
