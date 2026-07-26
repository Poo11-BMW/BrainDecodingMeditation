"""Load and validate raw EEG data and participant metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def list_subjects(data_dir: Path) -> list[str]:
    """Return sorted list of subject IDs that have a sub-XXX directory."""
    data_dir = Path(data_dir)
    subjects = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    logger.info("Found %d subjects in %s", len(subjects), data_dir)
    return subjects


def bdf_path(data_dir: Path, subject: str, task: str) -> Path:
    """Construct the expected path to a BDF EEG file."""
    return Path(data_dir) / subject / "eeg" / f"{subject}_task-{task}_eeg.bdf"


def load_participants(data_dir: Path) -> pd.DataFrame:
    """Load participants.tsv with basic validation."""
    tsv = Path(data_dir) / "participants.tsv"
    if not tsv.exists():
        logger.warning("participants.tsv not found at %s", tsv)
        return pd.DataFrame()
    df = pd.read_csv(tsv, sep="\t")
    df = df.rename(columns={"participant_id": "Subject"})
    logger.debug("Loaded participants: %d rows", len(df))
    return df


def load_rich_features(data_dir: Path) -> pd.DataFrame:
    """Load precomputed rich_features.csv, raising if absent."""
    csv = Path(data_dir) / "rich_features.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"rich_features.csv not found at {csv}. "
            "Run: python scripts/build_features.py --config configs/default.yaml"
        )
    df = pd.read_csv(csv)
    logger.info("Loaded rich_features.csv: %d rows × %d cols", len(df), df.shape[1])
    return df
