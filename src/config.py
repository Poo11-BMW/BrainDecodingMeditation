"""Configuration loading and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BandConfig:
    delta: tuple[float, float] = (0.5, 4.0)
    theta: tuple[float, float] = (4.0, 8.0)
    alpha: tuple[float, float] = (8.0, 12.0)
    beta: tuple[float, float] = (12.0, 30.0)
    gamma: tuple[float, float] = (30.0, 40.0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "Delta": self.delta,
            "Theta": self.theta,
            "Alpha": self.alpha,
            "Beta": self.beta,
            "Gamma": self.gamma,
        }


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    safety_gap_epochs: int = 1  # epochs of gap at partition boundaries


@dataclass
class EpochConfig:
    sfreq: int = 128
    epoch_len: float = 2.0       # seconds
    epoch_overlap: float = 0.0   # seconds; 0 = non-overlapping after fix


@dataclass
class ModelConfig:
    random_seed: int = 42
    n_estimators: int = 300
    lgbm_num_leaves: int = 63
    lgbm_learning_rate: float = 0.05
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 2


@dataclass
class Config:
    data_dir: Path = Path("brain")
    results_dir: Path = Path("results")
    artifacts_dir: Path = Path("artifacts")
    tasks: list[str] = field(default_factory=lambda: ["med1breath", "med2", "think1", "think2"])
    meditation_tasks: list[str] = field(default_factory=lambda: ["med1breath", "med2"])
    thinking_tasks: list[str] = field(default_factory=lambda: ["think1", "think2"])
    bands: BandConfig = field(default_factory=BandConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    low_confidence_threshold: float = 0.6
    api_port: int = 8000

    region_defs: dict[str, list[str]] = field(default_factory=lambda: {
        "frontal":   ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "F7", "F3", "Fz", "F4", "F8"],
        "central":   ["FC5", "FC1", "FCz", "FC2", "FC6", "C3", "Cz", "C4"],
        "temporal":  ["T7", "T8", "TP7", "TP8"],
        "parietal":  ["CP5", "CP1", "CPz", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
        "occipital": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
    })


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if "data_dir" in raw:
        cfg.data_dir = Path(raw["data_dir"])
    if "results_dir" in raw:
        cfg.results_dir = Path(raw["results_dir"])
    if "artifacts_dir" in raw:
        cfg.artifacts_dir = Path(raw["artifacts_dir"])
    if "tasks" in raw:
        cfg.tasks = raw["tasks"]
    if "meditation_tasks" in raw:
        cfg.meditation_tasks = raw["meditation_tasks"]
    if "thinking_tasks" in raw:
        cfg.thinking_tasks = raw["thinking_tasks"]
    if "low_confidence_threshold" in raw:
        cfg.low_confidence_threshold = float(raw["low_confidence_threshold"])
    if "api_port" in raw:
        cfg.api_port = int(raw["api_port"])

    bands_raw = raw.get("bands", {})
    if bands_raw:
        for band_name in ("delta", "theta", "alpha", "beta", "gamma"):
            if band_name in bands_raw:
                setattr(cfg.bands, band_name, tuple(bands_raw[band_name]))

    split_raw = raw.get("split", {})
    if split_raw:
        for k in ("train_frac", "val_frac", "test_frac"):
            if k in split_raw:
                setattr(cfg.split, k, float(split_raw[k]))
        if "safety_gap_epochs" in split_raw:
            cfg.split.safety_gap_epochs = int(split_raw["safety_gap_epochs"])

    epoch_raw = raw.get("epoch", {})
    if epoch_raw:
        if "sfreq" in epoch_raw:
            cfg.epoch.sfreq = int(epoch_raw["sfreq"])
        if "epoch_len" in epoch_raw:
            cfg.epoch.epoch_len = float(epoch_raw["epoch_len"])
        if "epoch_overlap" in epoch_raw:
            cfg.epoch.epoch_overlap = float(epoch_raw["epoch_overlap"])

    model_raw = raw.get("model", {})
    if model_raw:
        for k in vars(cfg.model):
            if k in model_raw:
                v = model_raw[k]
                expected_type = type(getattr(cfg.model, k))
                setattr(cfg.model, k, expected_type(v))

    if "region_defs" in raw:
        cfg.region_defs = raw["region_defs"]

    logger.info("Loaded config from %s", path)
    return cfg
