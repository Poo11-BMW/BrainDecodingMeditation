"""
EEG feature extraction — 41 neuroscience-grounded features per 2-second epoch.

Feature families:
  F1  Regional band power         (5 bands × 5 regions = 25)
  F2  Global band power & ratios  (5 + 3 = 8)
  F3  Frontal Alpha Asymmetry     (1)
  F4  Frontal midline Theta       (1)
  F5  Hjorth parameters           (3)
  F6  Permutation entropy         (1)
  F7  Phase-Locking Value         (2)
  ──────────────────────────────────
  Total                           41
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

logger = logging.getLogger(__name__)

# ── Exported feature group membership ─────────────────────────────────────────
FEATURE_GROUPS: dict[str, list[str]] = {}  # filled at module load via _register_groups()


# ── Low-level signal helpers ───────────────────────────────────────────────────

def bandpass(data: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """Zero-phase bandpass filter (n_channels, n_times)."""
    nyq = fs / 2.0
    lo  = max(lo, 0.1)
    hi  = min(hi, nyq - 0.1)
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def compute_plv(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Phase Locking Value between two channel groups.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray, shape (n_channels, n_times)

    Returns
    -------
    float in [0, 1]
    """
    h_a   = hilbert(sig_a.mean(axis=0))
    h_b   = hilbert(sig_b.mean(axis=0))
    phase = np.angle(h_a) - np.angle(h_b)
    plv   = float(np.abs(np.mean(np.exp(1j * phase))))
    # PLV must be in [0, 1] by construction; clamp for floating-point safety
    return float(np.clip(plv, 0.0, 1.0))


def hjorth(data: np.ndarray) -> tuple[float, float, float]:
    """
    Hjorth Activity, Mobility, Complexity.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)

    Returns
    -------
    activity, mobility, complexity : float
    """
    dx   = np.diff(data, axis=-1)
    ddx  = np.diff(dx,   axis=-1)
    act  = float(np.var(data, axis=-1).mean())
    mob  = float(np.sqrt(np.var(dx, axis=-1) / (np.var(data, axis=-1) + 1e-12)).mean())
    comp = float(
        (np.sqrt(np.var(ddx, axis=-1) / (np.var(dx, axis=-1) + 1e-12)) /
         (np.sqrt(np.var(dx, axis=-1) / (np.var(data, axis=-1) + 1e-12)) + 1e-12)).mean()
    )
    return act, mob, comp


def perm_entropy(x: np.ndarray, m: int = 4, delay: int = 1) -> float:
    """
    Permutation entropy of a multichannel signal, averaged over channels.

    Parameters
    ----------
    x : np.ndarray, shape (n_channels, n_times)
    """
    results = []
    for ch in x:
        n = len(ch)
        patterns: dict[tuple, int] = {}
        for i in range(n - (m - 1) * delay):
            pat = tuple(np.argsort(ch[i: i + m * delay: delay]))
            patterns[pat] = patterns.get(pat, 0) + 1
        counts = np.array(list(patterns.values()), dtype=float)
        probs  = counts / counts.sum()
        results.append(float(-np.sum(probs * np.log2(probs + 1e-12))))
    return float(np.mean(results))


def band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Mean PSD power in [lo, hi] Hz across all channels."""
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    return float(psd[:, idx].mean()) if len(idx) else float("nan")


def regional_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    ch_indices: list[int],
    bands: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Mean power per frequency band for a set of channel indices."""
    if len(ch_indices) == 0:
        return {b: float("nan") for b in bands}
    sub = psd[ch_indices]
    return {b: band_power(sub, freqs, lo, hi) for b, (lo, hi) in bands.items()}


# ── Main extraction function ───────────────────────────────────────────────────

def extract_features(
    epoch_data: np.ndarray,
    psd_data: np.ndarray,
    freqs: np.ndarray,
    ch_names: list[str],
    region_defs: dict[str, list[str]],
    bands: dict[str, tuple[float, float]],
    sfreq: float,
) -> dict[str, Any]:
    """
    Extract all 41 features from one epoch.

    Parameters
    ----------
    epoch_data : np.ndarray, shape (n_channels, n_times)
        Raw (preprocessed) signal for this epoch.
    psd_data : np.ndarray, shape (n_channels, n_freqs)
        Pre-computed Welch PSD for this epoch.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency axis for psd_data.
    ch_names : list[str]
        Channel names (uppercase) matching axis-0 of epoch_data/psd_data.
    region_defs : dict[str, list[str]]
        Brain region → channel name mapping (uppercase).
    bands : dict[str, tuple[float, float]]
        Frequency band → (lo, hi) Hz.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    dict mapping feature name → float value
    """
    feat: dict[str, Any] = {}
    ch_upper = [c.upper() for c in ch_names]

    # Build region → index map
    region_idx: dict[str, list[int]] = {}
    for reg, clist in region_defs.items():
        idx = [ch_upper.index(c.upper()) for c in clist if c.upper() in ch_upper]
        region_idx[reg] = idx

    # F1 — Regional band power (5 bands × 5 regions = 25)
    for reg, idx in region_idx.items():
        rbp = regional_band_power(psd_data, freqs, idx, bands)
        for band_name, val in rbp.items():
            feat[f"{reg}_{band_name}"] = val

    # F2 — Global band power & ratios (8)
    g = {b: band_power(psd_data, freqs, lo, hi) for b, (lo, hi) in bands.items()}
    feat.update({f"global_{b}": v for b, v in g.items()})
    feat["ratio_ThetaBeta"] = g["Theta"] / (g["Beta"] + 1e-12)
    feat["ratio_AlphaBeta"] = g["Alpha"] / (g["Beta"] + 1e-12)
    feat["ratio_GammaBeta"] = g["Gamma"] / (g["Beta"] + 1e-12)

    # F3 — Frontal Alpha Asymmetry (1)
    f3_idx = ch_upper.index("F3") if "F3" in ch_upper else None
    f4_idx = ch_upper.index("F4") if "F4" in ch_upper else None
    if f3_idx is not None and f4_idx is not None:
        idx_a = np.where((freqs >= 8) & (freqs <= 12))[0]
        f3_alpha = float(psd_data[f3_idx, idx_a].mean()) + 1e-12
        f4_alpha = float(psd_data[f4_idx, idx_a].mean()) + 1e-12
        feat["FAA"] = float(np.log(f4_alpha) - np.log(f3_alpha))
    else:
        feat["FAA"] = float("nan")

    # F4 — Frontal midline Theta (1)
    fz_idx = ch_upper.index("FZ") if "FZ" in ch_upper else None
    if fz_idx is not None:
        idx_t = np.where((freqs >= 4) & (freqs <= 8))[0]
        feat["Fz_Theta"] = float(psd_data[fz_idx, idx_t].mean())
    else:
        feat["Fz_Theta"] = float("nan")

    # F5 — Hjorth (3)
    act, mob, comp = hjorth(epoch_data)
    feat["Hjorth_Activity"]   = act
    feat["Hjorth_Mobility"]   = mob
    feat["Hjorth_Complexity"] = comp

    # F6 — Permutation entropy (1) — downsample for speed
    feat["PermEntropy"] = perm_entropy(epoch_data[:, ::4])

    # F7 — PLV frontal-parietal Alpha & Theta (2)
    f_idx = region_idx.get("frontal", [])
    p_idx = region_idx.get("parietal", [])
    if f_idx and p_idx:
        alpha_sig = bandpass(epoch_data, 8, 12, sfreq)
        theta_sig = bandpass(epoch_data, 4, 8,  sfreq)
        feat["PLV_FP_Alpha"] = compute_plv(alpha_sig[f_idx], alpha_sig[p_idx])
        feat["PLV_FP_Theta"] = compute_plv(theta_sig[f_idx], theta_sig[p_idx])
    else:
        feat["PLV_FP_Alpha"] = float("nan")
        feat["PLV_FP_Theta"] = float("nan")

    return feat


# ── Feature group registry ─────────────────────────────────────────────────────

def get_feature_groups(all_feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Map each feature name to its neuroscience family.

    Parameters
    ----------
    all_feature_cols : list[str]
        All feature column names (excluding Subject/Task/label/split/etc.).

    Returns
    -------
    dict mapping group_name → list[feature_name]
    """
    regions = ["frontal", "central", "temporal", "parietal", "occipital"]
    bands   = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    groups: dict[str, list[str]] = {
        "regional_band_power": [],
        "global_band_power":   [],
        "band_ratios":         [],
        "faa":                 [],
        "fz_theta":            [],
        "hjorth":              [],
        "perm_entropy":        [],
        "connectivity_plv":    [],
    }
    for col in all_feature_cols:
        if any(col.startswith(r + "_") for r in regions):
            groups["regional_band_power"].append(col)
        elif col.startswith("global_"):
            groups["global_band_power"].append(col)
        elif col.startswith("ratio_"):
            groups["band_ratios"].append(col)
        elif col == "FAA":
            groups["faa"].append(col)
        elif col == "Fz_Theta":
            groups["fz_theta"].append(col)
        elif col.startswith("Hjorth_"):
            groups["hjorth"].append(col)
        elif col == "PermEntropy":
            groups["perm_entropy"].append(col)
        elif col.startswith("PLV_"):
            groups["connectivity_plv"].append(col)
        else:
            # Catch-all for any unexpected columns
            groups.setdefault("other", []).append(col)
    return {k: v for k, v in groups.items() if v}
