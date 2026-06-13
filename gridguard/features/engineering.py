"""Stage 2 — feature engineering.

Turns each consumer's cleaned daily series into a fixed-width feature vector for
the tree model, and into a normalised sequence for the neural models.

Feature families (the inspector-facing names are kept human-readable so they read
well in SHAP plots):

  level        — mean / median / std / min / max / total consumption
  rolling      — mean & std of rolling-window means (7d, 30d): smoothness & trend
  weekday      — weekend-to-weekday consumption ratio (tamperers often forget the
                 weekly rhythm), plus weekday coefficient of variation
  volatility   — coefficient of variation, std of day-over-day differences,
                 fraction of (near-)zero days
  drops        — sudden-drop count, largest single-day relative drop, longest
                 consecutive zero-run, post-vs-pre-drop level ratio
  autocorr     — lag-1 and lag-7 autocorrelation (loss of structure signals tamper)

All features are deterministic, NaN-safe, and computed without leakage (each
consumer is summarised independently of any label).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import FeatureConfig

FEATURE_GROUPS: Dict[str, List[str]] = {
    "level": ["mean", "median", "std", "min", "max", "total"],
    "rolling": ["roll7_mean", "roll7_std", "roll30_mean", "roll30_std",
                "trend_ratio"],
    "weekday": ["weekend_weekday_ratio", "weekday_cv"],
    "volatility": ["cv", "diff_std", "zero_fraction", "low_fraction"],
    "drops": ["sudden_drop_count", "max_rel_drop", "longest_zero_run",
              "post_pre_drop_ratio"],
    "autocorr": ["autocorr_lag1", "autocorr_lag7"],
}

_EPS = 1e-9


def _safe_autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < _EPS:
        return 0.0
    return float((a * b).sum() / denom)


def _longest_zero_run(x: np.ndarray, thresh: float) -> int:
    is_zero = x <= thresh
    best = cur = 0
    for z in is_zero:
        cur = cur + 1 if z else 0
        best = max(best, cur)
    return best


def _row_features(values: np.ndarray, dow: np.ndarray, cfg: FeatureConfig) -> dict:
    x = values.astype(float)
    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    median = float(np.median(x))
    total = float(np.sum(x))
    low_thresh = max(_EPS, 0.05 * (median if median > 0 else mean))

    # rolling
    s = pd.Series(x)
    roll7 = s.rolling(cfg.rolling_windows[0], min_periods=1).mean()
    roll30 = s.rolling(cfg.rolling_windows[1], min_periods=1).mean()
    first_q = x[: n // 4].mean() if n >= 4 else mean
    last_q = x[-n // 4:].mean() if n >= 4 else mean
    trend_ratio = float(last_q / (first_q + _EPS))

    # weekday / weekend
    wk = x[dow < 5]
    we = x[dow >= 5]
    wk_mean = wk.mean() if wk.size else mean
    we_mean = we.mean() if we.size else mean
    weekend_weekday_ratio = float(we_mean / (wk_mean + _EPS))
    weekday_cv = float(wk.std() / (wk.mean() + _EPS)) if wk.size else 0.0

    # volatility
    diffs = np.diff(x)
    cv = float(std / (mean + _EPS))
    diff_std = float(np.std(diffs)) if diffs.size else 0.0
    zero_fraction = float(np.mean(x <= _EPS))
    low_fraction = float(np.mean(x <= low_thresh))

    # sudden drops: day-over-day relative fall beyond the configured threshold
    prev = x[:-1]
    rel_drop = (prev - x[1:]) / (prev + _EPS)
    sudden_drop_count = int(np.sum(rel_drop >= cfg.drop_threshold))
    max_rel_drop = float(np.max(rel_drop)) if rel_drop.size else 0.0
    longest_zero_run = _longest_zero_run(x, low_thresh)
    # level before vs after the single biggest drop (captures step-changes)
    if rel_drop.size and max_rel_drop > 0:
        bp = int(np.argmax(rel_drop)) + 1
        pre = x[:bp].mean() if bp > 0 else mean
        post = x[bp:].mean() if bp < n else mean
        post_pre_drop_ratio = float(post / (pre + _EPS))
    else:
        post_pre_drop_ratio = 1.0

    return {
        "mean": mean, "median": median, "std": std,
        "min": float(np.min(x)), "max": float(np.max(x)), "total": total,
        "roll7_mean": float(roll7.mean()), "roll7_std": float(roll7.std()),
        "roll30_mean": float(roll30.mean()), "roll30_std": float(roll30.std()),
        "trend_ratio": trend_ratio,
        "weekend_weekday_ratio": weekend_weekday_ratio, "weekday_cv": weekday_cv,
        "cv": cv, "diff_std": diff_std,
        "zero_fraction": zero_fraction, "low_fraction": low_fraction,
        "sudden_drop_count": sudden_drop_count, "max_rel_drop": max_rel_drop,
        "longest_zero_run": longest_zero_run,
        "post_pre_drop_ratio": post_pre_drop_ratio,
        "autocorr_lag1": _safe_autocorr(x, cfg.autocorr_lags[0]),
        "autocorr_lag7": _safe_autocorr(x, cfg.autocorr_lags[1]),
    }


def build_features(consumption: pd.DataFrame, cfg: FeatureConfig | None = None
                   ) -> pd.DataFrame:
    """Return a (n_users x n_features) DataFrame indexed like ``consumption``."""
    cfg = cfg or FeatureConfig()
    dow = consumption.columns.dayofweek.to_numpy()
    records = {
        idx: _row_features(row, dow, cfg)
        for idx, row in zip(consumption.index, consumption.to_numpy())
    }
    feats = pd.DataFrame.from_dict(records, orient="index")
    feats.index.name = consumption.index.name
    ordered = [c for grp in FEATURE_GROUPS.values() for c in grp]
    return feats[ordered]


def to_sequences(consumption: pd.DataFrame, weekly: bool = True
                 ) -> np.ndarray:
    """Return a per-user normalised sequence array for the neural models.

    Each row is min-max scaled by its own max (robust to absolute-load differences,
    and the scale itself is already captured by the tabular ``level`` features).
    With ``weekly=True`` the daily series is aggregated to weekly means, cutting the
    sequence length ~7x so the LSTM/CNN train comfortably on CPU.
    """
    arr = consumption.to_numpy(dtype=float)
    if weekly:
        n_days = arr.shape[1]
        n_weeks = n_days // 7
        if n_weeks >= 1:
            arr = arr[:, : n_weeks * 7].reshape(arr.shape[0], n_weeks, 7).mean(axis=2)
    maxes = arr.max(axis=1, keepdims=True)
    maxes[maxes < _EPS] = 1.0
    return (arr / maxes).astype(np.float32)
