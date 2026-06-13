"""SGCC-like synthetic data generator.

The real SGCC release (State Grid Corp. of China, published with the *Wide & Deep
CNN for Electricity-Theft Detection* paper) is a wide table: one row per consumer
(``CONS_NO``), one column per calendar day of daily kWh, plus a binary ``FLAG``
(1 = theft). It is highly imbalanced (~8.5% theft) and riddled with missing days.

This module reproduces those statistical properties so the whole pipeline is
runnable without the multi-hundred-MB download. The injected theft behaviours
mirror the canonical taxonomy used in the literature:

    h1  byteft  : multiply a window by a constant < 1   (under-reporting)
    h2  cut     : zero out a contiguous window          (meter bypass)
    h3  mean    : replace by a flat low value           (fixed fake reading)
    h4  reverse : intermittent on/off (rapid zeroing)   (tampering)
    h5  ramp    : slow declining drift                  (gradual tamper)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SyntheticConfig


def _base_profile(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """A plausible honest daily-consumption curve: annual seasonality + weekly
    rhythm + household baseline + multiplicative noise, with sporadic missing days."""
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    baseline = rng.uniform(4.0, 25.0)                       # household size proxy
    season_amp = rng.uniform(0.15, 0.6) * baseline
    # Two seasonal humps (summer cooling + winter heating).
    season = season_amp * (np.cos(2 * np.pi * (doy - 200) / 365.0) * 0.5
                           + np.cos(4 * np.pi * doy / 365.0) * 0.25)
    weekend = np.where(dow >= 5, rng.uniform(1.05, 1.4), 1.0)

    series = (baseline + season) * weekend
    series *= rng.normal(1.0, 0.12, size=n)                 # day-to-day noise
    series = np.clip(series, 0.0, None)

    # Inject naturally-occurring missing readings (~3% of days).
    miss = rng.random(n) < 0.03
    series[miss] = np.nan
    return series


def _inject_theft(rng: np.random.Generator, series: np.ndarray) -> np.ndarray:
    """Corrupt an honest curve with one of the canonical theft patterns.

    Magnitudes are deliberately *subtle* (and windows can be partial) so the two
    classes overlap — a perfectly separable toy problem would make the metrics
    meaningless. Plus, honest meters get some of the same surface symptoms below.
    """
    s = series.copy()
    n = len(s)
    pattern = rng.integers(0, 5)
    # Theft starts somewhere in the first 75% and runs to (near) the end.
    start = rng.integers(int(0.05 * n), int(0.75 * n))
    end = rng.integers(start + int(0.10 * n), n)

    if pattern == 0:                       # h1 constant under-reporting
        factor = rng.uniform(0.45, 0.85)   # subtle: only 15-55% shaved off
        s[start:end] *= factor
    elif pattern == 1:                     # h2 meter bypass (near-zeros, not exact)
        s[start:end] = np.abs(rng.normal(0.0, 0.3, end - start))
    elif pattern == 2:                     # h3 flat fake reading
        s[start:end] = np.nanmean(series) * rng.uniform(0.25, 0.55)
    elif pattern == 3:                     # h4 intermittent tampering
        mask = rng.random(end - start) < rng.uniform(0.25, 0.6)
        window = s[start:end].copy()
        window[mask] *= rng.uniform(0.0, 0.3)
        s[start:end] = window
    else:                                  # h5 gradual declining drift
        ramp = np.linspace(1.0, rng.uniform(0.3, 0.7), end - start)
        s[start:end] *= ramp
    return s


def _inject_honest_anomaly(rng: np.random.Generator, series: np.ndarray) -> np.ndarray:
    """Legitimate confounders that *look* like theft to a naive detector:
    a holiday/vacancy gap (zeros), a relocation level-shift, or a new appliance."""
    s = series.copy()
    n = len(s)
    kind = rng.integers(0, 3)
    if kind == 0:                          # vacation / vacancy: a real zero gap
        start = rng.integers(int(0.1 * n), int(0.85 * n))
        length = rng.integers(7, max(8, int(0.12 * n)))
        s[start:start + length] = np.abs(rng.normal(0.0, 0.1, len(s[start:start + length])))
    elif kind == 1:                        # relocation / occupancy change: step shift
        bp = rng.integers(int(0.2 * n), int(0.8 * n))
        s[bp:] *= rng.uniform(0.5, 0.8)
    else:                                  # efficiency upgrade: mild gradual decline
        ramp = np.linspace(1.0, rng.uniform(0.7, 0.9), n)
        s *= ramp
    return s


def generate_sgcc_like(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    """Return a wide SGCC-style DataFrame: index ``CONS_NO``, one column per day,
    plus a trailing ``FLAG`` column. Missing days are left as NaN on purpose."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")
    date_cols = [d.strftime("%Y-%m-%d") for d in dates]

    n_theft = int(round(cfg.n_users * cfg.theft_rate))
    labels = np.zeros(cfg.n_users, dtype=int)
    labels[:n_theft] = 1
    rng.shuffle(labels)

    rows = np.empty((cfg.n_users, cfg.n_days), dtype=float)
    for i in range(cfg.n_users):
        profile = _base_profile(rng, dates)
        if labels[i] == 1:
            profile = _inject_theft(rng, profile)
        elif rng.random() < 0.18:
            # ~18% of honest meters carry a legitimate anomaly (vacation, move,
            # upgrade) — the overlap that makes top-k precision a real challenge.
            profile = _inject_honest_anomaly(rng, profile)
        rows[i] = profile

    df = pd.DataFrame(rows, columns=date_cols)
    df.insert(0, "CONS_NO", [f"U{idx:06d}" for idx in range(cfg.n_users)])
    df["FLAG"] = labels
    return df.set_index("CONS_NO")
