"""Stage 1 — data cleaning.

Two operations, applied per consumer (per row), in this order:

1. Missing-value interpolation. Gaps up to ``interpolate_limit`` consecutive days
   are bridged linearly (consumption is smooth at the daily scale). Longer gaps,
   and any leading/trailing NaNs, are filled with 0 — a deliberately conservative
   choice: a genuinely dark meter *should* read low, and theft detectors must not
   be fed imputed "normal" values for long dead stretches.

2. Outlier capping. Real meters spike from data-glitches and meter rollovers.
   We winsorise each consumer's own series (per-row IQR fence by default) so a
   single 10000 kWh glitch does not dominate the rolling/volatility features.
   Capping is per-consumer because absolute load varies by orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CleanConfig


def _interpolate_row(row: pd.Series, limit: int) -> pd.Series:
    out = row.interpolate(method="linear", limit=limit, limit_area="inside")
    return out.fillna(0.0)


def _cap_row_iqr(row: pd.Series, k: float) -> pd.Series:
    q1, q3 = row.quantile(0.25), row.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return row
    upper = q3 + k * iqr
    lower = max(0.0, q1 - k * iqr)
    return row.clip(lower=lower, upper=upper)


def clean_consumption(consumption: pd.DataFrame, cfg: CleanConfig | None = None
                      ) -> pd.DataFrame:
    """Return a cleaned copy of the wide consumption matrix (same shape/index)."""
    cfg = cfg or CleanConfig()
    df = consumption.astype(float)

    # 1) interpolation (vectorised along the day axis, row by row).
    df = df.apply(lambda r: _interpolate_row(r, cfg.interpolate_limit), axis=1)

    # 2) outlier capping.
    if cfg.outlier_method == "iqr":
        df = df.apply(lambda r: _cap_row_iqr(r, cfg.iqr_k), axis=1)
    elif cfg.outlier_method == "quantile":
        caps = df.quantile(cfg.quantile_cap, axis=1)
        df = df.clip(upper=caps, axis=0)
    elif cfg.outlier_method != "none":
        raise ValueError(f"unknown outlier_method: {cfg.outlier_method}")

    # Negative readings are physically impossible (meter errors) -> floor at 0.
    df = df.clip(lower=0.0)
    return df
