"""Load the SGCC dataset (real if present, otherwise synthetic) into a tidy form.

Public contract used everywhere else in the codebase:

    consumption : DataFrame, index = consumer id, columns = DatetimeIndex of days,
                  values = daily kWh (may contain NaN).
    labels      : Series aligned to ``consumption.index``, 1 = theft / 0 = normal.

The real dataset ships as a wide CSV with a ``CONS_NO`` id column, a ``FLAG``
label column, and the remaining columns being dates. ``load_wide_csv`` accepts
exactly that layout (and the synthetic generator emits it), so an uploaded file
in the Streamlit app flows through the identical path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from ..config import Config, DEFAULT, RAW_DIR
from .synthetic import generate_sgcc_like

ID_COL = "CONS_NO"
LABEL_COL = "FLAG"

# Filenames we will look for under data/raw/ before falling back to synthetic.
_KNOWN_RAW_NAMES = ("sgcc.csv", "SGCC.csv", "data.csv", "electricity_theft.csv")


def _split_wide(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Split a wide table into (consumption matrix, labels-or-None)."""
    df = df.copy()
    if ID_COL in df.columns:
        df = df.set_index(ID_COL)
    df.index.name = ID_COL

    labels = None
    if LABEL_COL in df.columns:
        labels = df[LABEL_COL].astype(int)
        df = df.drop(columns=[LABEL_COL])

    # Coerce remaining columns to dates; keep the ones that parse.
    parsed = pd.to_datetime(df.columns, errors="coerce")
    keep = ~parsed.isna()
    df = df.loc[:, keep]
    df.columns = pd.DatetimeIndex(parsed[keep])
    df = df.sort_index(axis=1)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df, labels


def load_wide_csv(path: str | Path) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load a wide SGCC-style CSV from disk."""
    raw = pd.read_csv(path)
    return _split_wide(raw)


def load_wide_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Same as :func:`load_wide_csv` but for an in-memory DataFrame (Streamlit upload)."""
    return _split_wide(df)


def _find_raw_file() -> Optional[Path]:
    for name in _KNOWN_RAW_NAMES:
        p = RAW_DIR / name
        if p.exists():
            return p
    # Otherwise take the first CSV in data/raw/, if any.
    csvs = sorted(RAW_DIR.glob("*.csv"))
    return csvs[0] if csvs else None


def load_dataset(cfg: Config = DEFAULT,
                 prefer_synthetic: bool = False
                 ) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (consumption, labels), using a real CSV in data/raw/ if available."""
    raw_file = None if prefer_synthetic else _find_raw_file()
    if raw_file is not None:
        consumption, labels = load_wide_csv(raw_file)
        if labels is None:
            raise ValueError(
                f"{raw_file} has no '{LABEL_COL}' column; cannot train without labels."
            )
        return consumption, labels

    wide = generate_sgcc_like(cfg.synthetic)
    return load_wide_dataframe(wide.reset_index())  # type: ignore[return-value]
