"""SMOTE oversampling for the tabular (LightGBM) path.

SMOTE synthesises minority-class feature vectors by interpolating between a theft
sample and its nearest theft neighbours. It is applied to the **training fold
only** — never to validation/test — to avoid leaking synthetic neighbours across
the split. The neural and ``scale_pos_weight`` paths get their imbalance handling
from focal loss / class weights instead; this module exists so the README's
"focal loss vs SMOTE comparison" can be run head-to-head.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def class_balance(y: np.ndarray) -> dict:
    y = np.asarray(y)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return {
        "n": len(y), "positive": pos, "negative": neg,
        "positive_rate": pos / max(1, len(y)),
        "imbalance_ratio": neg / max(1, pos),
    }


def apply_smote(X: pd.DataFrame, y: np.ndarray, k_neighbors: int = 5,
                random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return an oversampled (X, y). Falls back gracefully if a class is tiny."""
    from imblearn.over_sampling import SMOTE

    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos <= 1:
        return X, y
    k = min(k_neighbors, n_pos - 1)
    if k < 1:
        return X, y

    sm = SMOTE(k_neighbors=k, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    return X_res, np.asarray(y_res)
