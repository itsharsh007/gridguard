"""Inference helper shared by the dashboard and the evaluate script.

Takes a saved LightGBM bundle and a raw wide consumption table (uploaded CSV or a
DataFrame), runs the identical clean -> feature path used in training, and returns
a ranked suspect list with the per-meter revenue-at-risk estimate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, DEFAULT
from .data import clean_consumption
from .data.loader import load_wide_dataframe, load_wide_csv
from .features import build_features, to_sequences


def _monthly_kwh(consumption: pd.DataFrame) -> np.ndarray:
    return consumption.mean(axis=1).to_numpy() * 30.0


def score_consumption(consumption: pd.DataFrame, bundle: dict,
                      cfg: Optional[Config] = None) -> pd.DataFrame:
    """Score a cleaned-or-raw wide consumption matrix; return a ranked DataFrame."""
    cfg = cfg or bundle.get("config", DEFAULT)
    cleaned = clean_consumption(consumption, cfg.clean)
    feats = build_features(cleaned, cfg.features)
    feats = feats[bundle["feature_names"]]      # enforce training column order

    lgbm = bundle["lgbm"]
    scores = lgbm.predict_proba(feats)
    monthly = _monthly_kwh(cleaned)

    risk = (cfg.eval.avg_theft_underreport * monthly
            * cfg.eval.tariff_per_kwh * cfg.eval.recovery_months)

    ranked = pd.DataFrame({
        "meter_id": cleaned.index,
        "theft_score": scores,
        "monthly_kwh": monthly,
        "revenue_at_risk": risk,
    }).sort_values("theft_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked, feats, cleaned


def score_all_models(consumption: pd.DataFrame, bundle: dict,
                     cfg: Optional[Config] = None) -> pd.DataFrame:
    """Score the consumption matrix with every model in the bundle.

    Returns a DataFrame indexed by meter id with one ``theft_score`` column per
    model (LightGBM always; the 1D-CNN / LSTM-Attention if they were persisted).
    The neural nets use global pooling / variable-length recurrence, so they
    handle any sequence length — but scores are most comparable when the upload
    matches the trained day-count.
    """
    cfg = cfg or bundle.get("config", DEFAULT)
    cleaned = clean_consumption(consumption, cfg.clean)

    feats = build_features(cleaned, cfg.features)[bundle["feature_names"]]
    scores = {"LightGBM": bundle["lgbm"].predict_proba(feats)}

    torch_models = bundle.get("torch_models") or {}
    if torch_models:
        seqs = to_sequences(cleaned, weekly=cfg.model.nn_weekly_downsample)
        for name, model in torch_models.items():
            scores[name] = model.predict_proba(seqs)

    return pd.DataFrame(scores, index=cleaned.index)


def score_csv(path, bundle: dict, cfg: Optional[Config] = None):
    consumption, _ = load_wide_csv(path)
    return score_consumption(consumption, bundle, cfg)


def score_uploaded_dataframe(df: pd.DataFrame, bundle: dict,
                             cfg: Optional[Config] = None):
    consumption, _ = load_wide_dataframe(df)
    return score_consumption(consumption, bundle, cfg)
