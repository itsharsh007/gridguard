"""Fast smoke tests — exercise the whole pipeline on a tiny synthetic dataset.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from gridguard.config import Config
from gridguard.data import generate_sgcc_like, clean_consumption
from gridguard.data.loader import load_wide_dataframe
from gridguard.features import build_features, to_sequences, FEATURE_GROUPS
from gridguard.evaluation import evaluate_scores, top_k_precision
from gridguard.pipeline import run_pipeline


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.synthetic.n_users = 400
    cfg.synthetic.n_days = 140
    cfg.model.nn_epochs = 2
    cfg.model.test_size = 0.25
    cfg.model.val_size = 0.15
    cfg.eval.top_k = 20
    return cfg


def test_synthetic_shape_and_balance():
    cfg = _tiny_cfg()
    df = generate_sgcc_like(cfg.synthetic)
    assert df.shape[0] == 400
    assert "FLAG" in df.columns
    rate = df["FLAG"].mean()
    assert 0.03 < rate < 0.18


def test_clean_removes_nans_and_caps():
    cfg = _tiny_cfg()
    wide = generate_sgcc_like(cfg.synthetic).reset_index()
    consumption, labels = load_wide_dataframe(wide)
    assert consumption.isna().any().any()           # raw has gaps
    cleaned = clean_consumption(consumption, cfg.clean)
    assert not cleaned.isna().any().any()           # all filled
    assert (cleaned >= 0).all().all()               # no negatives


def test_features_complete_and_finite():
    cfg = _tiny_cfg()
    wide = generate_sgcc_like(cfg.synthetic).reset_index()
    consumption, _ = load_wide_dataframe(wide)
    cleaned = clean_consumption(consumption, cfg.clean)
    feats = build_features(cleaned, cfg.features)
    expected = [c for grp in FEATURE_GROUPS.values() for c in grp]
    assert list(feats.columns) == expected
    assert np.isfinite(feats.to_numpy()).all()


def test_sequences_normalised():
    cfg = _tiny_cfg()
    wide = generate_sgcc_like(cfg.synthetic).reset_index()
    consumption, _ = load_wide_dataframe(wide)
    cleaned = clean_consumption(consumption, cfg.clean)
    seqs = to_sequences(cleaned, weekly=True)
    assert seqs.shape[0] == cleaned.shape[0]
    assert seqs.max() <= 1.0 + 1e-6
    assert seqs.min() >= 0.0


def test_topk_precision_logic():
    y = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    # top-2 are scores 0.9 (y=0) and 0.8 (y=1) -> precision 0.5
    assert abs(top_k_precision(y, scores, 2) - 0.5) < 1e-9


def test_end_to_end_pipeline_beats_random():
    cfg = _tiny_cfg()
    art = run_pipeline(cfg, prefer_synthetic=True, train_neural=True, verbose=False)
    assert "LightGBM" in art.metrics
    m = art.metrics["LightGBM"]
    # PR-AUC must beat the positive base-rate (what random scoring achieves).
    base_rate = art.class_balance["positive_rate"]
    assert m["pr_auc"] > base_rate
    assert m["roc_auc"] > 0.6
