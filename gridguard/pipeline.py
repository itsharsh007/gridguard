"""End-to-end orchestration: data -> features -> models -> evaluation -> artifacts.

This is the single entry point used by ``scripts/train.py`` and reused (in part)
by the Streamlit app. It keeps the tabular and sequence representations aligned to
the *same* train/val/test consumer split so the three models are compared fairly,
and persists everything the dashboard needs to score new uploads.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, DEFAULT, MODELS_DIR
from .data import load_dataset, clean_consumption
from .features import build_features, to_sequences
from .imbalance.sampling import class_balance
from .models import LGBMTheftModel, CNN1D, LSTMAttention, TorchSequenceModel
from .evaluation import evaluate_scores, metrics_table


@dataclass
class PipelineArtifacts:
    config: Config
    lgbm: LGBMTheftModel
    feature_names: list
    metrics: Dict[str, Dict[str, float]]
    class_balance: dict
    torch_models: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def _monthly_kwh(consumption: pd.DataFrame) -> np.ndarray:
    """Approx. mean monthly kWh per meter (daily mean * 30), for the revenue model."""
    return consumption.mean(axis=1).to_numpy() * 30.0


def _split_indices(n: int, y: np.ndarray, cfg: Config):
    idx = np.arange(n)
    test_size = cfg.model.test_size
    val_size = cfg.model.val_size
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=cfg.model.random_state)
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=rel_val, stratify=y[train_idx],
        random_state=cfg.model.random_state)
    return train_idx, val_idx, test_idx


def run_pipeline(cfg: Config = DEFAULT,
                 prefer_synthetic: bool = False,
                 train_neural: bool = True,
                 verbose: bool = True) -> PipelineArtifacts:
    t0 = time.time()

    # ---- 1. load + clean -------------------------------------------------
    if verbose:
        print("[1/5] loading + cleaning ...")
    consumption, labels = load_dataset(cfg, prefer_synthetic=prefer_synthetic)
    consumption = clean_consumption(consumption, cfg.clean)
    y = labels.to_numpy().astype(int)
    bal = class_balance(y)
    if verbose:
        print(f"      {bal['n']} meters, {bal['positive']} theft "
              f"({bal['positive_rate']:.1%}), imbalance 1:{bal['imbalance_ratio']:.1f}")

    # ---- 2. features + sequences ----------------------------------------
    if verbose:
        print("[2/5] engineering features ...")
    feats = build_features(consumption, cfg.features)
    seqs = to_sequences(consumption, weekly=cfg.model.nn_weekly_downsample)
    monthly = _monthly_kwh(consumption)

    # ---- 3. aligned split ------------------------------------------------
    tr, va, te = _split_indices(len(y), y, cfg)
    Xtr, Xva, Xte = feats.iloc[tr], feats.iloc[va], feats.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    Str, Sva, Ste = seqs[tr], seqs[va], seqs[te]
    monthly_te = monthly[te]

    # The inspection budget (top_k) is defined against the *full* monthly meter
    # population. The held-out fold is a fraction of it, so evaluate the fold at
    # the same inspection *rate* — otherwise precision@k is capped by how few
    # thefts land in the fold rather than by model quality.
    fold_eval = copy.deepcopy(cfg.eval)
    fold_eval.top_k = max(10, round(cfg.eval.top_k * len(te) / len(y)))

    results: Dict[str, Dict[str, float]] = {}
    test_scores: Dict[str, np.ndarray] = {}

    # ---- 4. models -------------------------------------------------------
    if verbose:
        print(f"[3/5] training LightGBM (imbalance={cfg.model.imbalance}) ...")
    lgbm = LGBMTheftModel(cfg.model).fit(Xtr, ytr, Xva, yva)
    s = lgbm.predict_proba(Xte)
    test_scores["LightGBM"] = s
    results["LightGBM"] = evaluate_scores(yte, s, monthly_te, fold_eval)

    torch_models = {}
    if train_neural:
        seq_len = seqs.shape[1]
        specs = {
            "1D-CNN": CNN1D(),
            "LSTM-Attention": LSTMAttention(),
        }
        for name, net in specs.items():
            if verbose:
                print(f"[4/5] training {name} (seq_len={seq_len}) ...")
            tm = TorchSequenceModel(net, cfg.model, name=name)
            tm.fit(Str, ytr, Sva, yva, verbose=verbose)
            s = tm.predict_proba(Ste)
            test_scores[name] = s
            results[name] = evaluate_scores(yte, s, monthly_te, fold_eval)
            torch_models[name] = tm

    # ---- 5. report -------------------------------------------------------
    if verbose:
        print("[5/5] evaluation (held-out test set):")
        print(metrics_table(results).round(4).to_string())
        print(f"      done in {time.time() - t0:.1f}s")

    return PipelineArtifacts(
        config=cfg,
        lgbm=lgbm,
        feature_names=list(feats.columns),
        metrics=results,
        class_balance=bal,
        torch_models=torch_models,
        extras={
            "test_index": consumption.index[te],
            "test_scores": test_scores,
            "test_labels": yte,
            "test_features": Xte,
            "test_consumption": consumption.iloc[te],
            "feature_sample": Xtr.sample(min(500, len(Xtr)),
                                         random_state=cfg.model.random_state),
        },
    )


def save_artifacts(artifacts: PipelineArtifacts, path: Path = MODELS_DIR) -> Path:
    """Persist what the dashboard needs: LightGBM model, config, metrics, sample."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "lgbm": artifacts.lgbm,
        "torch_models": artifacts.torch_models,   # {name: TorchSequenceModel}
        "feature_names": artifacts.feature_names,
        "config": artifacts.config,
        "metrics": artifacts.metrics,
        "class_balance": artifacts.class_balance,
        "feature_sample": artifacts.extras.get("feature_sample"),
    }
    out = path / "gridguard_lgbm.joblib"
    joblib.dump(bundle, out)
    metrics_table(artifacts.metrics).round(4).to_csv(path / "metrics.csv")
    return out


def load_artifacts(path: Path = MODELS_DIR) -> dict:
    bundle = joblib.load(Path(path) / "gridguard_lgbm.joblib")
    return bundle
