"""LightGBM theft model — the workhorse and the explainable one.

Gradient-boosted trees on the engineered tabular features are the strongest
baseline for SGCC-style data and, crucially, support exact TreeSHAP so every
flagged meter gets a per-feature reason an inspector can read. Imbalance is
handled by either ``scale_pos_weight`` (default, in-loss reweighting) or by SMOTE
oversampling of the training fold (set ``ModelConfig.imbalance = "smote"``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..config import ModelConfig
from ..imbalance.sampling import apply_smote


class LGBMTheftModel:
    def __init__(self, cfg: ModelConfig, name: str = "lightgbm"):
        self.cfg = cfg
        self.name = name
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: Optional[list] = None

    def _params(self, y_train: np.ndarray) -> dict:
        pos = max(1, int(np.sum(y_train)))
        neg = max(1, int(len(y_train) - pos))
        params = dict(
            n_estimators=self.cfg.lgbm_n_estimators,
            learning_rate=self.cfg.lgbm_learning_rate,
            num_leaves=self.cfg.lgbm_num_leaves,
            max_depth=self.cfg.lgbm_max_depth,
            subsample=self.cfg.lgbm_subsample,
            subsample_freq=1,
            colsample_bytree=self.cfg.lgbm_colsample,
            objective="binary",
            random_state=self.cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        # Reweight in-loss unless we are oversampling instead.
        if self.cfg.imbalance in ("scale_pos_weight", "focal", "none"):
            params["scale_pos_weight"] = neg / pos
        return params

    def fit(self, X_train: pd.DataFrame, y_train, X_val=None, y_val=None,
            verbose: bool = False):
        self.feature_names_ = list(X_train.columns)
        y_train = np.asarray(y_train)

        if self.cfg.imbalance == "smote":
            X_train, y_train = apply_smote(
                X_train, y_train, self.cfg.smote_k_neighbors, self.cfg.random_state)

        self.model = lgb.LGBMClassifier(**self._params(y_train))
        fit_kw = {}
        if X_val is not None and y_val is not None:
            fit_kw.update(
                eval_set=[(X_val, np.asarray(y_val))],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, verbose=verbose),
                           lgb.log_evaluation(0)],
            )
        self.model.fit(X_train, y_train, **fit_kw)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "model not fitted"
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        assert self.model is not None
        return pd.Series(self.model.feature_importances_,
                         index=self.feature_names_).sort_values(ascending=False)
