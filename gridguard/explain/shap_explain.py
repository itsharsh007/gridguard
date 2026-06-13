"""Stage 6 — per-meter SHAP explanations.

TreeSHAP on the LightGBM model gives an exact, additive attribution: for each
flagged meter, score = base_value + sum(feature contributions). The inspector sees
*which* behaviours pushed this meter up the suspect list (e.g. "weekend/weekday
ratio collapsed", "42-day zero-run", "low autocorrelation"). Plots are rendered
with matplotlib (no JS dependency) so they embed cleanly in Streamlit and reports.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


# Human-readable phrasing for the engineered feature names.
_FRIENDLY = {
    "mean": "average daily use",
    "median": "median daily use",
    "std": "use variability",
    "min": "minimum day",
    "max": "peak day",
    "total": "total consumption",
    "roll7_mean": "weekly-smoothed level",
    "roll7_std": "weekly-smoothed variability",
    "roll30_mean": "monthly-smoothed level",
    "roll30_std": "monthly-smoothed variability",
    "trend_ratio": "end-vs-start trend",
    "weekend_weekday_ratio": "weekend/weekday ratio",
    "weekday_cv": "weekday variability",
    "cv": "overall volatility",
    "diff_std": "day-to-day jumpiness",
    "zero_fraction": "fraction of zero days",
    "low_fraction": "fraction of near-zero days",
    "sudden_drop_count": "number of sudden drops",
    "max_rel_drop": "largest single-day drop",
    "longest_zero_run": "longest zero-reading streak",
    "post_pre_drop_ratio": "level after vs before biggest drop",
    "autocorr_lag1": "day-to-day regularity",
    "autocorr_lag7": "weekly regularity",
}


def friendly(name: str) -> str:
    return _FRIENDLY.get(name, name)


class ShapExplainer:
    """Wraps a TreeExplainer over a fitted :class:`LGBMTheftModel`."""

    def __init__(self, lgbm_model):
        self.model = lgbm_model
        self.explainer = shap.TreeExplainer(lgbm_model.model)
        self.feature_names = lgbm_model.feature_names_

    def _shap_for(self, X: pd.DataFrame) -> np.ndarray:
        vals = self.explainer.shap_values(X)
        # Older SHAP returns a list [neg, pos] for binary; take the positive class.
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:          # (n, features, classes)
            vals = vals[:, :, -1]
        return vals

    @property
    def base_value(self) -> float:
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.asarray(ev).ravel()[-1]
        return float(ev)

    def explain_row(self, x_row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
        """Return the top contributing features for a single meter (one-row X)."""
        vals = self._shap_for(x_row)[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "friendly": [friendly(f) for f in self.feature_names],
            "value": x_row.iloc[0].to_numpy(),
            "shap": vals,
        })
        df["abs"] = df["shap"].abs()
        return df.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")

    def plot_row(self, x_row: pd.DataFrame, meter_id: str = "", top_n: int = 6):
        """Horizontal bar of the signed contributions for one meter."""
        top = self.explain_row(x_row, top_n).iloc[::-1]
        colors = ["#c0392b" if s > 0 else "#2980b9" for s in top["shap"]]
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(top) + 1.2))
        ax.barh(top["friendly"], top["shap"], color=colors)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_xlabel("← lowers suspicion        SHAP contribution        raises suspicion →")
        title = "Why this meter was flagged"
        if meter_id:
            title += f"  ·  {meter_id}"
        ax.set_title(title, fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        return fig

    def global_importance_plot(self, X_sample: pd.DataFrame, max_display: int = 12):
        """Mean |SHAP| bar across a sample — the model's global drivers."""
        vals = self._shap_for(X_sample)
        mean_abs = pd.Series(np.abs(vals).mean(axis=0),
                             index=self.feature_names).sort_values()
        mean_abs.index = [friendly(f) for f in mean_abs.index]
        mean_abs = mean_abs.tail(max_display)
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(mean_abs) + 1))
        ax.barh(mean_abs.index, mean_abs.values, color="#34495e")
        ax.set_xlabel("mean |SHAP| (average impact on suspicion score)")
        ax.set_title("What drives GridGuard's theft scores", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        return fig
