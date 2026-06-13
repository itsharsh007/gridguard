"""Stage 5 — evaluation, framed around how the utility actually operates.

Inspectors are the bottleneck: a crew can physically visit only ~``top_k`` meters
a month. So accuracy / ROC-AUC are the wrong lens — what matters is *precision in
the top-k ranked suspects* and the *revenue that ranking recovers*. We report:

  PR-AUC (average precision) — threshold-free quality on the rare class
  ROC-AUC                    — reported for comparability with the literature
  Precision@k / Recall@k     — hit-rate and coverage within the inspection budget
  Lift@k                     — how much better than random the budget spends
  Revenue recovered@k        — money model on the confirmed catches
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import EvalConfig


def _topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    # argpartition for the top-k, then order them by score descending.
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def top_k_precision(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = _topk_idx(np.asarray(scores), k)
    return float(np.mean(np.asarray(y_true)[idx])) if len(idx) else 0.0


def top_k_recall(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true)
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0
    idx = _topk_idx(np.asarray(scores), k)
    return float(y_true[idx].sum() / total_pos)


def top_k_lift(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    base = float(np.mean(y_true))
    if base == 0:
        return 0.0
    return top_k_precision(y_true, scores, k) / base


def estimate_revenue_recovered(y_true: np.ndarray, scores: np.ndarray,
                               monthly_kwh: np.ndarray, k: int,
                               cfg: EvalConfig) -> float:
    """Money recovered by inspecting the top-k, counting only true positives.

    Per confirmed theft we recover the hidden portion of consumption, billed over
    the recovery window:

        recovered = under_report_fraction * monthly_kwh * tariff * recovery_months
    """
    idx = _topk_idx(np.asarray(scores), k)
    y_true = np.asarray(y_true)
    monthly_kwh = np.asarray(monthly_kwh, dtype=float)
    caught = idx[y_true[idx] == 1]
    if len(caught) == 0:
        return 0.0
    recovered = (cfg.avg_theft_underreport * monthly_kwh[caught]
                 * cfg.tariff_per_kwh * cfg.recovery_months)
    return float(recovered.sum())


def evaluate_scores(y_true: Sequence[int], scores: Sequence[float],
                    monthly_kwh: Sequence[float] | None = None,
                    cfg: EvalConfig | None = None) -> Dict[str, float]:
    """Compute the full metric bundle for one model's scores."""
    cfg = cfg or EvalConfig()
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    k = cfg.top_k

    out = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        f"precision@{k}": top_k_precision(y_true, scores, k),
        f"recall@{k}": top_k_recall(y_true, scores, k),
        f"lift@{k}": top_k_lift(y_true, scores, k),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    if monthly_kwh is not None:
        out[f"revenue@{k}"] = estimate_revenue_recovered(
            y_true, scores, np.asarray(monthly_kwh), k, cfg)
    return out


def metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Stack per-model metric dicts into one comparison table (models as rows)."""
    df = pd.DataFrame(results).T
    # Put the headline columns first when present.
    order = ["pr_auc", "roc_auc"]
    order += [c for c in df.columns if c.startswith("precision@")]
    order += [c for c in df.columns if c.startswith("recall@")]
    order += [c for c in df.columns if c.startswith("lift@")]
    order += [c for c in df.columns if c.startswith("revenue@")]
    order += [c for c in df.columns if c not in order]
    return df[[c for c in order if c in df.columns]]
