"""Score a dataset with the saved model and print the inspection-budget report.

Usage:
    python scripts/evaluate.py                       # score data/raw/sgcc.csv
    python scripts/evaluate.py --csv path/to/file.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from gridguard.config import RAW_DIR
from gridguard.pipeline import load_artifacts
from gridguard.data.loader import load_wide_csv
from gridguard.scoring import score_consumption
from gridguard.evaluation import evaluate_scores


def main():
    ap = argparse.ArgumentParser(description="Score data with the saved model")
    ap.add_argument("--csv", type=str, default=str(RAW_DIR / "sgcc.csv"))
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    bundle = load_artifacts()
    cfg = bundle["config"]
    if args.top_k:
        cfg.eval.top_k = args.top_k

    consumption, labels = load_wide_csv(args.csv)
    ranked, _, cleaned = score_consumption(consumption, bundle, cfg)

    k = cfg.eval.top_k
    print(f"\nTop {min(10, k)} suspects:")
    print(ranked.head(10).to_string(index=False))
    print(f"\nEstimated revenue at risk in top-{k}: "
          f"{ranked.head(k)['revenue_at_risk'].sum():,.0f}")

    if labels is not None:
        scores = ranked.sort_values("meter_id")["theft_score"].to_numpy()
        y = labels.reindex(ranked.sort_values("meter_id")["meter_id"]).to_numpy()
        m = evaluate_scores(y, scores,
                            ranked.sort_values("meter_id")["monthly_kwh"].to_numpy(),
                            cfg.eval)
        print("\nMetrics vs ground-truth labels:")
        for key, val in m.items():
            print(f"  {key:18s}: {val:,.4f}" if isinstance(val, float)
                  else f"  {key:18s}: {val}")


if __name__ == "__main__":
    main()
