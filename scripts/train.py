"""Train + compare all three models, then persist artifacts for the dashboard.

Usage:
    python scripts/train.py                       # full run (LGBM + CNN + LSTM)
    python scripts/train.py --no-neural           # LightGBM only (fastest)
    python scripts/train.py --imbalance smote      # SMOTE instead of focal/weights
    python scripts/train.py --compare-imbalance    # focal vs smote vs none table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from gridguard.config import Config
from gridguard.pipeline import run_pipeline, save_artifacts
from gridguard.evaluation import metrics_table


def main():
    ap = argparse.ArgumentParser(description="Train GridGuard models")
    ap.add_argument("--no-neural", action="store_true",
                    help="train LightGBM only (skip CNN/LSTM)")
    ap.add_argument("--imbalance", default=None,
                    choices=["none", "smote", "scale_pos_weight", "focal"])
    ap.add_argument("--users", type=int, default=None,
                    help="override synthetic user count (if no real data present)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config", type=str, default=None, help="path to config.yaml")
    ap.add_argument("--compare-imbalance", action="store_true",
                    help="run LightGBM under several imbalance strategies")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.imbalance:
        cfg.model.imbalance = args.imbalance
    if args.users:
        cfg.synthetic.n_users = args.users
    if args.epochs:
        cfg.model.nn_epochs = args.epochs

    if args.compare_imbalance:
        rows = {}
        for strat in ["none", "scale_pos_weight", "focal", "smote"]:
            cfg.model.imbalance = strat
            art = run_pipeline(cfg, train_neural=False, verbose=False)
            rows[f"LightGBM/{strat}"] = art.metrics["LightGBM"]
        print("\nImbalance-strategy comparison (LightGBM, held-out test):")
        print(metrics_table(rows).round(4).to_string())
        return

    artifacts = run_pipeline(cfg, train_neural=not args.no_neural, verbose=True)
    out = save_artifacts(artifacts)
    print(f"\nSaved model bundle -> {out}")
    print("Metrics CSV       -> models/metrics.csv")


if __name__ == "__main__":
    main()
