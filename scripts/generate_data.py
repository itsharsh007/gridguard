"""Generate an SGCC-like synthetic dataset and write it to data/raw/sgcc.csv.

Usage:
    python scripts/generate_data.py --users 4000 --days 730 --theft-rate 0.085
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridguard.config import SyntheticConfig, RAW_DIR
from gridguard.data.synthetic import generate_sgcc_like


def main():
    ap = argparse.ArgumentParser(description="Generate SGCC-like synthetic data")
    ap.add_argument("--users", type=int, default=4000)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--theft-rate", type=float, default=0.085)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(RAW_DIR / "sgcc.csv"))
    args = ap.parse_args()

    cfg = SyntheticConfig(n_users=args.users, n_days=args.days,
                          theft_rate=args.theft_rate, seed=args.seed)
    df = generate_sgcc_like(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Wrote {df.shape[0]} meters x {df.shape[1]-1} days -> {out}")
    print(f"Theft prevalence: {df['FLAG'].mean():.1%}")


if __name__ == "__main__":
    main()
