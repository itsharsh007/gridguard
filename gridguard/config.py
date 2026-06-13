"""Central configuration for GridGuard.

All paths are resolved relative to the repository root so the project runs the
same way from a script, a notebook, or the Streamlit app. Values are plain
dataclasses with sane CPU-friendly defaults; override them in code or via
``Config.from_yaml`` if you keep a ``config.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Repository root = parent of the package directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class SyntheticConfig:
    """Parameters for the SGCC-like synthetic generator (used when the real
    dataset is not present)."""
    n_users: int = 4000
    n_days: int = 730              # ~2 years of daily readings
    theft_rate: float = 0.085      # SGCC real prevalence is ~8.5%
    start_date: str = "2014-01-01"
    seed: int = 42


@dataclass
class CleanConfig:
    interpolate_limit: int = 7     # max consecutive NaNs to bridge by interpolation
    outlier_method: str = "iqr"    # {"iqr", "quantile", "none"}
    iqr_k: float = 3.0             # cap beyond Q3 + k*IQR (and below Q1 - k*IQR)
    quantile_cap: float = 0.999    # used when outlier_method == "quantile"


@dataclass
class FeatureConfig:
    rolling_windows: tuple = (7, 30)
    drop_threshold: float = 0.5    # day-over-day relative fall counted as a "sudden drop"
    autocorr_lags: tuple = (1, 7)


@dataclass
class ModelConfig:
    # Shared
    val_size: float = 0.15
    test_size: float = 0.20
    random_state: int = 42

    # Imbalance
    imbalance: str = "focal"       # {"none", "smote", "scale_pos_weight", "focal"}
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    smote_k_neighbors: int = 5

    # LightGBM
    lgbm_n_estimators: int = 600
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 63
    lgbm_max_depth: int = -1
    lgbm_subsample: float = 0.8
    lgbm_colsample: float = 0.8

    # Neural nets (kept small to stay comfortable on CPU)
    nn_epochs: int = 12
    nn_batch_size: int = 256
    nn_lr: float = 1e-3
    nn_weight_decay: float = 1e-5
    nn_weekly_downsample: bool = True   # aggregate daily -> weekly for the RNN/CNN
    nn_patience: int = 4                # early-stopping patience on val PR-AUC


@dataclass
class EvalConfig:
    top_k: int = 200               # "inspector can visit 200 meters/month"
    tariff_per_kwh: float = 0.12   # currency per kWh, for the revenue estimate
    recovery_months: int = 12      # months of recovered billing per confirmed theft
    avg_theft_underreport: float = 0.6  # fraction of true consumption that was hidden


@dataclass
class Config:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls()
        for section, values in raw.items():
            if hasattr(cfg, section) and isinstance(values, dict):
                sub = getattr(cfg, section)
                for k, v in values.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
