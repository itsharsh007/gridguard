"""GridGuard — electricity theft detection on the SGCC daily-consumption dataset.

A CPU-only, end-to-end pipeline:
    data cleaning -> feature engineering -> imbalance handling ->
    LightGBM / 1D-CNN / LSTM-Attention -> PR-AUC & top-k evaluation -> SHAP.
"""

__version__ = "0.1.0"
