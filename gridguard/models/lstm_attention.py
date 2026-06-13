"""Bi-LSTM with additive (Bahdanau-style) temporal attention.

The Bi-LSTM reads the consumption sequence in both directions; an attention layer
then learns a soft weighting over time steps and produces a context vector that
emphasises the suspicious stretch (e.g. the weeks where the meter went dark). The
attention weights are returned so the dashboard can show *when* a meter looked
anomalous, complementing the SHAP "why" on the tabular side.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LSTMAttention(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 48,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(feat, feat // 2),
            nn.Tanh(),
            nn.Linear(feat // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, feat // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False
                ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T) -> (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h, _ = self.lstm(x)                      # (B, T, 2H)
        scores = self.attn(h).squeeze(-1)        # (B, T)
        weights = torch.softmax(scores, dim=1)   # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, 2H)
        logit = self.head(context).squeeze(-1)
        if return_attn:
            return logit, weights
        return logit
