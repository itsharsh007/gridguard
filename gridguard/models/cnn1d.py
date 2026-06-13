"""1D-CNN over the consumption sequence.

Stacked dilated 1D convolutions act as learnable multi-scale pattern detectors:
short kernels catch abrupt drops/zeroing, dilation widens the receptive field to
weeks/months without inflating parameters. Global average + max pooling collapse
the time axis to a fixed vector before a small MLP head emits a single logit.
Deliberately compact (~50k params) to train in seconds per epoch on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 32,
                 dropout: float = 0.2):
        super().__init__()

        def block(cin, cout, dilation):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=3, padding=dilation,
                          dilation=dilation),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, channels, 1),
            block(channels, channels, 2),
            block(channels, channels * 2, 4),
            block(channels * 2, channels * 2, 8),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels * 2),  # avg+max concat -> *4
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.features(x)
        avg = h.mean(dim=2)
        mx = h.max(dim=2).values
        return self.head(torch.cat([avg, mx], dim=1)).squeeze(-1)
