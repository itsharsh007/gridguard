"""Focal loss for the neural models (Lin et al., 2017).

Focal loss down-weights easy negatives so the ~9:1 majority of honest meters does
not swamp the gradient. For logits ``z`` and target ``y``:

    p_t  = sigmoid(z) if y==1 else 1-sigmoid(z)
    FL   = -alpha_t * (1 - p_t)^gamma * log(p_t)

``gamma`` controls the focusing strength (0 -> ordinary weighted BCE); ``alpha``
balances the positive class. We use the numerically-stable BCE-with-logits form.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
