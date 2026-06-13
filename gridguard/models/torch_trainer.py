"""A thin, CPU-friendly training wrapper shared by the CNN and LSTM models.

Handles the boilerplate uniformly so both nets are trained and compared on equal
footing: focal loss (or weighted BCE), Adam, early stopping on validation PR-AUC,
and best-weights restore. Everything is deterministic given a seed and never
touches the GPU.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from ..config import ModelConfig
from ..imbalance.focal_loss import BinaryFocalLoss


class TorchSequenceModel:
    """Fit/predict wrapper around a torch ``nn.Module`` that maps sequence -> logit."""

    def __init__(self, network: torch.nn.Module, cfg: ModelConfig,
                 name: str = "torch"):
        torch.manual_seed(cfg.random_state)
        self.net = network
        self.cfg = cfg
        self.name = name
        self.device = torch.device("cpu")
        self.net.to(self.device)

    def _loader(self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if y is None:
            ds = TensorDataset(Xt)
        else:
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
            ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=self.cfg.nn_batch_size, shuffle=shuffle)

    def _make_loss(self, y_train: np.ndarray):
        if self.cfg.imbalance == "focal":
            return BinaryFocalLoss(self.cfg.focal_gamma, self.cfg.focal_alpha)
        # weighted BCE fallback (pos_weight = n_neg / n_pos)
        pos = max(1, int(np.sum(y_train)))
        neg = max(1, int(len(y_train) - pos))
        pw = torch.tensor([neg / pos], dtype=torch.float32)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = True):
        criterion = self._make_loss(np.asarray(y_train))
        optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.nn_lr,
                                 weight_decay=self.cfg.nn_weight_decay)
        train_loader = self._loader(X_train, y_train, shuffle=True)

        best_score, best_state, since_best = -np.inf, None, 0
        for epoch in range(self.cfg.nn_epochs):
            self.net.train()
            running = 0.0
            for xb, yb in train_loader:
                optim.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                # Gradient clipping keeps the recurrent path stable under focal loss.
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optim.step()
                running += loss.item() * len(xb)
            train_loss = running / len(train_loader.dataset)

            if X_val is not None and y_val is not None:
                val_scores = self.predict_proba(X_val)
                score = average_precision_score(y_val, val_scores)
                improved = score > best_score
                if improved:
                    best_score = score
                    best_state = copy.deepcopy(self.net.state_dict())
                    since_best = 0
                else:
                    since_best += 1
                if verbose:
                    print(f"[{self.name}] epoch {epoch+1:02d} "
                          f"loss={train_loss:.4f} val_pr_auc={score:.4f}"
                          f"{' *' if improved else ''}")
                if since_best >= self.cfg.nn_patience:
                    if verbose:
                        print(f"[{self.name}] early stop at epoch {epoch+1}")
                    break
            elif verbose:
                print(f"[{self.name}] epoch {epoch+1:02d} loss={train_loss:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        out = []
        for (xb,) in self._loader(X, None, shuffle=False):
            out.append(torch.sigmoid(self.net(xb)).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def attention_weights(self, X) -> Optional[np.ndarray]:
        """Return per-timestep attention if the wrapped net supports it (LSTM)."""
        if not hasattr(self.net, "forward"):
            return None
        self.net.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        try:
            _, w = self.net(Xt, return_attn=True)
            return w.cpu().numpy()
        except TypeError:
            return None
