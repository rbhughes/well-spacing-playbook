"""The interference model (Phase 8): two permutation-invariant set encoders — one over a well's
OWN legs (intra-well / fishbone self-interference), one over its NEIGHBOUR wells (inter-well
spacing) — fused with the well's own static features to predict production pace as P10/P50/P90.

The counterfactual (Phase 10) is a forward pass with one or both sets emptied: empty the
neighbour set -> "isolated well" pace -> inter-well penalty; thin the own-leg set -> intra-well
penalty. So the empty-set path is not an edge case — it IS the headline output, and it must be
finite and tested (see tests/test_model.py).

This module is self-contained (no data/IO) so the architecture is unit-testable in isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """Shared per-element MLP + masked attention pooling over a variable-size set. Padded slots
    are masked out of the softmax; an all-empty set pools to zeros (the counterfactual input)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.score = nn.Linear(hidden, 1)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)   mask: (B, N) bool, True = real element
        h = self.phi(x)                                     # (B, N, H)
        scores = self.score(h).squeeze(-1)                  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        # A fully-empty set would make softmax all -inf -> NaN; detect and zero it explicitly.
        empty = ~mask.any(dim=1, keepdim=True)              # (B, 1)
        weights = torch.softmax(scores, dim=1)              # (B, N)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)     # (B, H)
        return torch.where(empty, torch.zeros_like(pooled), pooled)


class InterferenceModel(nn.Module):
    """own-static + pooled(own legs) + pooled(neighbour wells) -> 3 monotone quantiles of pace."""

    def __init__(
        self,
        own_static_dim: int,
        leg_feat_dim: int,
        neighbor_feat_dim: int,
        n_formation: int,
        n_play: int,
        n_province: int,
        emb=(16, 8, 4),
        hidden: int = 64,
    ):
        super().__init__()
        self.emb_formation = nn.Embedding(n_formation, emb[0])
        self.emb_play = nn.Embedding(n_play, emb[1])
        self.emb_province = nn.Embedding(n_province, emb[2])
        self.leg_enc = SetEncoder(leg_feat_dim, hidden)
        self.nbr_enc = SetEncoder(neighbor_feat_dim, hidden)
        fused = own_static_dim + sum(emb) + self.leg_enc.out_dim + self.nbr_enc.out_dim + 2
        self.head = nn.Sequential(
            nn.Linear(fused, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 3)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        e = torch.cat(
            [
                self.emb_formation(batch["formation_idx"]),
                self.emb_play(batch["play_idx"]),
                self.emb_province(batch["province_idx"]),
            ],
            dim=-1,
        )
        legs = self.leg_enc(batch["legs"], batch["legs_mask"])
        nbrs = self.nbr_enc(batch["neighbors"], batch["neighbors_mask"])
        n_legs = batch["legs_mask"].sum(1, keepdim=True).float() / 24.0
        n_nbrs = batch["neighbors_mask"].sum(1, keepdim=True).float() / 12.0
        x = torch.cat([batch["own_static"], e, legs, nbrs, n_legs, n_nbrs], dim=-1)
        q = self.head(x)                                    # (B, 3) raw quantiles
        # Enforce P10 <= P50 <= P90 by construction (cumulative softplus offsets).
        base = q[:, :1]
        d = torch.nn.functional.softplus(q[:, 1:])
        return torch.cat([base, base + d[:, :1], base + d[:, :1] + d[:, 1:]], dim=1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles=(0.1, 0.5, 0.9)) -> torch.Tensor:
    """Summed quantile (pinball) loss. Robust to the fat pace tail that would wreck MSE, and it
    yields calibrated P10/P50/P90 for free (checked in Phase 9)."""
    target = target.unsqueeze(1)
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    return torch.maximum(q * err, (q - 1.0) * err).sum(dim=1).mean()
