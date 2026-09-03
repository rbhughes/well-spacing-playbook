"""Phase 8 — training. The set-encoder model against the Phase 7 bar.

    uv run python -m borehole_geometry.train --fold 0
        -> models/interference_f0.pt   (self-contained: state+norms+vocabs+dims)

The bar (fold-cycled Phase 7 baselines): B3 quantile GBM pinball 80.6 with
coverage 16/52/85. Two ways to win: lower val pinball, honest coverage.

Choices, each traceable to a lesson or the recipe:
  batch 512, budgeted in updates              (lesson 10)
  pinball on the RAW target                   (lesson 07; fat tail)
  per-fold z-scoring from models/norms.json   (lesson 06 across folds)
  young-well down-weighting min(m,12)/12      (RECIPE ph8, adapted to the
                                               12-month pace window: an
                                               annualized 6-month pace is a
                                               noisier label)
  early stop on UNWEIGHTED val pinball        (comparable to B3)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .model import InterferenceModel, pinball_loss

P = C.DATA_PROC
from .dataset import NBR_KEYS  # canonical -- probes and SQL share it

LEG_KEYS = ["length_km", "sin_az", "cos_az", "tvd_km"]


class WellDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, static_cols, norm):
        mu = np.array([norm[c]["mean"] for c in static_cols])
        sd = np.array([norm[c]["std"] for c in static_cols])
        self.samples = []
        for r in frame.itertuples():
            own = (np.array([getattr(r, c) for c in static_cols]) - mu) / sd
            legs = (np.array([[leg[k] for k in LEG_KEYS] for leg in r.legs])
                    if r.legs is not None and len(r.legs) else np.zeros((0, len(LEG_KEYS))))
            nbrs = (np.array([[nb[k] for k in NBR_KEYS] for nb in r.neighbors])
                    if r.neighbors is not None and len(r.neighbors) else np.zeros((0, len(NBR_KEYS))))
            self.samples.append({
                "own_static": torch.tensor(own, dtype=torch.float32),
                "formation_idx": int(r.formation_idx),
                "play_idx": int(r.play_idx),
                "province_idx": int(r.province_idx),
                "legs": torch.tensor(legs, dtype=torch.float32),
                "neighbors": torch.tensor(nbrs, dtype=torch.float32),
                "y": float(r.target_pace_pct),
                "w": min(float(r.months_elapsed), C.PACE_MONTHS) / C.PACE_MONTHS,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    def pad(key, dim):
        width = max(1, max(s[key].shape[0] for s in batch))
        out = torch.zeros(len(batch), width, dim)
        mask = torch.zeros(len(batch), width, dtype=torch.bool)
        for i, s in enumerate(batch):
            k = s[key].shape[0]
            out[i, :k] = s[key]
            mask[i, :k] = True
        return out, mask
    legs, legs_mask = pad("legs", len(LEG_KEYS))
    nbrs, nbrs_mask = pad("neighbors", len(NBR_KEYS))
    return {
        "own_static": torch.stack([s["own_static"] for s in batch]),
        "formation_idx": torch.tensor([s["formation_idx"] for s in batch]),
        "play_idx": torch.tensor([s["play_idx"] for s in batch]),
        "province_idx": torch.tensor([s["province_idx"] for s in batch]),
        "legs": legs, "legs_mask": legs_mask,
        "neighbors": nbrs, "neighbors_mask": nbrs_mask,
        "y": torch.tensor([s["y"] for s in batch], dtype=torch.float32),
        "w": torch.tensor([s["w"] for s in batch], dtype=torch.float32),
    }


def weighted_pinball(pred, y, w, quantiles=(0.1, 0.5, 0.9)):
    err = y.unsqueeze(1) - pred
    q = torch.tensor(quantiles, dtype=pred.dtype)
    per = torch.maximum(q * err, (q - 1.0) * err).sum(1)
    return (per * w).sum() / w.sum()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    for b in loader:
        preds.append(model(b))
        ys.append(b["y"])
    model.train()
    pred, y = torch.cat(preds), torch.cat(ys)
    pb = pinball_loss(pred, y).item()
    cov = [(y <= pred[:, i]).float().mean().item() for i in range(3)]
    return pb, cov


def train(fold=0, seed=0, max_epochs=300, patience=30, lr=1e-3, batch=512,
          drop_cols=(), tag="", demeaned=False):
    t0 = time.time()
    norms = json.loads((C.MODELS / "norms.json").read_text())
    vocabs = json.loads((C.MODELS / "vocabs.json").read_text())
    static_cols = [c for c in norms["static_cols"] if c not in drop_cols]
    df = pd.read_parquet(P / "dataset_all.parquet")
    tv = df[df.split == "trainval"]
    if demeaned:
        # PAD-DEMEANED TARGET: predict each well's deviation from its own
        # pad's LEAVE-ONE-OUT sibling mean. The within-pad contrast showed
        # the proximity signal lives at this level (within -5.3 pts/unit,
        # CI excl. 0) while the pooled objective is dominated by the
        # confounded between-pad variation (+1.9). Subtracting the pad mean
        # out of the TARGET removes the confounded variance from the loss.
        # Leave-one-out so a well's own label never enters its own target's
        # reference; pads are intact within folds, so no cross-fold mixing.
        g = tv.groupby("pad_id").target_pace_pct
        pad_sum, pad_n = g.transform("sum"), g.transform("count")
        tv = tv.assign(target_pace_pct=tv.target_pace_pct
                       - (pad_sum - tv.target_pace_pct) / (pad_n - 1))
        tv = tv[(pad_n >= 2) & (tv.pad_id != -1)]
        print(f"  demeaned mode: {len(tv):,} wells on multi-target pads")
    tr = WellDataset(tv[tv.fold != fold], static_cols, norms[str(fold)])
    va = WellDataset(tv[tv.fold == fold], static_cols, norms[str(fold)])
    print(f"  fold {fold}: train {len(tr):,}  val {len(va):,}")

    torch.manual_seed(seed)
    model = InterferenceModel(
        own_static_dim=len(static_cols), leg_feat_dim=len(LEG_KEYS),
        neighbor_feat_dim=len(NBR_KEYS),
        n_formation=len(vocabs["formation"]), n_play=len(vocabs["play"]),
        n_province=len(vocabs["province"]))
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tl = DataLoader(tr, batch_size=batch, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(va, batch_size=1024, shuffle=False, collate_fn=collate)

    best, best_state, best_cov, since = float("inf"), None, None, 0
    for epoch in range(max_epochs):
        for b in tl:
            opt.zero_grad()
            weighted_pinball(model(b), b["y"], b["w"]).backward()
            opt.step()
        pb, cov = evaluate(model, vl)
        if pb < best - 1e-4:
            best, best_cov, since = pb, cov, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 10 == 0 or since == 0 and epoch < 5:
            print(f"    epoch {epoch:3d}  val pinball {pb:7.1f}  "
                  f"cov {cov[0]*100:.0f}/{cov[1]*100:.0f}/{cov[2]*100:.0f}"
                  f"{'  *' if since == 0 else ''}")
        if since >= patience:
            break
    out = C.MODELS / f"interference_f{fold}{tag}.pt"
    torch.save({"state": best_state, "static_cols": static_cols,
                "norm": norms[str(fold)], "vocabs": vocabs,
                "dims": {"own": len(static_cols), "leg": len(LEG_KEYS),
                         "nbr": len(NBR_KEYS)},
                "fold": fold, "val_pinball": best, "val_coverage": best_cov},
               out)
    print(f"  best val pinball {best:.1f}  coverage "
          f"{best_cov[0]*100:.0f}/{best_cov[1]*100:.0f}/{best_cov[2]*100:.0f}  "
          f"({n_par:,} params, {time.time()-t0:.0f}s, stopped epoch {epoch})")
    print(f"  bar: B3 quantile GBM pinball 80.6, coverage 16/52/85")
    print(f"  saved {out.name}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(fold=a.fold, seed=a.seed)
