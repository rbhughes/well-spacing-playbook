"""Phase 10 — counterfactual scoring. For every well with features:

    (1) forward pass with actual sets                 -> p50_real
    (2) neighbours emptied                            -> p50_iso, inter-well
        penalty_pct = (1 - p50_real / p50_iso) * 100
    (3) the deviation model's proximity penalty: within-pad P50 with real
        neighbours minus with neighbours emptied (its probes passed; its
        magnitude is small and stated).

    uv run python -m borehole_geometry.score
        -> data/processed/interference_scores.parquet
        -> reports/counterfactual_summary.md

TWO MODELS, TWO CHANNELS, deliberately (Phase 9 archaeology):
  * The RAW model's counterfactual prices what it demonstrably learned --
    the DEPLETION-AND-CONTEXT effect of the neighbourhood. Its bare-distance
    response is flat, so differences here should be read as "what the
    neighbours' withdrawal history costs", not "what proximity costs".
  * The DEVIATION model prices within-pad PROXIMITY (the only level where
    it is identified). Small numbers, honestly small.

`in_training_domain`: play seen in training AND first production inside the
window AND not an excluded regime (thermal/storage/commingled get scores
too, but their flag is FALSE and a penalty without this flag must never be
presented -- RECIPE Phase 10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import config as C
from .dataset import NBR_KEYS
from .report import ensemble_predict, load_models

I_LOG_CUM = NBR_KEYS.index("log_cum")
I_CUM_K = NBR_KEYS.index("cum_k300")

P = C.DATA_PROC


def empty_neighbors(batch, norm=None, static_cols=None):
    """The CONSISTENT counterfactual: empty the bag AND zero the count
    feature through the model's own normalization. An earlier version left
    log_n_nbrs saying "crowded" while showing an empty bag -- an input the
    model never saw, and its garbage output (isolated-well "penalties" of
    600%) was caught by the summary's own sanity line."""
    batch["neighbors_mask"] = torch.zeros_like(batch["neighbors_mask"])
    i = static_cols.index("log_n_nbrs")
    z0 = (0.0 - norm["log_n_nbrs"]["mean"]) / norm["log_n_nbrs"]["std"]
    batch["own_static"][:, i] = z0
    return batch


def zero_depletion(batch, norm=None, static_cols=None):
    """The MARGINAL depletion counterfactual: same neighbours, same bag, same
    count -- only their withdrawal HISTORY erased. This stays on the training
    manifold (fresh neighbours exist everywhere in the data), unlike bag
    removal, and it isolates the one channel the probes verified."""
    batch["neighbors"][:, :, I_LOG_CUM] = 0.0
    batch["neighbors"][:, :, I_CUM_K] = 0.0
    return batch


def score():
    df = pd.read_parquet(P / "dataset_all.parquet")
    df = df[df.months_elapsed.notna()].reset_index(drop=True)
    print(f"scoring {len(df):,} wells (targeted + score-only)")

    raw = load_models()
    dev = load_models("_dev")

    chunks = []
    for lo in range(0, len(df), 4000):
        part = df.iloc[lo:lo + 4000].reset_index(drop=True)
        p_real = ensemble_predict(raw, part)
        p_iso = ensemble_predict(raw, part, mutate=empty_neighbors)
        p_fresh = ensemble_predict(raw, part, mutate=zero_depletion)
        d_real = ensemble_predict(dev, part)[:, 1]
        d_iso = ensemble_predict(dev, part, mutate=empty_neighbors)[:, 1]
        out = pd.DataFrame({
            "well_key": part.well_key,
            "split": part.split,
            "n_nbrs": part.n_nbrs_total,
            "p50_real_pct": p_real[:, 1],
            "p10_real_pct": p_real[:, 0],
            "p90_real_pct": p_real[:, 2],
            "p50_iso_pct": p_iso[:, 1],
            # context penalty in PERCENTAGE POINTS of peer-P50 pace (the
            # target's own unit): isolated-prediction minus real-prediction.
            # A ratio was tried first and explodes when the denominator is
            # small; differences cannot. Negatives are REPORTED, not clipped
            # -- they mark residual good-rock confounding in levels.
            "penalty_ctx_pts": p_iso[:, 1] - p_real[:, 1],
            # MARGINAL depletion penalty: what the neighbours' withdrawal
            # history costs, holding the neighbourhood itself fixed. The
            # probe-verified channel, priced without the bag-removal OOD.
            "penalty_dep_pts": p_fresh[:, 1] - p_real[:, 1],
            # proximity penalty from the deviation model (pad-relative pts)
            "penalty_prox_pts": d_iso - d_real,
        })
        chunks.append(out)
        print(f"  {min(lo + 4000, len(df)):,} done")
    scores = pd.concat(chunks, ignore_index=True)

    # fail-loud domain flag
    coh = pd.read_parquet(P / "cohort.parquet")[
        ["well_key", "is_storage", "is_thermal", "is_commingled",
         "born_in_window", "p50_ref_boe_yr", "pace_boe_yr"]]
    scores = scores.merge(coh, on="well_key", how="left")
    trained_plays = set(df[df.split == "trainval"].play_code)
    play_of = dict(zip(df.well_key, df.play_code))
    scores["in_training_domain"] = (
        scores.well_key.map(play_of).isin(trained_plays)
        & scores.born_in_window.fillna(False)
        & ~scores.is_storage.fillna(False)
        & ~scores.is_thermal.fillna(False)
        & ~scores.is_commingled.fillna(False))

    # volumes: penalty as BOE/yr against the peer P50 reference where known
    scores["penalty_ctx_boe_yr"] = (
        scores.penalty_ctx_pts / 100.0 * scores.p50_ref_boe_yr)
    scores["penalty_dep_boe_yr"] = (
        scores.penalty_dep_pts / 100.0 * scores.p50_ref_boe_yr)

    scores.to_parquet(P / "interference_scores.parquet", index=False)

    # ---- summary ----------------------------------------------------------
    lines = ["# Counterfactual summary (Phase 10)", ""]
    def say(t=""):
        print(t)
        lines.append(t)
    dom = scores[scores.in_training_domain]
    iso0 = scores[scores.n_nbrs == 0]
    say(f"wells scored: {len(scores):,}   in training domain: {len(dom):,}")
    say(f"isolated wells' context penalty (must be ~0): "
        f"max |{iso0.penalty_ctx_pts.abs().max():.4f}| pts")
    say("")
    say("context (depletion) penalty, pts of peer-P50 pace — in-domain wells:")
    for b, m in dom.groupby(pd.cut(dom.n_nbrs, [0, 3, 8, 12, 100],
                                   labels=["1-3", "4-8", "9-12", ">12"],
                                   include_lowest=False), observed=True):
        say(f"  {b:>5} nbrs: n={len(m):>6,}  median {m.penalty_ctx_pts.median():+6.1f}  "
            f"p25/p75 {m.penalty_ctx_pts.quantile(.25):+.1f}/{m.penalty_ctx_pts.quantile(.75):+.1f}  "
            f"negative share {100 * (m.penalty_ctx_pts < 0).mean():.0f}%")
    say("")
    say("MARGINAL depletion penalty (withdrawal history erased, bag kept) — in-domain:")
    for b, m in dom.groupby(pd.cut(dom.n_nbrs, [0, 3, 8, 12, 100],
                                   labels=["1-3", "4-8", "9-12", ">12"],
                                   include_lowest=False), observed=True):
        say(f"  {b:>5} nbrs: n={len(m):>6,}  median {m.penalty_dep_pts.median():+6.1f} pts  "
            f"p25/p75 {m.penalty_dep_pts.quantile(.25):+.1f}/{m.penalty_dep_pts.quantile(.75):+.1f}  "
            f"negative share {100 * (m.penalty_dep_pts < 0).mean():.0f}%")
    med_boe = dom[dom.penalty_dep_boe_yr.notna()].penalty_dep_boe_yr.median()
    say(f"  in volume terms: median {med_boe:,.0f} boe/yr per in-domain well")
    say("")
    say("within-pad proximity penalty (deviation model, pad-relative pts):")
    say(f"  in-domain median {dom.penalty_prox_pts.median():+.2f}  "
        f"p25/p75 {dom.penalty_prox_pts.quantile(.25):+.2f}/{dom.penalty_prox_pts.quantile(.75):+.2f}")
    say("")
    say("READ ME FIRST: penalty_ctx prices the neighbourhood's withdrawal")
    say("history (the identified channel), NOT bare proximity. Negative values")
    say("are printed, not hidden: they mark residual good-rock confounding in")
    say("levels. penalty_prox is small by honest necessity. No number without")
    say("in_training_domain=True should ever be quoted.")
    (C.REPORTS / "counterfactual_summary.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/counterfactual_summary.md")


if __name__ == "__main__":
    score()
