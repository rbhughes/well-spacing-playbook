"""Phase 7 — baselines FIRST. The must-beat bar, fold-cycled on train/val.
The test set stays locked until Phase 10.

    uv run python -m borehole_geometry.baselines   -> reports/baselines.md

  B0  global median of the training folds (the MAE floor)
  B1  ridge on OWN-well features only (geology/completion share)
  B2  HistGradientBoosting on own + flattened neighbour aggregates
  B3  HistGradientBoosting with quantile loss (0.1/0.5/0.9) -- the
      format-matched competitor for the torch model: pinball + coverage

GATE (RECIPE Phase 7): if B2 <= B1, the neighbour/leg features carry no
signal and Phases 4-5 need investigation before any torch. Note the raw
neighbour-density gradient is CONFOUNDED (crowded = good rock, see
reports/data_product_review.md), so B2 beating B1 means the features carry
information -- not that the causal sign is settled.

Targets: winsorized pace (target_pace_pct_winsor) for the squared-error
models, exactly why the winsor column exists; B3 quantile fits use the RAW
target, like the torch model will.
"""

from __future__ import annotations

from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

P = C.DATA_PROC
K_FOLDS = 5
QUANTILES = (0.1, 0.5, 0.9)


def flatten_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """List-column neighbour sets -> per-well aggregate columns."""
    rows = []
    for nb in df["neighbors"]:
        if nb is None or len(nb) == 0:
            rows.append(dict(min_dist_km=2.0, top3_overlap=0.0, sum_log_during=0.0,
                             sum_log_cum=0.0, max_log_inj=0.0, any_steam=0.0,
                             frac_same_play=0.0, frac_censored=0.0))
            continue
        d = pd.DataFrame(list(nb))
        rows.append(dict(
            min_dist_km=float(d.dist_km.min()),
            top3_overlap=float(d.nsmallest(3, "dist_km").overlap.mean()),
            sum_log_during=float(d.log_during.sum()),
            sum_log_cum=float(d.log_cum.sum()),
            max_log_inj=float(d.log_inj.max()),
            any_steam=float(d.is_steam.max()),
            frac_same_play=float(d.same_play.mean()),
            frac_censored=float(d.censored.mean()),
        ))
    return pd.DataFrame(rows, index=df.index)


def pinball(y, q_pred, qs=QUANTILES):
    err = y[:, None] - q_pred
    qa = np.asarray(qs)[None, :]
    return float(np.maximum(qa * err, (qa - 1) * err).sum(1).mean())


def run_baselines():
    df = pd.read_parquet(P / "dataset_all.parquet")
    df = df[df.split == "trainval"].reset_index(drop=True)
    nbr_agg = flatten_neighbors(df)
    df = pd.concat([df, nbr_agg], axis=1)

    OWN = ["total_lateral_km", "n_legs", "n_bores_reported", "mean_leg_km",
           "fan_spread_norm", "tvdss_km", "vintage_yr"]
    NBR = ["log_n_nbrs", "min_dist_km", "top3_overlap", "sum_log_during",
           "sum_log_cum", "max_log_inj", "any_steam", "frac_same_play",
           "frac_censored"]
    CAT = ["play_idx", "formation_idx"]

    def ridge_pipe(cols):
        return Pipeline([("prep", ColumnTransformer(
            [("num", StandardScaler(), cols),
             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])),
            ("m", Ridge(alpha=10.0))])

    def gbm(cols, loss="squared_error", q=None):
        # HistGB takes categoricals natively; column order: cols then CAT
        kw = dict(loss=loss, random_state=0,
                  categorical_features=[len(cols), len(cols) + 1])
        if q is not None:
            kw = dict(loss="quantile", quantile=q, random_state=0,
                      categorical_features=[len(cols), len(cols) + 1])
        return HistGradientBoostingRegressor(**kw)

    results = {k: [] for k in ("B0", "B1", "B2", "B3_pinball")}
    cover = {q: [] for q in QUANTILES}
    for f in range(K_FOLDS):
        tr, va = df[df.fold != f], df[df.fold == f]
        yw_tr, yw_va = tr.target_pace_pct_winsor.values, va.target_pace_pct_winsor.values
        yr_tr, yr_va = tr.target_pace_pct.values, va.target_pace_pct.values

        results["B0"].append(float(np.abs(yw_va - np.median(yw_tr)).mean()))

        m1 = ridge_pipe(OWN).fit(tr, yw_tr)
        results["B1"].append(float(np.abs(yw_va - m1.predict(va)).mean()))

        X2_tr = tr[OWN + NBR + CAT].values
        X2_va = va[OWN + NBR + CAT].values
        m2 = gbm(OWN + NBR).fit(X2_tr, yw_tr)
        results["B2"].append(float(np.abs(yw_va - m2.predict(X2_va)).mean()))

        qp = np.column_stack([
            gbm(OWN + NBR, q=q).fit(X2_tr, yr_tr).predict(X2_va) for q in QUANTILES])
        qp.sort(axis=1)     # quantile GBMs can cross; sorting is the honest fix
        results["B3_pinball"].append(pinball(yr_va, qp))
        for i, q in enumerate(QUANTILES):
            cover[q].append(float((yr_va <= qp[:, i]).mean()))

    lines = ["# Baselines (Phase 7) -- fold-cycled on train/val, test locked", "",
             f"{len(df):,} wells, {K_FOLDS} folds. MAE in percentage points of "
             "peer-P50 pace, on the winsorized target.", "",
             "| model | val MAE (mean over folds) | spread |", "|---|---|---|"]
    print(f"  {len(df):,} train/val wells")
    for name, label in (("B0", "B0 global median"), ("B1", "B1 ridge, own-only"),
                        ("B2", "B2 HistGB own+nbr")):
        m, s = mean(results[name]), stdev(results[name])
        print(f"  {label:<22} MAE {m:6.1f}  (+/- {s:.1f})")
        lines.append(f"| {label} | {m:.1f} | +/-{s:.1f} |")
    bp = mean(results["B3_pinball"])
    print(f"  {'B3 HistGB quantile':<22} pinball {bp:6.1f}   coverage "
          + "  ".join(f"P{int(q*100)}={mean(cover[q])*100:.0f}%" for q in QUANTILES))
    lines += ["", f"B3 quantile GBM: mean pinball {bp:.1f}; coverage "
              + ", ".join(f"P{int(q*100)}={mean(cover[q])*100:.0f}%" for q in QUANTILES),
              "", f"Gate: B2 {'BEATS' if mean(results['B2']) < mean(results['B1']) else 'DOES NOT BEAT'} "
              f"B1 -> neighbour/leg features "
              f"{'carry signal' if mean(results['B2']) < mean(results['B1']) else 'carry no signal: STOP'}"]
    gate = mean(results["B2"]) < mean(results["B1"])
    print(f"  GATE: B2 {'beats' if gate else 'DOES NOT beat'} B1 -- "
          f"{'proceed to torch' if gate else 'investigate phases 4-5 first'}")
    (C.REPORTS / "baselines.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote reports/baselines.md")
    return results


if __name__ == "__main__":
    run_baselines()
