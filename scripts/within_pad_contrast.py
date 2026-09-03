#!/usr/bin/env python3
"""The within-pad contrast: the cleanest natural experiment in this data.

Siblings on one pad share rock, operator, vintage and completion practice BY
CONSTRUCTION -- what differs is geometry: the edge well faces neighbours on
one flank, the interior well on both. If proximity-interference is real and
identifiable, the more-flanked sibling should pace SLOWER than its own
pad-mates, even though BETWEEN pads the confound runs the other way
(crowded pads sit in better rock).

    exposure := sum over draining neighbours of exp(-dist/300 m)
    within-pad slope := regression of pad-demeaned target on pad-demeaned
                        exposure (pad fixed effects), train-era wells only.
    CI := bootstrap resampling PADS (clusters), not wells.

Writes reports/within_pad_contrast.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from borehole_geometry import config as C  # noqa: E402

P = C.DATA_PROC
con = duckdb.connect()

# exposure per subject: kernel-weighted count of DRAINING neighbours
expo = con.execute(f"""
    SELECT well_key,
           sum(exp(-dist_m / 300.0)) FILTER (WHERE nbr_boe_during > 0) AS exposure,
           sum(ln(1 + nbr_boe_during * exp(-dist_m / 300.0)) / 10.0)
               FILTER (WHERE nbr_boe_during > 0) AS drain_expo
    FROM read_parquet('{P}/spacing_pairs_ctx.parquet')
    GROUP BY 1
""").df()

df = pd.read_parquet(P / "dataset_all.parquet")
df = df[(df.split == "trainval")][["well_key", "pad_id", "target_pace_pct"]]
df = df.merge(expo, on="well_key", how="left").fillna({"exposure": 0.0, "drain_expo": 0.0})

# pads with >=2 train-era targeted wells AND within-pad exposure variation
g = df.groupby("pad_id")
df = df[df.pad_id.isin(g.size()[g.size() >= 2].index) & (df.pad_id != -1)]
spread = df.groupby("pad_id").exposure.transform(lambda x: x.max() - x.min())
df = df[spread > 0.05]

lines = ["# Within-pad contrast (train-era wells; test untouched)", ""]
def say(t=""):
    print(t)
    lines.append(t)

say(f"wells {len(df):,} on {df.pad_id.nunique():,} pads with >=2 targeted "
    f"siblings and real exposure variation")

def slope(frame, xcol):
    x = frame[xcol] - frame.groupby("pad_id")[xcol].transform("mean")
    y = frame.target_pace_pct - frame.groupby("pad_id").target_pace_pct.transform("mean")
    return float((x * y).sum() / (x * x).sum())

def naive_slope(frame, xcol):
    x, y = frame[xcol], frame.target_pace_pct
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x * x).sum())

for xcol, label in (("exposure", "kernel-weighted draining-neighbour count"),
                    ("drain_expo", "kernel-weighted withdrawal")):
    b_within = slope(df, xcol)
    b_naive = naive_slope(df, xcol)
    pads = df.pad_id.unique()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        take = rng.choice(pads, size=len(pads), replace=True)
        sub = pd.concat([df[df.pad_id == p_] for p_ in take]) if False else None
    # fast cluster bootstrap: precompute per-pad sums
    grp = df.assign(
        xd=df[xcol] - df.groupby("pad_id")[xcol].transform("mean"),
        yd=df.target_pace_pct - df.groupby("pad_id").target_pace_pct.transform("mean"))
    per_pad = grp.groupby("pad_id").apply(
        lambda t: pd.Series({"xy": (t.xd * t.yd).sum(), "xx": (t.xd * t.xd).sum()}),
        include_groups=False)
    arr = per_pad.to_numpy()
    idx = rng.integers(0, len(arr), size=(2000, len(arr)))
    xy = arr[:, 0][idx].sum(1)
    xx = arr[:, 1][idx].sum(1)
    bs = xy / xx
    lo, hi = np.percentile(bs, [2.5, 97.5])
    say("")
    say(f"exposure = {label}")
    say(f"  BETWEEN pads (naive, confounded): slope {b_naive:+.1f} pts per unit")
    say(f"  WITHIN pads (rock held fixed):    slope {b_within:+.1f} pts per unit  "
        f"[95% CI {lo:+.1f} .. {hi:+.1f}]")
    say(f"  physics predicts negative; confound predicts positive")

(C.REPORTS / "within_pad_contrast.md").write_text("\n".join(lines) + "\n")
print("\nwrote reports/within_pad_contrast.md")
