"""Phase 2 — legs. Rebuild every surveyed wellbore's LATERAL from its real
3-D survey stations, group legs into wells (multilaterals) by licence, and
write:

    data/processed/legs.parquet        one row per LEG
    data/processed/wells_legs.parquet  one row per WELL (licence) with the
                                       set-level aggregates

Only `SURVEY_TYPE='SURVEYED'` bores participate: `Calculated` bores are
2-vertex sticks whose depths are fabricated (max TVD == measured depth; see
data/README.md), so they can contribute neither a lateral nor an honest depth.

The lateral is found by LANDING-POINT detection: walk the stations to the
first one within LANDING_TVD_TOL_M of the bore's maximum depth -- that is
where the well stops building angle and starts running -- and take everything
from there to the toe. A vertical well "lands" next to its own toe, yields a
tiny lateral, and is filtered by MIN_LEG_M, which is exactly the right
outcome: no lateral, no leg.

Depths use STATION_TVDSS (positive down, derived directly from the shapefile
Z), which is present on every station; STATION_TVD needs a KB elevation and
is occasionally NULL.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config as C
from .geometry import acute_angle_deg, bearing_deg, latlon_to_local_m, seg_seg_min_dist

DB = C.ROOT / "data" / "ppdm.duckdb"
LEGS_OUT = C.DATA_PROC / "legs.parquet"
WELLS_OUT = C.DATA_PROC / "wells_legs.parquet"


def _fetch_stations(con):
    """All stations of surveyed bores, ordered. Decimals cast to DOUBLE up
    front -- everything after this is numpy."""
    return con.execute("""
        SELECT s.UWI,
               CAST(s.LATITUDE AS DOUBLE)       AS lat,
               CAST(s.LONGITUDE AS DOUBLE)      AS lon,
               CAST(s.STATION_TVDSS AS DOUBLE)  AS tvdss
        FROM well_dir_srvy_station s
        JOIN well_dir_srvy d USING (UWI, SURVEY_ID, SOURCE)
        WHERE d.SURVEY_TYPE = 'SURVEYED'
        ORDER BY s.UWI, s.DEPTH_OBS_NO
    """).fetch_arrow_table()


def _licence_map(con):
    return dict(con.execute("""
        SELECT UWI, ALIAS_LONG_NAME FROM well_alias WHERE WELL_ALIAS_ID = 'LICENCE'
    """).fetchall())


def _leg_from_bore(lat, lon, tvdss):
    """One bore's stations -> its lateral, or None if it never really runs.

    Path lengths are computed in a per-bore local metre frame (reference =
    first station); the frame is only used for differences, so the choice of
    reference is irrelevant to the length.
    """
    if len(lat) < 2:
        return None
    x, y = latlon_to_local_m(lat, lon, float(lat[0]), float(lon[0]))
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(tvdss) ** 2)
    deepest = float(tvdss.max())
    landed = np.flatnonzero(tvdss >= deepest - C.LANDING_TVD_TOL_M)
    heel_i = int(landed[0])
    if heel_i >= len(lat) - 1:
        return None                       # lands at the toe: vertical, no lateral
    lat_len = float(seg[heel_i:].sum())
    if lat_len < C.MIN_LEG_M:
        return None
    return {
        "heel_obs_no": heel_i + 1,      # DEPTH_OBS_NO of the landing point --
                                        # lets SQL slice the lateral's stations
        "heel_lat": float(lat[heel_i]), "heel_lon": float(lon[heel_i]),
        "toe_lat": float(lat[-1]), "toe_lon": float(lon[-1]),
        "leg_length_m": lat_len,
        "azimuth_deg": bearing_deg(float(lat[heel_i]), float(lon[heel_i]),
                                   float(lat[-1]), float(lon[-1])),
        "tvdss_mean_m": float(tvdss[heel_i:].mean()),
        "n_stations": int(len(lat)),
    }


def build_legs():
    t0 = time.time()
    con = duckdb.connect(str(DB), read_only=True)
    lic_of = _licence_map(con)
    tbl = _fetch_stations(con)
    uwi = np.asarray(tbl.column("UWI"))
    lat = np.asarray(tbl.column("lat"), dtype=float)
    lon = np.asarray(tbl.column("lon"), dtype=float)
    tvd = np.asarray(tbl.column("tvdss"), dtype=float)
    print(f"  {len(uwi):,} stations fetched in {time.time() - t0:.0f}s")

    # bore boundaries in the sorted station stream
    cut = np.flatnonzero(uwi[1:] != uwi[:-1]) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(uwi)]])

    legs, no_licence = [], 0
    for a, b in zip(starts, ends):
        leg = _leg_from_bore(lat[a:b], lon[a:b], tvd[a:b])
        if leg is None:
            continue
        bore_uwi = str(uwi[a])
        lic = lic_of.get(bore_uwi)
        if lic is None or lic == "":
            no_licence += 1
            lic = bore_uwi                # a well of one: its own group
        leg["uwi"] = bore_uwi
        leg["well_key"] = lic
        legs.append(leg)
    legs_df = pd.DataFrame(legs)
    print(f"  {len(legs_df):,} legs from {len(starts):,} surveyed bores "
          f"({no_licence:,} without a licence alias) in {time.time() - t0:.0f}s")

    # ---- well-level aggregates -------------------------------------------
    wells = []
    for lic, grp in legs_df.groupby("well_key"):
        azs = grp["azimuth_deg"].to_numpy()
        fan = max((acute_angle_deg(a, b) for i, a in enumerate(azs)
                   for b in azs[i + 1:]), default=0.0)
        # inter-leg spacing: each leg's closest approach to any sibling leg,
        # as straight heel->toe segments in a common local frame
        spacing = np.nan
        if len(grp) > 1:
            ref_lat = float(grp["heel_lat"].mean())
            ref_lon = float(grp["heel_lon"].mean())
            hx, hy = latlon_to_local_m(grp["heel_lat"].to_numpy(),
                                       grp["heel_lon"].to_numpy(), ref_lat, ref_lon)
            tx, ty = latlon_to_local_m(grp["toe_lat"].to_numpy(),
                                       grp["toe_lon"].to_numpy(), ref_lat, ref_lon)
            nearest = []
            for i in range(len(grp)):
                d = min(seg_seg_min_dist((hx[i], hy[i]), (tx[i], ty[i]),
                                         (hx[j], hy[j]), (tx[j], ty[j]))
                        for j in range(len(grp)) if j != i)
                nearest.append(d)
            spacing = float(np.mean(nearest))
        wells.append({
            "well_key": lic,
            "n_legs": len(grp),
            "total_lateral_m": float(grp["leg_length_m"].sum()),
            "mean_leg_m": float(grp["leg_length_m"].mean()),
            "fan_spread_deg": fan,
            "mean_interleg_m": spacing,
            "tvdss_mean_m": float(grp["tvdss_mean_m"].mean()),
        })
    wells_df = pd.DataFrame(wells)

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    legs_df.to_parquet(LEGS_OUT, index=False)
    wells_df.to_parquet(WELLS_OUT, index=False)

    # ---- the Phase 2 report checks ---------------------------------------
    multi = wells_df[wells_df.n_legs >= 2]
    capped = int((wells_df.n_legs > C.MAX_OWN_LEGS).sum())
    print(f"  wells (licences) with >=1 leg: {len(wells_df):,}")
    print(f"  multilateral (>=2 legs): {len(multi):,} "
          f"({len(multi) / len(wells_df) * 100:.1f}%)  "
          f"leg-count p50/p95/max: {int(wells_df.n_legs.median())}/"
          f"{int(wells_df.n_legs.quantile(0.95))}/{int(wells_df.n_legs.max())}")
    print(f"  wells above MAX_OWN_LEGS={C.MAX_OWN_LEGS}: {capped:,}")
    print(f"  lateral length p50: {wells_df.total_lateral_m.median():,.0f} m   "
          f"leg length p50: {legs_df.leg_length_m.median():,.0f} m")
    # Spot-probe the biggest multilaterals rather than a fixed licence:
    # licence 0257776 (9 surveyed bores) turned out to be a cluster of
    # shallow DEVIATED bores with 20-220 m laterals, mostly and correctly
    # rejected by MIN_LEG_M -- a reminder that "many bores" != "fishbone".
    top = wells_df.nlargest(3, "n_legs")
    for _, w in top.iterrows():
        print(f"  probe: licence {w.well_key} -> {int(w.n_legs)} legs, "
              f"total {w.total_lateral_m:,.0f} m, "
              f"mean inter-leg {w.mean_interleg_m:,.0f} m")
    print(f"  wrote {LEGS_OUT.name} + {WELLS_OUT.name} in {time.time() - t0:.0f}s")
    return legs_df, wells_df


if __name__ == "__main__":
    build_legs()
