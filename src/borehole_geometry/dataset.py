"""Phase 6 — dataset assembly. One example per well: own statics +
categoricals, the OWN-LEG set, the NEIGHBOUR set (nearest MAX_NEIGHBOR_WELLS
by closest approach), honest splits, and per-fold normalization stats.

    uv run python -m borehole_geometry.dataset
        -> data/processed/{train,val,test,score}.parquet   (list-columns for sets)
        -> models/vocabs.json, models/norms.json

SPLITS (lesson 08 as production code -- both defences, composed):
  test   = targeted wells with first production >= TEMPORAL_HOLDOUT.
  folds  = GroupKFold by PAD over the remaining targeted wells (K=5).
  PURGE  = a pad that contains any test-era well has its train-era wells
           REMOVED from train/val entirely: a sibling standing on the test
           well's pad would leak the pad effect across the temporal wall.
  PAD    = licences whose surface points share a 150 m grid cell. Surface
           point = the first survey station of the licence's bores.

VOCABS (lesson 11): play = Petrinex pool code; formation = first word of the
pool NAME (coarse). Codes below MIN_PEER_WELLS training wells bucket to
OTHER; index 0 is reserved for UNKNOWN/missing. Vocabs are built from
TRAIN-ERA wells only, so a play first seen in the holdout maps to OTHER
exactly as it would in deployment.

NORMALIZATION: set features are hand-scaled to O(1) (documented inline --
they carry physical units into interpretable ranges). own_static is
z-scored at TRAIN TIME using models/norms.json, which stores per-fold
mean/std computed EXCLUDING that fold (and always excluding test): lesson
06's train-stats discipline survives fold cycling.

Score set = wells with geometry and a production window but no valid target
(censored births, storage, thermal, commingled): the model can be ASKED
about them; it just never learns from their labels.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from . import config as C

P = C.DATA_PROC
DB = C.ROOT / "data" / "ppdm.duckdb"
K_FOLDS = 5
PAD_CELL_M = 150.0


# ---------------------------------------------------- neighbour features --
# THE canonical neighbour feature transform. The SQL in build_dataset MUST
# mirror this exactly (a consistency check at build time enforces it), and
# the physics probes in report.py build synthetic/moved neighbours through
# this function -- so a probe can never sweep distance while forgetting to
# sweep the distance-DERIVED features.
#
# The kernel features exist because Phase 9 showed the model distance-
# indifferent: raw dist_km next to raw log_during asks the net to invent
# "withdrawal x nearness" itself. exp(-d/lambda) is the shape interference
# physics suggests (pressure influence decays with distance), and the
# interaction features hand over the product ready-made (lesson 06: a model
# can only build shapes from the pieces it is given).
NBR_KEYS = ["dist_km", "overlap", "az_delta", "dz_100m", "age_yr", "log_cum",
            "log_during", "months_frac", "log_inj", "is_steam", "same_play",
            "censored", "has_prod",
            "inv_d", "k300", "drain_k300", "cum_k300"]


def nbr_vec(dist_m, overlap, az_delta_deg, dz_m, age_months, cum_boe,
            during_boe, months_active, inj_boe, is_steam, same_play,
            censored, has_prod):
    import math
    k300 = math.exp(-dist_m / 300.0)
    return [dist_m / 1000.0, overlap, az_delta_deg / 90.0, dz_m / 100.0,
            age_months / 12.0, math.log1p(cum_boe) / 10.0,
            math.log1p(during_boe) / 10.0, months_active / 12.0,
            math.log1p(inj_boe) / 10.0, float(is_steam), float(same_play),
            float(censored), float(has_prod),
            1.0 / (dist_m / 1000.0 + 0.05), k300,
            math.log1p(during_boe * k300) / 10.0,
            math.log1p(cum_boe * k300) / 10.0]


# ------------------------------------------------------------------ pads --
def assign_pads(con) -> pd.DataFrame:
    """well_key -> pad_id from surface-point clustering on a 150 m grid."""
    return con.execute(f"""
        WITH surf AS (
            SELECT l.well_key,
                   median(CAST(s.LONGITUDE AS DOUBLE)) lon,
                   median(CAST(s.LATITUDE AS DOUBLE)) lat
            FROM read_parquet('{P}/legs.parquet') l
            JOIN well_dir_srvy_station s
              ON s.UWI = l.uwi AND s.SOURCE = 'AER' AND s.DEPTH_OBS_NO = 1
            GROUP BY 1)
        SELECT well_key,
               CAST(floor(lon * 111320 * cos(radians(55)) / {PAD_CELL_M}) AS BIGINT) * 100000
             + CAST(floor(lat * 111320 / {PAD_CELL_M}) AS BIGINT) AS pad_id
        FROM surf
    """).df()


# ---------------------------------------------------------------- vocabs --
def build_vocab(series: pd.Series, min_count: int) -> dict:
    counts = series.value_counts()
    keep = [c for c in counts.index if counts[c] >= min_count]
    return {"UNKNOWN": 0, "OTHER": 1, **{c: i + 2 for i, c in enumerate(sorted(keep))}}


def to_idx(series: pd.Series, vocab: dict) -> pd.Series:
    return series.map(lambda v: vocab.get(v, vocab["OTHER"])
                      if pd.notna(v) and v != "UNKNOWN" else vocab["UNKNOWN"])


# ----------------------------------------------------------------- build --
def build_dataset():
    con = duckdb.connect(str(DB), read_only=True)
    cohort = pd.read_parquet(P / "cohort.parquet")
    cohort = cohort[cohort.length_class.notna() & cohort.first_m.notna()].copy()

    pads = assign_pads(con)
    cohort = cohort.merge(pads, on="well_key", how="left")

    # AGS Geological Framework surfaces sampled at each well (Phase 9 finding:
    # without a real rock-quality measurement, neighbour presence proxies
    # geology and poisons the distance channel; these are the measurement).
    # NaN means the unit does not exist at the well -- real geology; the
    # derived features encode absence as 0 thickness plus a coverage count.
    ags = pd.read_parquet(P / "ags_surfaces.parquet")
    cohort = cohort.merge(ags, on="well_key", how="left")
    cohort["pad_id"] = cohort.pad_id.fillna(-1).astype("int64")

    # -- categoricals ------------------------------------------------------
    cohort["formation"] = cohort.play_name.str.split().str[0]
    targeted = cohort[cohort.target_pace_pct.notna()].copy()
    targeted["is_test"] = targeted.is_holdout_era
    train_era = targeted[~targeted.is_test]

    vocabs = {
        "play": build_vocab(train_era.play_code[train_era.play_code != "UNKNOWN"],
                            C.MIN_PEER_WELLS),
        "formation": build_vocab(train_era.formation.dropna(), C.MIN_PEER_WELLS),
        "province": {"AB": 0},
    }
    cohort["play_idx"] = to_idx(cohort.play_code, vocabs["play"])
    cohort["formation_idx"] = to_idx(cohort.formation, vocabs["formation"])
    cohort["province_idx"] = 0

    # -- own_static (raw; z-scored at train time via norms.json) -----------
    vintage0 = 2022 * 12 + 1
    first_mi = (cohort.first_m.str[:4].astype(int) * 12
                + cohort.first_m.str[5:7].astype(int))
    statics = pd.DataFrame({
        "well_key": cohort.well_key,
        "total_lateral_km": cohort.total_lateral_m / 1000.0,
        "n_legs": cohort.n_legs.astype(float),
        "n_bores_reported": cohort.n_bores_reported.fillna(cohort.n_legs).astype(float),
        "mean_leg_km": cohort.mean_leg_m / 1000.0,
        "fan_spread_norm": cohort.fan_spread_deg / 90.0,
        "tvdss_km": cohort.tvdss_mean_m / 1000.0,
        "vintage_yr": (first_mi - vintage0) / 12.0,   # drift feature; test-era
                                                      # extrapolates it (lesson 08 exp3)
        # --- AGS geology (grids sampled at the well; elevations in m ASL) ---
        "geo_column_km": (cohort.top_sediment - cohort.top_precambrian) / 1000.0,
        "geo_rel_viking_km": (cohort.top_viking - (-cohort.tvdss_mean_m)) / 1000.0,
        "geo_vt_mnv": cohort.vt_mannville / 100.0,
        "geo_vt_mo": cohort.vt_montney / 100.0,
        "geo_vt_sr": cohort.vt_spirit_river / 100.0,
        "geo_vt_cd": cohort.vt_cardium / 100.0,
    })
    unit_cols = ["geo_vt_mnv", "geo_vt_mo", "geo_vt_sr", "geo_vt_cd"]
    statics["geo_cover"] = statics[unit_cols].notna().sum(axis=1) / len(unit_cols)
    statics[unit_cols] = statics[unit_cols].fillna(0.0)
    statics[["geo_column_km", "geo_rel_viking_km"]] = (
        statics[["geo_column_km", "geo_rel_viking_km"]].fillna(0.0))
    STATIC_COLS = [c for c in statics.columns if c != "well_key"]

    # -- own-leg set (hand-scaled) ----------------------------------------
    legs = con.execute(f"""
        SELECT well_key,
               list(struct_pack(
                   length_km := leg_length_m / 1000.0,
                   sin_az := sin(radians(azimuth_deg)),
                   cos_az := cos(radians(azimuth_deg)),
                   tvd_km := tvdss_mean_m / 1000.0
               ) ORDER BY leg_length_m DESC) AS legs
        FROM read_parquet('{P}/legs.parquet') GROUP BY 1
    """).df()

    # -- neighbour set: nearest MAX_NEIGHBOR_WELLS by closest approach -----
    nbrs = con.execute(f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY well_key ORDER BY dist_m) rn
            FROM read_parquet('{P}/spacing_pairs_ctx.parquet'))
        SELECT well_key,
               count(*) AS n_nbrs_total,
               list(struct_pack(
                   dist_km := dist_m / 1000.0,
                   overlap := overlap_frac,
                   az_delta := azimuth_delta_deg / 90.0,
                   dz_100m := dz_m / 100.0,
                   age_yr := coalesce(nbr_age_months, 0) / 12.0,
                   log_cum := ln(1 + nbr_cum_boe_before) / 10.0,
                   log_during := ln(1 + nbr_boe_during) / 10.0,
                   months_frac := nbr_months_active / 12.0,
                   log_inj := ln(1 + nbr_inj_steam_during + nbr_inj_water_during
                                 + nbr_inj_gas_during + nbr_inj_co2_during) / 10.0,
                   is_steam := CAST(nbr_inj_steam_during > 0 AS DOUBLE),
                   same_play := CAST(same_play AS DOUBLE),
                   censored := CAST(nbr_censored AS DOUBLE),
                   has_prod := CAST(nbr_has_production AS DOUBLE),
                   inv_d := 1.0 / (dist_m / 1000.0 + 0.05),
                   k300 := exp(-dist_m / 300.0),
                   drain_k300 := ln(1 + nbr_boe_during * exp(-dist_m / 300.0)) / 10.0,
                   cum_k300 := ln(1 + nbr_cum_boe_before * exp(-dist_m / 300.0)) / 10.0
               ) ORDER BY dist_m) FILTER (WHERE rn <= {C.MAX_NEIGHBOR_WELLS}) AS neighbors
        FROM ranked GROUP BY 1
    """).df()

    # -- the LOCAL BENCHMARK ("neighbourhood report card") -----------------
    # Phase 9's approach probe failed because nothing tells the model how
    # good the local rock is, so neighbour PRESENCE proxies rock quality and
    # poisons the distance channel. This feature hands the confounder over
    # explicitly: the median peer-normalized pace of nearby OLDER wells.
    # Fairness rules, all three load-bearing:
    #   * the neighbour's whole pace window must CLOSE before the subject's
    #     first month (pure past -- the as-of discipline);
    #   * only wells with a valid target qualify (excluded cohorts' paces
    #     are lies; censored births have no honest pace);
    #   * only TRAIN-ERA neighbours qualify (a holdout-era label must never
    #     ride into any well's features -- not even another test well's).
    holdout_mi = (int(C.TEMPORAL_HOLDOUT[:4]) * 12 + int(C.TEMPORAL_HOLDOUT[5:7]))
    bench = con.execute(f"""
        WITH coh AS (
            SELECT well_key, target_pace_pct,
                   CAST(substr(first_m,1,4) AS INT)*12
                   + CAST(substr(first_m,6,2) AS INT) AS fmi
            FROM read_parquet('{P}/cohort.parquet') WHERE first_m IS NOT NULL)
        SELECT s.well_key,
               median(n.target_pace_pct) AS local_bench_pct,
               count(*) AS n_bench
        FROM read_parquet('{P}/spacing_pairs_ctx.parquet') x
        JOIN coh s ON s.well_key = x.well_key
        JOIN coh n ON n.well_key = x.nbr_key
        WHERE n.target_pace_pct IS NOT NULL
          AND n.fmi + {C.PACE_MONTHS} <= s.fmi
          AND n.fmi < {holdout_mi}
        GROUP BY 1
    """).df()

    df = (cohort[["well_key", "pad_id", "play_idx", "formation_idx", "province_idx",
                  "target_pace_pct", "target_pace_pct_winsor", "is_holdout_era",
                  "first_m", "operator_name", "play_code", "months_elapsed"]]
          .merge(statics, on="well_key")
          .merge(legs, on="well_key", how="left")
          .merge(nbrs, on="well_key", how="left")
          .merge(bench, on="well_key", how="left"))
    df["n_nbrs_total"] = df.n_nbrs_total.fillna(0).astype(int)
    # 77.6% of wells have more neighbours than MAX_NEIGHBOR_WELLS, so the
    # mask-derived count the model computes saturates at 12 -- the TRUE
    # crowding has to arrive via own_static or it is lost.
    df["log_n_nbrs"] = np.log1p(df.n_nbrs_total)
    STATIC_COLS.append("log_n_nbrs")
    # report card: centred at 0 = "an average neighbourhood"; has_bench tells
    # the model when the card is real vs imputed
    df["local_bench"] = (df.local_bench_pct.fillna(100.0) - 100.0) / 100.0
    df["has_bench"] = df.local_bench_pct.notna().astype(float)
    df["log_n_bench"] = np.log1p(df.n_bench.fillna(0))
    STATIC_COLS.extend(["local_bench", "has_bench", "log_n_bench"])

    # -- splits ------------------------------------------------------------
    tgt = df.target_pace_pct.notna()
    test_mask = tgt & df.is_holdout_era
    test_pads = set(df.loc[test_mask, "pad_id"]) - {-1}
    purged = tgt & ~df.is_holdout_era & df.pad_id.isin(test_pads)
    trainval = tgt & ~df.is_holdout_era & ~df.pad_id.isin(test_pads)

    # GroupKFold by pad: deal pads round-robin by size for balanced folds
    rng = np.random.default_rng(C.SPLIT_SEED if hasattr(C, "SPLIT_SEED") else 7)
    pad_sizes = df.loc[trainval].groupby("pad_id").size().sample(frac=1, random_state=7)
    fold_of = {}
    loads = [0] * K_FOLDS
    for pad, size in pad_sizes.sort_values(ascending=False).items():
        f = int(np.argmin(loads))
        fold_of[pad] = f
        loads[f] += size
    df["fold"] = df.pad_id.map(fold_of)
    df.loc[~trainval, "fold"] = np.nan

    df["split"] = "score"
    df.loc[trainval, "split"] = "trainval"
    df.loc[test_mask, "split"] = "test"
    df.loc[purged, "split"] = "purged"

    # -- per-fold norms (exclude the fold itself and test) ------------------
    norms = {}
    for f in range(K_FOLDS):
        sub = df[(df.split == "trainval") & (df.fold != f)]
        norms[str(f)] = {c: {"mean": float(sub[c].mean()), "std": float(sub[c].std() + 1e-8)}
                         for c in STATIC_COLS}
    norms["static_cols"] = STATIC_COLS

    # -- write --------------------------------------------------------------
    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.MODELS / "vocabs.json").write_text(json.dumps(
        {k: v for k, v in vocabs.items()}, indent=1))
    (C.MODELS / "norms.json").write_text(json.dumps(norms, indent=1))
    out = {"train": df[(df.split == "trainval") & (df.fold != 0)],
           "val": df[(df.split == "trainval") & (df.fold == 0)],
           "test": df[df.split == "test"],
           "score": df[df.split == "score"]}
    for name, part in out.items():
        part.to_parquet(P / f"{name}.parquet", index=False)
    df.to_parquet(P / "dataset_all.parquet", index=False)

    # -- python twin vs SQL consistency (the probes depend on this) --------
    chk = con.execute(f"""
        SELECT dist_m, overlap_frac, azimuth_delta_deg, dz_m,
               coalesce(nbr_age_months, 0), nbr_cum_boe_before, nbr_boe_during,
               nbr_months_active,
               nbr_inj_steam_during + nbr_inj_water_during
               + nbr_inj_gas_during + nbr_inj_co2_during,
               CAST(nbr_inj_steam_during > 0 AS DOUBLE),
               CAST(same_play AS DOUBLE), CAST(nbr_censored AS DOUBLE),
               CAST(nbr_has_production AS DOUBLE)
        FROM read_parquet('{P}/spacing_pairs_ctx.parquet') LIMIT 200
    """).fetchall()
    sql_rows = con.execute(f"""
        SELECT dist_m / 1000.0, overlap_frac, azimuth_delta_deg / 90.0, dz_m / 100.0,
               coalesce(nbr_age_months, 0) / 12.0,
               ln(1 + nbr_cum_boe_before) / 10.0, ln(1 + nbr_boe_during) / 10.0,
               nbr_months_active / 12.0,
               ln(1 + nbr_inj_steam_during + nbr_inj_water_during
                  + nbr_inj_gas_during + nbr_inj_co2_during) / 10.0,
               CAST(nbr_inj_steam_during > 0 AS DOUBLE),
               CAST(same_play AS DOUBLE), CAST(nbr_censored AS DOUBLE),
               CAST(nbr_has_production AS DOUBLE),
               1.0 / (dist_m / 1000.0 + 0.05), exp(-dist_m / 300.0),
               ln(1 + nbr_boe_during * exp(-dist_m / 300.0)) / 10.0,
               ln(1 + nbr_cum_boe_before * exp(-dist_m / 300.0)) / 10.0
        FROM read_parquet('{P}/spacing_pairs_ctx.parquet') LIMIT 200
    """).fetchall()
    worst = max(max(abs(a - b) for a, b in zip(nbr_vec(*raw), sql))
                for raw, sql in zip(chk, sql_rows))
    assert worst < 1e-9, f"python nbr_vec drifted from SQL: {worst}"
    print(f"  nbr_vec python/SQL consistency: max dev {worst:.1e}  OK")

    # -- report -------------------------------------------------------------
    print(f"  targeted wells: {int(tgt.sum()):,}   pads among them: "
          f"{df.loc[tgt, 'pad_id'].nunique():,}")
    print(f"  test (>= {C.TEMPORAL_HOLDOUT[:7]}): {int(test_mask.sum()):,}   "
          f"purged (train-era siblings of test pads): {int(purged.sum()):,}   "
          f"train/val: {int(trainval.sum()):,}")
    sizes = df[df.split == 'trainval'].fold.value_counts().sort_index().tolist()
    print(f"  fold sizes: {sizes}")
    bc = df.loc[tgt, 'has_bench'].mean() * 100
    print(f"  targeted wells with a real report card: {bc:.1f}%   "
          f"median card (where real): "
          f"{df.loc[tgt & (df.has_bench > 0), 'local_bench_pct'].median():.0f}%")
    trunc = (df.loc[tgt, 'n_nbrs_total'] > C.MAX_NEIGHBOR_WELLS).mean() * 100
    print(f"  wells truncated at {C.MAX_NEIGHBOR_WELLS} neighbours: {trunc:.1f}%   "
          f"score set: {len(out['score']):,}")
    print(f"  vocab sizes: play={len(vocabs['play'])} formation={len(vocabs['formation'])}")
    # structural checks
    a = df[(df.split == 'trainval') & (df.fold == 0)].pad_id
    b = df[(df.split == 'trainval') & (df.fold != 0)].pad_id
    assert not (set(a) & set(b)), "a pad straddles val and train folds"
    assert not (set(df.loc[test_mask, 'pad_id']) - {-1}) & set(b), "test pad leaked into train"
    print("  split integrity: no pad straddles folds; no test pad in train  OK")
    return df


if __name__ == "__main__":
    build_dataset()
