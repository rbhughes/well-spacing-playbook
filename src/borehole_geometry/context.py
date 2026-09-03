"""Phase 5 — context. Characterize every geometric neighbour AS OF the
subject's pace window, so the model sees what the subject's first year
actually experienced: who was draining nearby, how depleted they already
were, and who was pushing pressure back IN (injection).

    uv run python -m borehole_geometry.context
        -> data/processed/spacing_pairs_ctx.parquet   (directed, feature-complete)
        -> data/processed/spacing_wells.parquet       (per-subject rollups)

THE AS-OF DISCIPLINE (lesson 08's time leak, applied to FEATURES): every
neighbour quantity is computed strictly from months <= the subject's pace
window end. A neighbour that started producing after the window closed
contributes exactly nothing -- at the time being predicted, it did not exist.

DEVIATIONS FROM RECIPE.md Phase 5 (2026-08-26):
  * Timing anchors on first PRODUCTION month, not spud: spud is not in the
    loaded Petrinex event data, and the pace window is when interference is
    experienced. days_gap becomes age_months, signed.
  * Arps/EUR is skipped: it existed to construct the pace target, which the
    peer-P50 design (Phase 1) replaced.
  * Injection context is NEW (recipe predates it): INJ activity is loaded
    per well-month, split water/steam/gas/CO2. An injecting neighbour
    supports pressure -- opposite sign to a draining one.

CENSORING, stated rather than hidden: the volumetric window opens 2022-01.
A neighbour whose first observed month IS 2022-01 was usually already
producing, so its age and cumulative withdrawal are FLOORS, not values.
`nbr_censored` carries that; the model can learn to trust censored numbers
differently rather than being lied to.
"""

from __future__ import annotations

import time

import duckdb

from . import config as C
from .cohort import COND_BOE, GAS_BOE, OIL_BOE

DB = C.ROOT / "data" / "ppdm.duckdb"
PAIRS = C.DATA_PROC / "spacing_pairs.parquet"
COHORT = C.DATA_PROC / "cohort.parquet"
CTX_OUT = C.DATA_PROC / "spacing_pairs_ctx.parquet"
WELLS_OUT = C.DATA_PROC / "spacing_wells.parquet"


def build_context():
    t0 = time.time()
    con = duckdb.connect(str(DB), read_only=True)

    # -- per-licence monthly activity, as integer month index ---------------
    con.execute(f"""
        CREATE TEMP TABLE lic_month AS
        SELECT a.ALIAS_LONG_NAME AS well_key,
               (CAST(substr(v.PERIOD_ID, 1, 4) AS INT) * 12
                + CAST(substr(v.PERIOD_ID, 6, 2) AS INT)) AS mi,
               sum(CASE WHEN v.ACTIVITY_TYPE = 'PROD' THEN
                     coalesce(CAST(v.OIL_VOLUME AS DOUBLE), 0) * {OIL_BOE}
                     + coalesce(CAST(v.NGL_VOLUME AS DOUBLE), 0) * {COND_BOE}
                     + coalesce(CAST(v.GAS_VOLUME AS DOUBLE), 0) * {GAS_BOE}
                   ELSE 0 END) AS prod_boe,
               sum(CASE WHEN v.ACTIVITY_TYPE = 'INJ' AND v.PRIMARY_PRODUCT = 'STEAM'
                        THEN coalesce(CAST(v.WATER_VOLUME AS DOUBLE), 0) ELSE 0 END) AS inj_steam,
               sum(CASE WHEN v.ACTIVITY_TYPE = 'INJ' AND v.PRIMARY_PRODUCT IS NULL
                        THEN coalesce(CAST(v.WATER_VOLUME AS DOUBLE), 0) ELSE 0 END) AS inj_water,
               sum(CASE WHEN v.ACTIVITY_TYPE = 'INJ'
                        THEN coalesce(CAST(v.GAS_VOLUME AS DOUBLE), 0) ELSE 0 END) AS inj_gas,
               sum(CASE WHEN v.ACTIVITY_TYPE = 'INJ'
                        THEN coalesce(CAST(v.CO2_VOLUME AS DOUBLE), 0) ELSE 0 END) AS inj_co2
        FROM pden_vol_summary v
        JOIN well_alias a ON a.UWI = v.PDEN_ID AND a.WELL_ALIAS_ID = 'LICENCE'
        GROUP BY 1, 2
    """)
    # running cumulative BOE, so depletion-before-X is a lookup not a rescan
    con.execute("""
        CREATE TEMP TABLE lic_cum AS
        SELECT well_key, mi, prod_boe, inj_steam, inj_water, inj_gas, inj_co2,
               sum(prod_boe) OVER (PARTITION BY well_key ORDER BY mi) AS cum_boe
        FROM lic_month
    """)
    print(f"[{time.time()-t0:4.0f}s] licence-months: "
          f"{con.execute('SELECT count(*) FROM lic_cum').fetchone()[0]:,}")

    # -- subjects: every cohort licence with a production window ------------
    con.execute(f"""
        CREATE TEMP TABLE subjects AS
        SELECT well_key,
               (CAST(substr(first_m, 1, 4) AS INT) * 12
                + CAST(substr(first_m, 6, 2) AS INT)) AS w0,
               play_code
        FROM read_parquet('{COHORT}') WHERE first_m IS NOT NULL
    """)
    con.execute(f"""
        CREATE TEMP TABLE nbr_meta AS
        SELECT c.well_key,
               (CAST(substr(c.first_m, 1, 4) AS INT) * 12
                + CAST(substr(c.first_m, 6, 2) AS INT)) AS n0,
               c.first_m = '2022-01' AS censored,
               c.play_code, c.is_storage
        FROM read_parquet('{COHORT}') c WHERE c.first_m IS NOT NULL
    """)

    # -- the big join: pair x window --------------------------------------
    win = C.PACE_MONTHS - 1
    con.execute(f"""
        CREATE TEMP TABLE ctx AS
        SELECT p.well_key, p.nbr_key,
               p.dist_m, p.azimuth_delta_deg, p.overlap_frac, p.dz_m, p.n_close_legs,
               s.w0,
               n.n0 IS NOT NULL AS nbr_has_production,
               s.w0 - n.n0 AS nbr_age_months,           -- signed; >0 = older
               coalesce(n.censored, FALSE) AS nbr_censored,
               coalesce(n.is_storage, FALSE) AS nbr_is_storage,
               (n.play_code IS NOT NULL AND n.play_code = s.play_code) AS same_play,
               -- depletion: everything the neighbour withdrew BEFORE the window
               coalesce((SELECT max(cum_boe) FROM lic_cum m
                         WHERE m.well_key = p.nbr_key AND m.mi < s.w0), 0) AS nbr_cum_boe_before,
               -- activity DURING the subject's pace window
               coalesce(w.boe_during, 0)    AS nbr_boe_during,
               coalesce(w.months_active, 0) AS nbr_months_active,
               coalesce(w.steam_during, 0)  AS nbr_inj_steam_during,
               coalesce(w.water_during, 0)  AS nbr_inj_water_during,
               coalesce(w.gas_during, 0)    AS nbr_inj_gas_during,
               coalesce(w.co2_during, 0)    AS nbr_inj_co2_during
        FROM read_parquet('{PAIRS}') p
        JOIN subjects s ON s.well_key = p.well_key
        LEFT JOIN nbr_meta n ON n.well_key = p.nbr_key
        LEFT JOIN (
            SELECT p2.well_key, p2.nbr_key,
                   sum(m.prod_boe) AS boe_during,
                   sum(CASE WHEN m.prod_boe > 0 THEN 1 ELSE 0 END) AS months_active,
                   sum(m.inj_steam) AS steam_during, sum(m.inj_water) AS water_during,
                   sum(m.inj_gas) AS gas_during, sum(m.inj_co2) AS co2_during
            FROM read_parquet('{PAIRS}') p2
            JOIN subjects s2 ON s2.well_key = p2.well_key
            JOIN lic_cum m ON m.well_key = p2.nbr_key
                          AND m.mi BETWEEN s2.w0 AND s2.w0 + {win}
            GROUP BY 1, 2
        ) w ON w.well_key = p.well_key AND w.nbr_key = p.nbr_key
    """)
    con.execute(f"COPY ctx TO '{CTX_OUT}' (FORMAT parquet)")
    n = con.execute("SELECT count(*) FROM ctx").fetchone()[0]
    print(f"[{time.time()-t0:4.0f}s] context pairs written: {n:,}")

    # -- per-subject rollups ------------------------------------------------
    con.execute(f"""
        COPY (
            SELECT well_key,
                   count(*) AS n_nbrs_2km,
                   sum(CASE WHEN nbr_boe_during > 0 THEN 1 ELSE 0 END) AS n_draining,
                   sum(CASE WHEN nbr_inj_steam_during + nbr_inj_water_during
                            + nbr_inj_gas_during + nbr_inj_co2_during > 0
                       THEN 1 ELSE 0 END) AS n_injecting,
                   min(dist_m) AS min_nbr_dist_m,
                   sum(nbr_boe_during) AS total_nbr_boe_during,
                   sum(nbr_cum_boe_before) AS total_nbr_cum_before,
                   sum(CASE WHEN nbr_censored THEN 1 ELSE 0 END) AS n_censored
            FROM ctx GROUP BY 1
        ) TO '{WELLS_OUT}' (FORMAT parquet)
    """)

    # -- checks --------------------------------------------------------------
    r = con.execute("""
        SELECT round(100.0 * sum(CASE WHEN nbr_boe_during > 0 THEN 1 ELSE 0 END) / count(*), 1),
               round(100.0 * sum(CASE WHEN nbr_censored THEN 1 ELSE 0 END) / count(*), 1),
               round(100.0 * sum(CASE WHEN nbr_inj_steam_during > 0 THEN 1 ELSE 0 END) / count(*), 2),
               round(100.0 * sum(CASE WHEN same_play THEN 1 ELSE 0 END) / count(*), 1)
        FROM ctx""").fetchone()
    print(f"  pairs with draining nbr: {r[0]}%   censored nbr: {r[1]}%   "
          f"steam-injecting nbr: {r[2]}%   same play: {r[3]}%")
    # as-of sanity: an older sibling MUST show depletion; a younger one none
    r = con.execute("""
        SELECT round(avg(CASE WHEN nbr_age_months > 12 AND NOT nbr_censored
                              AND nbr_has_production
                         THEN CASE WHEN nbr_cum_boe_before > 0 THEN 100.0 ELSE 0 END END), 1),
               max(CASE WHEN nbr_age_months < 0 THEN nbr_cum_boe_before END)
        FROM ctx""").fetchone()
    print(f"  older(>12mo) uncensored nbrs with depletion: {r[0]}%   "
          f"max depletion shown by a YOUNGER nbr: {r[1]:.0f} (must be ~0)")
    assert r[1] == 0, "a neighbour born after the subject shows pre-window production: as-of leak"
    print(f"[{time.time()-t0:4.0f}s] TOTAL")


if __name__ == "__main__":
    build_context()
