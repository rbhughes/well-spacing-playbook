"""Phases 3+4 — pairs. Candidate pairing by SQL grid, exact leg-to-leg
closest approach on TRUE lateral polylines via DuckDB spatial, directed
well-pair features, and intra-well leg crowding.

    uv run python -m borehole_geometry.pairs
        -> data/processed/spacing_pairs.parquet   (directed well pairs)
        -> data/processed/well_intra.parquet      (own-leg crowding)

PIPELINE (order matters -- learned by measurement, see below):
  1. laterals: ST_MakeLine over each leg's stations from the heel down,
     ST_Transform 4269 -> EPSG:3400 (metres). ~1 s for 105k laterals.
  2. candidates: 2 km grid cells on bbox COLUMNS ONLY -- geometry payloads
     stay out of the join. An earlier version dragged geometries through
     this stage and spent 8 of 16 minutes on it.
  3. bbox-gap filter (scalar), THEN attach geometry and compute exact
     ST_Distance once per surviving pair. Exact stage: ~25 s for 8.5M pairs.
  4. roll leg-pairs up to well-pairs at the closest leg pair; emit BOTH
     directions with the subject's own overlap fraction (overlap is
     asymmetric: a short leg can be fully shadowed by a long one that
     barely notices it).

TRAPS built into the tools (verified in this repo, 2026-08-26):
  * ST_Transform without always_xy := true reads EPSG:4269 in authority
    axis order (lat, lon) and returns inf for (lon, lat) input.
  * ST_Distance on untransformed geometry silently returns DEGREES.
  * A naive ST_DWithin self-join is a nested loop -- billions of geometry
    comparisons; the grid prefilter is not optional.
GEOS is 2-D: horizontal distance from geometry, vertical offset separately
from TVDSS. That decomposition is what interference physics wants anyway.

Overlap is measured on the closest leg pair by projecting both legs'
heel/toe onto their mean azimuth axis and intersecting the intervals --
a straight-line approximation that is honest for laterals (tortuosity ~1
for all but 4 flagged legs) and directed by construction.
"""

from __future__ import annotations

import time

import duckdb

from . import config as C

DB = C.ROOT / "data" / "ppdm.duckdb"
LEGS = C.DATA_PROC / "legs.parquet"
PAIRS_OUT = C.DATA_PROC / "spacing_pairs.parquet"
INTRA_OUT = C.DATA_PROC / "well_intra.parquet"

CELL = C.GRID_CELL_M
RADIUS = C.NEIGHBOR_RADIUS_M


def build_pairs():
    t0 = time.time()
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("INSTALL spatial; LOAD spatial")

    # -- 1. lateral polylines in metres, plus per-leg scalars ---------------
    con.execute(f"""
        CREATE TEMP TABLE lat_geom AS
        SELECT l.uwi, l.well_key,
               l.leg_length_m, l.azimuth_deg, l.tvdss_mean_m,
               ST_Transform(ST_MakeLine(list(ST_Point(CAST(s.LONGITUDE AS DOUBLE),
                                                      CAST(s.LATITUDE AS DOUBLE))
                                             ORDER BY s.DEPTH_OBS_NO)),
                            'EPSG:4269', 'EPSG:3400', true) AS geom
        FROM read_parquet('{LEGS}') l
        JOIN well_dir_srvy_station s
          ON s.UWI = l.uwi AND s.SOURCE = 'AER' AND s.DEPTH_OBS_NO >= l.heel_obs_no
        GROUP BY l.uwi, l.well_key, l.leg_length_m, l.azimuth_deg, l.tvdss_mean_m
    """)
    # heel/toe in the same metre frame, for the overlap axis projection
    con.execute(f"""
        CREATE TEMP TABLE lat_meta AS
        SELECT g.uwi, g.well_key, g.leg_length_m, g.azimuth_deg, g.tvdss_mean_m,
               ST_XMin(g.geom) xmin, ST_XMax(g.geom) xmax,
               ST_YMin(g.geom) ymin, ST_YMax(g.geom) ymax,
               ST_X(ST_StartPoint(g.geom)) hx, ST_Y(ST_StartPoint(g.geom)) hy,
               ST_X(ST_EndPoint(g.geom)) tx, ST_Y(ST_EndPoint(g.geom)) ty
        FROM lat_geom g
    """)
    print(f"[{time.time()-t0:4.0f}s] laterals: "
          f"{con.execute('SELECT count(*) FROM lat_geom').fetchone()[0]:,}")

    # -- 2. grid candidates on scalars only ---------------------------------
    con.execute(f"""
        CREATE TEMP TABLE gridded AS
        SELECT c.uwi, gx.g cx, gy.g cy
        FROM (SELECT uwi,
                     CAST(floor(xmin/{CELL}) AS INT) x0, CAST(floor(xmax/{CELL}) AS INT) x1,
                     CAST(floor(ymin/{CELL}) AS INT) y0, CAST(floor(ymax/{CELL}) AS INT) y1
              FROM lat_meta) c,
             LATERAL (SELECT unnest(generate_series(c.x0-1, c.x1+1)) g) gx,
             LATERAL (SELECT unnest(generate_series(c.y0-1, c.y1+1)) g) gy
    """)
    con.execute("""
        CREATE TEMP TABLE cand AS
        SELECT DISTINCT a.uwi uwi_a, b.uwi uwi_b
        FROM gridded a JOIN gridded b ON a.cx = b.cx AND a.cy = b.cy AND a.uwi < b.uwi
    """)
    con.execute(f"""
        CREATE TEMP TABLE candb AS
        SELECT c.uwi_a, c.uwi_b
        FROM cand c
        JOIN lat_meta a ON a.uwi = c.uwi_a
        JOIN lat_meta b ON b.uwi = c.uwi_b
        WHERE a.xmin - {RADIUS} <= b.xmax AND b.xmin - {RADIUS} <= a.xmax
          AND a.ymin - {RADIUS} <= b.ymax AND b.ymin - {RADIUS} <= a.ymax
    """)
    print(f"[{time.time()-t0:4.0f}s] candidates after bbox gap: "
          f"{con.execute('SELECT count(*) FROM candb').fetchone()[0]:,}")

    # -- 3. exact polyline distance, once per surviving leg pair ------------
    con.execute(f"""
        CREATE TEMP TABLE leg_pairs AS
        SELECT c.uwi_a, c.uwi_b, ga.well_key wk_a, gb.well_key wk_b,
               ST_Distance(ga.geom, gb.geom) AS dist_m
        FROM candb c
        JOIN lat_geom ga ON ga.uwi = c.uwi_a
        JOIN lat_geom gb ON gb.uwi = c.uwi_b
        WHERE ST_Distance(ga.geom, gb.geom) <= {RADIUS}
    """)
    print(f"[{time.time()-t0:4.0f}s] leg pairs <= {RADIUS:.0f} m: "
          f"{con.execute('SELECT count(*) FROM leg_pairs').fetchone()[0]:,}")

    # -- 4a. intra-well crowding (same licence) -----------------------------
    # Sibling legs' polylines SHARE the mother-bore trunk, so their raw
    # ST_Distance is 0 for 93.6% of multilaterals (measured) -- true geometry,
    # useless as a crowding signal. The signal that means something is TIP
    # spacing: how far each leg's toe sits from its nearest sibling's path.
    # A tight fan has close toes; a wide fan has distant ones. The raw min is
    # kept as trunk_shared evidence, not spacing.
    con.execute("""
        CREATE TEMP TABLE intra AS
        SELECT lp.wk_a AS well_key,
               min(least(ST_Distance(ST_EndPoint(ga.geom), gb.geom),
                         ST_Distance(ST_EndPoint(gb.geom), ga.geom))) AS min_toe_dist_m,
               avg(least(ST_Distance(ST_EndPoint(ga.geom), gb.geom),
                         ST_Distance(ST_EndPoint(gb.geom), ga.geom))) AS mean_toe_dist_m,
               min(lp.dist_m) AS min_path_dist_m,
               count(*)       AS n_leg_pairs
        FROM leg_pairs lp
        JOIN lat_geom ga ON ga.uwi = lp.uwi_a
        JOIN lat_geom gb ON gb.uwi = lp.uwi_b
        WHERE lp.wk_a = lp.wk_b
        GROUP BY 1
    """)
    con.execute(f"COPY intra TO '{INTRA_OUT}' (FORMAT parquet)")

    # -- 4b. well-pair rollup at the closest leg pair, then both directions -
    # CANONICALIZE the well pair before rolling up. leg_pairs is ordered by
    # UWI (uwi_a < uwi_b), but well keys do not sort the same way, so one
    # well pair can arrive in both orientations; partitioning on the raw
    # (wk_a, wk_b) then emitted 405,780 duplicate/conflicting directed rows
    # (caught by the data-product review's mirror check).
    con.execute("""
        CREATE TEMP TABLE closest AS
        SELECT * FROM (
            SELECT least(wk_a, wk_b) wk_lo, greatest(wk_a, wk_b) wk_hi,
                   CASE WHEN wk_a <= wk_b THEN uwi_a ELSE uwi_b END uwi_lo,
                   CASE WHEN wk_a <= wk_b THEN uwi_b ELSE uwi_a END uwi_hi,
                   dist_m,
                   row_number() OVER (PARTITION BY least(wk_a, wk_b), greatest(wk_a, wk_b)
                                      ORDER BY dist_m) rn,
                   count(*) FILTER (WHERE dist_m <= 400)
                       OVER (PARTITION BY least(wk_a, wk_b), greatest(wk_a, wk_b)) n_close_legs
            FROM leg_pairs WHERE wk_a <> wk_b)
        WHERE rn = 1
    """)
    # overlap: project both legs' heel/toe on the mean-azimuth axis; the
    # interval intersection over the SUBJECT's own extent is its overlap
    con.execute(f"""
        CREATE TEMP TABLE pair_feats AS
        WITH j AS (
            SELECT c.wk_lo wk_a, c.wk_hi wk_b, c.dist_m, c.n_close_legs,
                   a.azimuth_deg az_a, b.azimuth_deg az_b,
                   a.tvdss_mean_m tv_a, b.tvdss_mean_m tv_b,
                   a.leg_length_m len_a, b.leg_length_m len_b,
                   radians((a.azimuth_deg + b.azimuth_deg) / 2) ax,
                   a.hx ahx, a.hy ahy, a.tx atx, a.ty aty,
                   b.hx bhx, b.hy bhy, b.tx btx, b.ty bty
            FROM closest c
            JOIN lat_meta a ON a.uwi = c.uwi_lo
            JOIN lat_meta b ON b.uwi = c.uwi_hi),
        proj AS (
            SELECT *,
                   least(ahx*sin(ax)+ahy*cos(ax), atx*sin(ax)+aty*cos(ax)) a_lo,
                   greatest(ahx*sin(ax)+ahy*cos(ax), atx*sin(ax)+aty*cos(ax)) a_hi,
                   least(bhx*sin(ax)+bhy*cos(ax), btx*sin(ax)+bty*cos(ax)) b_lo,
                   greatest(bhx*sin(ax)+bhy*cos(ax), btx*sin(ax)+bty*cos(ax)) b_hi
            FROM j)
        SELECT wk_a, wk_b, dist_m, n_close_legs,
               least(abs(az_a - az_b) % 180, 180 - (abs(az_a - az_b) % 180)) AS azimuth_delta_deg,
               greatest(0, least(a_hi, b_hi) - greatest(a_lo, b_lo)) AS overlap_len_m,
               tv_b - tv_a AS dz_m,
               len_a, len_b
        FROM proj
    """)
    con.execute(f"""
        COPY (
            SELECT wk_a AS well_key, wk_b AS nbr_key, dist_m, azimuth_delta_deg,
                   n_close_legs, dz_m,
                   least(1.0, overlap_len_m / greatest(len_a, 1)) AS overlap_frac
            FROM pair_feats
            UNION ALL
            SELECT wk_b, wk_a, dist_m, azimuth_delta_deg, n_close_legs, -dz_m,
                   least(1.0, overlap_len_m / greatest(len_b, 1))
            FROM pair_feats
        ) TO '{PAIRS_OUT}' (FORMAT parquet)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{PAIRS_OUT}')").fetchone()[0]
    print(f"[{time.time()-t0:4.0f}s] directed well pairs written: {n:,}")

    # -- checks: the paranoia, kept ----------------------------------------
    r = con.execute(f"""
        SELECT round(quantile_cont(dist_m, 0.1)), round(median(dist_m)),
               round(quantile_cont(dist_m, 0.9)),
               round(100.0 * avg(CASE WHEN azimuth_delta_deg < 15 THEN 1 ELSE 0 END), 1),
               round(median(overlap_frac), 2)
        FROM read_parquet('{PAIRS_OUT}')
    """).fetchone()
    print(f"  dist p10/p50/p90: {r[0]:.0f}/{r[1]:.0f}/{r[2]:.0f} m   "
          f"parallel(<15deg): {r[3]}%   median overlap_frac: {r[4]}")
    sib = con.execute(f"""
        SELECT round(quantile_cont(dist_m, 0.5))
        FROM read_parquet('{PAIRS_OUT}') WHERE dist_m < 600 AND azimuth_delta_deg < 15
    """).fetchone()[0]
    print(f"  close-parallel median (pad-sibling regime, expect ~150-400 m): {sib:.0f} m")
    assert r[1] > 10, "median pair distance <= 10: distances are probably in DEGREES"
    ii = con.execute(f"""SELECT count(*), round(median(min_toe_dist_m)),
                                round(median(min_path_dist_m))
                         FROM read_parquet('{INTRA_OUT}')""").fetchone()
    print(f"  intra-well: {ii[0]:,} multilaterals, median toe spacing {ii[1]:.0f} m "
          f"(median raw path dist {ii[2]:.0f} m -- 0 means shared trunk, expected)")
    print(f"[{time.time()-t0:4.0f}s] TOTAL")


if __name__ == "__main__":
    build_pairs()
