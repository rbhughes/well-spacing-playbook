"""Phase 1 — cohort. One row per WELL (licence) with the pace target.

    uv run python -m borehole_geometry.cohort   ->  data/processed/cohort.parquet

THE UNIT IS THE LICENCE, not the UWI. Petrinex reports production per well
EVENT (UWI); a multilateral licence can produce through several. The physical
thing whose spacing we study is the licence -- the octopus, not each arm --
so production is summed across a licence's UWIs and geometry comes from
wells_legs.parquet (Phase 2), which is licence-keyed already.

THE TARGET: pace_pct_of_p50 = 100 * pace / P50(peer group), where
  pace       = BOE produced in the first PACE_MONTHS calendar months from
               first production, annualized when the observation window is
               shorter (but never shorter than MIN_MONTHS).
  peer group = (play, lateral-length class)   [decided 2026-08-25]
               play  = Petrinex PoolDeposit (the free play/formation proxy)
               class = total lateral length binned by LENGTH_CLASS_EDGES_M
  P50        = median pace of TRAIN-ERA targetable peers only (first prod
               before TEMPORAL_HOLDOUT). Computing it over everyone would
               leak the holdout era into every target via the denominator.
               Cells thinner than MIN_PEER_WELLS fall back to the length
               class alone, then to the global train-era median.

WHO GETS A TARGET (others stay in the cohort with target NULL -> score-only):
  * first production >= 2022-02. The public volumetric window opens 2022-01,
    and a well whose first REPORT is exactly 2022-01 was usually already
    producing -- its birth is unobservable, so its pace would be a lie.
  * at least MIN_MONTHS calendar months of observable history.
  * at least one qualifying leg (verticals have no lateral length class).

BOE uses the canonical Canadian conversion factors from Bryan's
geoai-server (meridian/api/analysis/sub_profilers/_util.py), which is the
one home for these constants across his production tooling: oil and
condensate in m3 x 6.29287 (1 bbl = 1 boe), gas in e3m3 x 35.3146667/6
(6 mcf = 1 boe). Petrinex reports gas in e3m3 and liquids in m3, per the
same convention that codebase uses against the same source.

LEG COUNT vs BORE COUNT: n_legs counts legs with REAL surveyed geometry.
n_bores_reported counts every bore the regulator recorded on the licence,
including `Calculated` ones whose existence is real even though their
published geometry is a fabricated stick. The model can use the honest
count without ever touching the fake shapes.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from . import config as C

DB = C.ROOT / "data" / "ppdm.duckdb"
INFRA_CSV = C.ROOT / "data" / "work" / "Well_Infrastructure-AB.CSV"
WELLS_LEGS = C.DATA_PROC / "wells_legs.parquet"
OUT = C.DATA_PROC / "cohort.parquet"

# Canonical Canadian BOE factors -- copied from geoai-server _util.py, do not re-derive.
OIL_BOE = 6.29287               # m3 oil -> boe (1 bbl oil = 1 boe)
GAS_BOE = 35.3146667 / 6        # e3m3 gas -> boe (~5.886); mcf = boe x 6
COND_BOE = 6.29287              # m3 condensate -> boe (stage 7 stores Petrinex COND in NGL_VOLUME)


def build_cohort():
    con = duckdb.connect(str(DB), read_only=True)

    # ---- per-licence production: first month, months observed, early BOE --
    # months_elapsed counts CALENDAR months in the pace window that the
    # archive could have observed (capped at PACE_MONTHS), so a quiet month
    # counts against pace rather than being skipped.
    per_licence = con.execute(f"""
        WITH lic AS (
            SELECT UWI, ALIAS_LONG_NAME AS well_key
            FROM well_alias WHERE WELL_ALIAS_ID = 'LICENCE'
        ),
        monthly AS (
            SELECT l.well_key, v.PERIOD_ID,
                   sum(coalesce(CAST(v.OIL_VOLUME AS DOUBLE), 0) * {OIL_BOE}
                       + coalesce(CAST(v.NGL_VOLUME AS DOUBLE), 0) * {COND_BOE}
                       + coalesce(CAST(v.GAS_VOLUME AS DOUBLE), 0) * {GAS_BOE}
                   ) AS boe
            FROM pden_vol_summary v
            JOIN lic l ON l.UWI = v.PDEN_ID
            WHERE v.ACTIVITY_TYPE = 'PROD'   -- INJ rows coexist since 2026-08-26;
                                             -- injected gas is not production
            GROUP BY 1, 2
            HAVING boe > 0
        ),
        firsts AS (
            SELECT well_key, min(PERIOD_ID) AS first_m, count(*) AS n_prod_months
            FROM monthly GROUP BY 1
        )
        SELECT f.well_key, f.first_m, f.n_prod_months,
               least({C.PACE_MONTHS},
                     datediff('month',
                              strptime(f.first_m, '%Y-%m'),
                              strptime('2026-07', '%Y-%m')) + 1) AS months_elapsed,
               sum(m.boe) FILTER (
                   WHERE datediff('month', strptime(f.first_m, '%Y-%m'),
                                  strptime(m.PERIOD_ID, '%Y-%m')) < {C.PACE_MONTHS}
               ) AS early_boe
        FROM firsts f JOIN monthly m USING (well_key)
        GROUP BY 1, 2, 3
    """).df()

    # ---- play proxy per licence (modal PoolDeposit across its UWIs) -------
    play = con.execute(f"""
        WITH lic AS (
            SELECT UWI, ALIAS_LONG_NAME AS well_key
            FROM well_alias WHERE WELL_ALIAS_ID = 'LICENCE'
        )
        SELECT l.well_key,
               mode(i.PoolDeposit)     AS play_code,
               mode(i.PoolDepositName) AS play_name
        FROM read_csv('{INFRA_CSV}', all_varchar=true, quote='"', header=true) i
        JOIN lic l ON l.UWI = i.WellIdentifier
        WHERE i.PoolDeposit IS NOT NULL
        GROUP BY 1
    """).df()

    # Every bore the regulator recorded on the licence -- Calculated bores
    # are real wellbores with fake published geometry, so they count as
    # bores (evidence of a leg) while contributing nothing to geometry.
    bores = con.execute("""
        SELECT a.ALIAS_LONG_NAME AS well_key,
               count(*) AS n_bores_reported,
               count(*) FILTER (WHERE d.SURVEY_TYPE = 'CALCULATED') AS n_bores_calculated
        FROM well_dir_srvy d
        JOIN well_alias a ON a.UWI = d.UWI AND a.WELL_ALIAS_ID = 'LICENCE'
        GROUP BY 1
    """).df()

    # Operator per licence: modal operator BA id across the licence's UWIs,
    # with the legal name from business_associate. Carried both as a future
    # model feature and so the quality report can attribute anomalies --
    # freak paces and broken geometries often cluster by operator (reporting
    # practices differ), and an unattributed outlier list is not actionable.
    operator = con.execute("""
        WITH op AS (
            SELECT a.ALIAS_LONG_NAME AS well_key, mode(w.OPERATOR) AS operator_id
            FROM well w
            JOIN well_alias a ON a.UWI = w.UWI AND a.WELL_ALIAS_ID = 'LICENCE'
            GROUP BY 1)
        SELECT op.well_key, op.operator_id, b.BA_LONG_NAME AS operator_name
        FROM op LEFT JOIN business_associate b
             ON b.BUSINESS_ASSOCIATE_ID = op.operator_id
    """).df()

    wells_legs = pd.read_parquet(WELLS_LEGS)
    df = (per_licence
          .merge(play, on="well_key", how="left")
          .merge(wells_legs, on="well_key", how="left")
          .merge(bores, on="well_key", how="left")
          .merge(operator, on="well_key", how="left"))

    # ---- flags, pace, length class ---------------------------------------
    # Petrinex masks confidential pools with the literal '***' -- that is a
    # missing value, not a play.
    df["play_code"] = df["play_code"].replace("***", None).fillna("UNKNOWN")
    df.loc[df.play_code == "UNKNOWN", "play_name"] = None
    df["born_in_window"] = df["first_m"] >= "2022-02"
    df["is_holdout_era"] = df["first_m"] >= C.TEMPORAL_HOLDOUT[:7]
    df["pace_boe_yr"] = df["early_boe"] * 12.0 / df["months_elapsed"]

    edges = list(C.LENGTH_CLASS_EDGES_M)
    def length_class(m):
        if pd.isna(m):
            return None
        for i, e in enumerate(edges):
            if m < e:
                return f"L{i}"
        return f"L{len(edges)}"
    df["length_class"] = df["total_lateral_m"].map(length_class)

    # Gas-storage operators cycle INJECTED gas; their withdrawal "pace" is
    # not production (observed median target was ~4,155% of peer P50 before
    # exclusion -- see reports/cohort_quality.md). They stay in the cohort as
    # geometry/neighbours but never get a target. Commingled pools are a
    # pending decision and are NOT excluded here.
    df["is_storage"] = df.operator_name.str.contains("STORAGE", na=False)

    # Thermal (SAGD/cyclic steam) is a different physical regime: production
    # is steam-driven, the nearest "neighbour" is often the well's own
    # injector twin stacked 5 m above it, and Petrinex pool codes are largely
    # absent in thermal country so the peer group degrades exactly there
    # (spot subjects scored 345% and 3% of "peer" P50 -- both meaningless).
    # Excluded from TARGETS 2026-08-26; kept as neighbours, where their steam
    # injection is context with the correct sign.
    thermal_keys = set(con.execute(f"""
        SELECT DISTINCT a.ALIAS_LONG_NAME
        FROM read_csv('{INFRA_CSV}', all_varchar=true, quote='"', header=true) i
        JOIN well_alias a ON a.UWI = i.WellIdentifier AND a.WELL_ALIAS_ID = 'LICENCE'
        WHERE i.WellStatusType IN ('SAGD', 'CYCL')
        UNION
        SELECT DISTINCT a.ALIAS_LONG_NAME
        FROM pden_vol_summary v
        JOIN well_alias a ON a.UWI = v.PDEN_ID AND a.WELL_ALIAS_ID = 'LICENCE'
        WHERE v.ACTIVITY_TYPE = 'INJ' AND v.PRIMARY_PRODUCT = 'STEAM'
    """).df().iloc[:, 0])
    df["is_thermal"] = df.well_key.isin(thermal_keys)

    # Commingled pools let several wells report production as ONE stream, so
    # the per-well pace label is an allocation artifact: median target is
    # normal (107%) but the tail is 2x fatter than everyone else's (P99
    # 2,499% vs 1,270%), and commingled wells took 37% of the top-100
    # extreme targets from 15.6% of the population. A wrong LABEL cannot be
    # fixed by a flag FEATURE. Excluded from targets 2026-08-26; kept as
    # neighbours -- their combined withdrawal from the rock is real however
    # the paperwork splits it.
    df["is_commingled"] = df.play_name.str.contains("COMMINGLED", na=False)

    targetable = (df.born_in_window
                  & (df.months_elapsed >= C.MIN_MONTHS)
                  & df.length_class.notna()
                  & (df.pace_boe_yr > 0)
                  & ~df.is_storage
                  & ~df.is_thermal
                  & ~df.is_commingled)

    # ---- P50 reference from TRAIN-ERA targetable wells only ---------------
    ref = df[targetable & ~df.is_holdout_era]
    p50_cell = ref.groupby(["play_code", "length_class"])["pace_boe_yr"].agg(["median", "size"])
    p50_class = ref.groupby("length_class")["pace_boe_yr"].median()
    p50_global = ref["pace_boe_yr"].median()

    def p50_for(row):
        cell = p50_cell.loc[(row.play_code, row.length_class)] \
            if (row.play_code, row.length_class) in p50_cell.index else None
        if cell is not None and cell["size"] >= C.MIN_PEER_WELLS:
            return cell["median"], "play+class"
        if row.length_class in p50_class.index:
            return p50_class[row.length_class], "class"
        return p50_global, "global"

    p50s, srcs = [], []
    for row in df.itertuples():
        if targetable.loc[row.Index]:
            v, src = p50_for(row)
            p50s.append(v)
            srcs.append(src)
        else:
            p50s.append(None)
            srcs.append(None)
    df["p50_ref_boe_yr"] = p50s
    df["p50_source"] = srcs
    df["target_pace_pct"] = 100.0 * df["pace_boe_yr"] / df["p50_ref_boe_yr"]

    # Winsorized target FOR BASELINES ONLY (see config.WINSOR_PCTL): cap from
    # train-era targets, applied to every targeted well. The raw column is
    # what the quantile model trains on.
    train_targets = df.loc[df.target_pace_pct.notna() & ~df.is_holdout_era,
                           "target_pace_pct"]
    cap = float(train_targets.quantile(C.WINSOR_PCTL))
    df["target_pace_pct_winsor"] = df["target_pace_pct"].clip(upper=cap)
    df.attrs["winsor_cap"] = cap

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # ---- Phase 1 report checks -------------------------------------------
    t = df[df.target_pace_pct.notna()]
    print(f"  cohort wells (licences with any production): {len(df):,}")
    print(f"  with geometry (>=1 leg): {df.length_class.notna().sum():,}")
    print(f"  born in window (>=2022-02): {int(df.born_in_window.sum()):,}   "
          f"ambiguous 2022-01 birth: {int((df.first_m == '2022-01').sum()):,}")
    print(f"  with target: {len(t):,} "
          f"(train-era {int((~t.is_holdout_era).sum()):,} / "
          f"holdout-era {int(t.is_holdout_era.sum()):,})")
    print(f"  multilateral share of targeted: by surveyed legs "
          f"{(t.n_legs >= 2).mean() * 100:.1f}%  /  by reported bores "
          f"{(t.n_bores_reported >= 2).mean() * 100:.1f}%")
    hidden = t[(t.n_bores_reported >= 2) & (t.n_legs < t.n_bores_reported)]
    print(f"  targeted wells with bores lacking surveys (leg-count floor): "
          f"{len(hidden):,}")
    print(f"  P50 source: {t.p50_source.value_counts().to_dict()}")
    print(f"  target median/P1/P99: {t.target_pace_pct.median():.0f} / "
          f"{t.target_pace_pct.quantile(0.01):.0f} / "
          f"{t.target_pace_pct.quantile(0.99):.0f}  (pct of peer P50)")
    print(f"  winsor cap (train-era P{int(C.WINSOR_PCTL * 100)}): {cap:.0f}%  "
          f"clipped wells: {int((df.target_pace_pct > cap).sum()):,}")
    write_quality_report(df, cap)
    print(f"  wrote {OUT.name}: {len(df):,} rows")
    return df


def _tortuosity(legs):
    """Path length over straight heel->toe distance. A lateral is nearly
    straight (ratio ~1); a high ratio means the 'lateral' doubles back --
    either genuinely weird drilling or a broken survey."""
    from .geometry import haversine_m
    straight = [max(haversine_m(r.heel_lat, r.heel_lon, r.toe_lat, r.toe_lon), 1.0)
                for r in legs.itertuples()]
    return legs.leg_length_m.to_numpy() / straight


def write_quality_report(df, cap):
    """reports/cohort_quality.md -- the anomalies a reader should know about,
    each attributed to its operator. Regenerated on every cohort build."""
    legs = pd.read_parquet(C.DATA_PROC / "legs.parquet")
    lines = ["# Cohort & geometry quality report", "",
             "Generated by `cohort.py`; regenerate rather than edit.", ""]

    t = df[df.target_pace_pct.notna()]
    clipped = t[t.target_pace_pct > cap].nlargest(10, "target_pace_pct")
    lines += [f"## Winsorization (baselines only)", "",
              f"Cap = train-era P{int(C.WINSOR_PCTL * 100)} = **{cap:.0f}%** of peer P50. "
              f"{int((t.target_pace_pct > cap).sum())} of {len(t):,} targeted wells clip. "
              f"Raw targets remain in `cohort.parquet` (`target_pace_pct`); baselines train on "
              f"`target_pace_pct_winsor`. Top clipped wells:", "",
              "| licence | operator | play | target % | pace boe/yr |", "|---|---|---|---|---|"]
    def txt(*vals):
        # NaN is truthy, so `nan or fallback` passes NaN through -- pd.isna
        # is the only honest emptiness test for these columns
        for v in vals:
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return str(v)
        return ""

    for r in clipped.itertuples():
        lines.append(f"| {r.well_key} | {txt(r.operator_name, r.operator_id)} | "
                     f"{txt(r.play_name, r.play_code)[:28]} | {r.target_pace_pct:,.0f} | "
                     f"{r.pace_boe_yr:,.0f} |")

    lines += ["", "## Suspicious geometries", ""]
    lg = legs.merge(df[["well_key", "operator_name"]], on="well_key", how="left")
    lg["tortuosity"] = _tortuosity(lg)

    twisted = lg[lg.tortuosity > 3.0]
    lines += [f"**Corkscrew laterals (path length > 3x heel-toe distance): "
              f"{len(twisted):,} of {len(lg):,} legs.** Either genuinely tortuous "
              f"drilling or a survey with disordered stations; excluded from nothing "
              f"yet, but closest-approach distances for these are unreliable.", ""]
    for r in twisted.nlargest(5, "tortuosity").itertuples():
        lines.append(f"- {r.uwi} ({r.operator_name or 'operator unknown'}): "
                     f"{r.leg_length_m:,.0f} m path, tortuosity {r.tortuosity:.1f}")

    huge = lg[lg.leg_length_m > 5000]
    lines += ["", f"**Single legs over 5,000 m: {len(huge):,}.** Plausible for modern "
              f"Duvernay/Montney laterals; listed so nobody mistakes them for errors:", ""]
    for r in huge.nlargest(5, "leg_length_m").itertuples():
        lines.append(f"- {r.uwi} ({r.operator_name or 'operator unknown'}): "
                     f"{r.leg_length_m:,.0f} m")

    shared_trunk = df[(df.n_legs >= 2) & (df.mean_interleg_m == 0)]
    lines += ["", f"**Multilaterals whose straight-leg spacing reads 0 m: "
              f"{len(shared_trunk):,}.** Known artifact: heel-toe segments of legs "
              f"sharing a trunk touch at the heel. Phase 3's polyline closest-approach "
              f"supersedes this number; do not use `mean_interleg_m` for these wells.", ""]

    # -- contamination candidates, surfaced by the operator attribution:
    # gas-STORAGE operators cycle injected gas, so their withdrawal "pace" is
    # not production; COMMINGLED pools report several wells as one stream.
    # Not excluded yet -- counted here for an explicit cohort decision.
    n_thermal = int(df.is_thermal.sum())
    lines_thermal = (f"**Thermal (SAGD/cyclic steam): EXCLUDED from targets "
                     f"(2026-08-26).** {n_thermal:,} licences flagged via "
                     f"WellStatusType SAGD/CYCL or self steam injection; they "
                     f"remain as neighbours, where their steam is context with "
                     f"the correct sign. Reasons: steam-driven pace is not "
                     f"comparable to primary recovery, the nearest neighbour is "
                     f"often the well's own injector twin at 0 m, and the play "
                     f"proxy is mostly missing in thermal country.")
    n_storage = int(df.is_storage.sum())
    n_commingled = int(df.is_commingled.sum())
    lines += ["## Cohort contamination", "", lines_thermal, "",
              f"**Gas-storage operators: EXCLUDED from targets (2026-08-26).** "
              f"{n_storage:,} storage licences remain in the cohort as "
              f"geometry/neighbours with NULL targets; before exclusion their "
              f"median target was ~4,155% of peer P50 -- withdrawal of injected "
              f"gas, not production.", "",
              f"**Commingled pools: EXCLUDED from targets (2026-08-26).** "
              f"{n_commingled:,} licences in COMMINGLED pools remain as "
              f"neighbours/score-only. Several wells report as one stream, so "
              f"the per-well pace label is an allocation artifact: before "
              f"exclusion their P99 target was 2,499% vs 1,270% for everyone "
              f"else, and they took 37% of the top-100 extreme targets from "
              f"15.6% of the population.", ""]

    floor = df[df.target_pace_pct.notna() & (df.n_bores_reported > df.n_legs)]
    lines += [f"**Targeted wells with unsurveyed bores (leg-count floor): "
              f"{len(floor):,}.** Their `n_legs` understates reality; "
              f"`n_bores_reported` carries the honest count.", ""]

    out = C.REPORTS / "cohort_quality.md"
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  wrote {out.relative_to(C.ROOT)}")
    return df


if __name__ == "__main__":
    build_cohort()
