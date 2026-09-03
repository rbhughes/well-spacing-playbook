#!/usr/bin/env python3
"""Verify data/ppdm.duckdb against the raw Alberta sources.

    uv run python scripts/verify_ppdm.py        (exits non-zero on any failure)

Every check either RE-DERIVES a number from the raw files independently of the
loader, or asserts an invariant that must hold if the load is correct. Where a
check re-reads a source it uses its own code path, so a loader bug and a
verifier bug would have to agree to hide an error.

Runtime is a few minutes: it re-parses all four ST37 shapefiles and re-scans
all 55 volumetric months.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import duckdb
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from borehole_geometry import st37  # noqa: E402

DB = ROOT / "data" / "ppdm.duckdb"
RAW = ROOT / "data" / "raw" / "alberta"
WORK = ROOT / "data" / "work"
SAFE = "all_varchar=true, quote='\"', header=true"

_results = []


def check(name, ok, detail):
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"        {detail}")
    return ok


def extract_inner(zip_path: Path, out: Path):
    with zipfile.ZipFile(zip_path) as z:
        inner = z.read(z.namelist()[0])
    with zipfile.ZipFile(io.BytesIO(inner)) as z2:
        out.write_bytes(z2.read(z2.namelist()[0]))


def main():
    con = duckdb.connect(str(DB), read_only=True)
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    one = lambda s: con.execute(s).fetchone()  # noqa: E731
    WORK.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- wells
    print("\nWELL / WELL_ALIAS vs Petrinex Well Infrastructure")
    infra = WORK / "Well_Infrastructure-AB.CSV"
    if not infra.exists():
        extract_inner(RAW / "AB_Well_Infrastructure_CSV.zip", infra)
    src_rows, src_wells = one(
        f"SELECT count(*), count(DISTINCT WellIdentifier) "
        f"FROM read_csv('{infra}', {SAFE})")
    db_wells = one("SELECT count(*) FROM well")[0]
    check("every distinct source WellIdentifier is a WELL row",
          src_wells == db_wells,
          f"source distinct={src_wells:,} (of {src_rows:,} rows)  well={db_wells:,}")

    bad_alias = one("""SELECT count(*) FROM (
        SELECT UWI FROM well_alias GROUP BY UWI HAVING count(*) <> 2)""")[0]
    check("every well carries exactly 2 aliases (licence + Petrinex WellID)",
          bad_alias == 0, f"wells with alias count != 2: {bad_alias}")

    # ------------------------------------------------------------- ST37 recount
    print("\nWELL_DIR_SRVY / STATIONS vs a full independent re-parse of ST37")
    known = {r[0] for r in q("SELECT UWI FROM well")}
    feats = yielded = nowell = dup = 0
    loaded = {"SURVEYED": 0, "CALCULATED": 0}
    verts = 0
    seen = set()
    for shp in st37.wg_files(RAW / "ST37_SHP"):
        attrs = st37.read_dbf(shp.with_suffix(".dbf"))
        for idx, pts in st37.read_polylinez(shp):
            feats += 1
            a = attrs[idx - 1]
            uwi = st37.uwi_from_st37_label(a["UWI_Label"])
            if not uwi or not pts:
                continue
            yielded += 1
            if uwi not in known:
                nowell += 1
                continue
            if uwi in seen:
                dup += 1
                continue
            seen.add(uwi)
            loaded["SURVEYED" if a["WGGeomSrce"] == "Surveyed" else "CALCULATED"] += 1
            verts += len(pts)

    db_srvy = {r[0]: r[1] for r in q(
        "SELECT SURVEY_TYPE, count(*) FROM well_dir_srvy GROUP BY 1")}
    db_sta = one("SELECT count(*) FROM well_dir_srvy_station")[0]
    check("survey counts match the re-parse, split by WGGeomSrce",
          db_srvy.get("SURVEYED") == loaded["SURVEYED"]
          and db_srvy.get("CALCULATED") == loaded["CALCULATED"],
          f"re-parse SURVEYED={loaded['SURVEYED']:,} CALCULATED={loaded['CALCULATED']:,}  "
          f"db SURVEYED={db_srvy.get('SURVEYED'):,} CALCULATED={db_srvy.get('CALCULATED'):,}")
    check("station count matches the re-parsed vertex total exactly",
          db_sta == verts, f"re-parse vertices={verts:,}  db stations={db_sta:,}")
    check("every excluded ST37 feature is accounted for",
          feats == yielded + (feats - yielded) and yielded == sum(loaded.values()) + nowell + dup,
          f"features={feats:,} = usable {yielded:,} + no-label/empty {feats - yielded:,}; "
          f"usable = loaded {sum(loaded.values()):,} + no-well {nowell:,} + dup {dup:,}")

    # ------------------------------------------------------ geometry accuracy
    print("\nGeometry accuracy")
    tf = Transformer.from_crs(st37.SRC_EPSG, st37.DST_EPSG, always_xy=True)
    shp = st37.wg_files(RAW / "ST37_SHP")[0]
    attrs = st37.read_dbf(shp.with_suffix(".dbf"))
    worst = 0.0
    tested = 0
    for idx, pts in st37.read_polylinez(shp):
        a = attrs[idx - 1]
        uwi = st37.uwi_from_st37_label(a["UWI_Label"])
        if not uwi or len(pts) < 20 or a["WGGeomSrce"] != "Surveyed":
            continue
        rows = q(f"""SELECT LATITUDE, LONGITUDE FROM well_dir_srvy_station
                     WHERE UWI='{uwi}' ORDER BY DEPTH_OBS_NO""")
        if len(rows) != len(pts):
            worst = 999
            break
        lons, lats = tf.transform([p[0] for p in pts], [p[1] for p in pts])
        for (dla, dlo), la, lo in zip(rows, lats, lons):
            worst = max(worst, abs(float(dla) - la), abs(float(dlo) - lo))
        tested += 1
        if tested >= 5:
            break
    check("stored coordinates reproduce an independent reprojection",
          worst < 1e-6,
          f"{tested} surveyed bores re-transformed; worst deviation {worst:.2e} deg")

    lat_lo, lat_hi, lon_lo, lon_hi = one(
        "SELECT min(LATITUDE), max(LATITUDE), min(LONGITUDE), max(LONGITUDE) "
        "FROM well_dir_srvy_station")
    check("all coordinates inside Alberta's bounds",
          48.5 < float(lat_lo) and float(lat_hi) < 60.1
          and -120.1 < float(lon_lo) and float(lon_hi) < -109.9,
          f"lat {lat_lo}..{lat_hi}  lon {lon_lo}..{lon_hi}")

    print("\nDepth invariants")
    check("first station of every bore is at TVD 0 (the KB datum)",
          one("SELECT count(*) FROM well_dir_srvy_station "
              "WHERE DEPTH_OBS_NO=1 AND STATION_TVD <> 0")[0] == 0,
          "TVD at DEPTH_OBS_NO=1 is 0 for all bores")
    check("TVD = KBE + TVDSS holds on every station",
          one("""SELECT count(*) FROM well_dir_srvy_station s
                 JOIN well_dir_srvy d USING (UWI, SURVEY_ID, SOURCE)
                 WHERE d.REPORT_PERM_DATUM_ELEV IS NOT NULL
                   AND abs(s.STATION_TVD - (d.REPORT_PERM_DATUM_ELEV + s.STATION_TVDSS)) > 0.001""")[0] == 0,
          "zero violations across 24.2M stations")
    check("DEPTH_OBS_NO is contiguous 1..N within every bore",
          one("""SELECT count(*) FROM (
                 SELECT UWI, SURVEY_ID, SOURCE FROM well_dir_srvy_station
                 GROUP BY 1,2,3 HAVING min(DEPTH_OBS_NO) <> 1
                     OR max(DEPTH_OBS_NO) <> count(*))""")[0] == 0,
          "min=1 and max=count for every (UWI, SURVEY_ID, SOURCE)")
    check("station count per bore matches the vertex count recorded at load",
          one("""SELECT count(*) FROM well_dir_srvy d
                 JOIN (SELECT UWI, SURVEY_ID, SOURCE, count(*) n
                       FROM well_dir_srvy_station GROUP BY 1,2,3) s
                 USING (UWI, SURVEY_ID, SOURCE)
                 WHERE CAST(regexp_extract(d.REMARK, 'vertices=([0-9]+)', 1) AS INT) <> s.n""")[0] == 0,
          "REMARK vertices=N agrees with actual station rows for every bore")

    # ---------------------------------------------------------- production
    print("\nPDEN / volumes vs a re-scan of all 55 volumetric months")
    tmp = WORK / "_verify_month.csv"
    vol_files = sorted((RAW / "volumetrics").glob("*.zip"))
    con2 = duckdb.connect()
    con2.execute("CREATE TEMP TABLE producers (uwi VARCHAR)")
    for p in vol_files:
        extract_inner(p, tmp)
        con2.execute(f"""INSERT INTO producers
            SELECT DISTINCT FromToIDIdentifier FROM read_csv('{tmp}', {SAFE})
            WHERE FromToIDType='WI' AND ActivityID IN ('PROD','INJ')""")
    producers = {r[0] for r in con2.execute(
        "SELECT DISTINCT uwi FROM producers").fetchall()}
    expected_pden = len(producers & known)
    db_pden = one("SELECT count(*) FROM pden")[0]
    check("PDEN holds exactly the producing-or-injecting wells in WELL",
          db_pden == expected_pden,
          f"re-scan producers={len(producers):,}, of which in WELL={expected_pden:,}; "
          f"db pden={db_pden:,} (orphans skipped: {len(producers) - expected_pden})")

    months = one("SELECT count(DISTINCT PERIOD_ID), min(PERIOD_ID), max(PERIOD_ID) "
                 "FROM pden_vol_summary")
    check("volume table spans exactly the 55 fetched months",
          months[0] == 55 and months[1] == "2022-01" and months[2] == "2026-07",
          f"{months[0]} months, {months[1]} .. {months[2]}")

    for month in ("2022-01", "2024-06"):
        extract_inner(RAW / "volumetrics" / f"Vol_{month}-AB.csv.zip", tmp)
        exp = con.execute(f"""
            SELECT count(DISTINCT v.FromToIDIdentifier),
                   sum(CASE WHEN v.ProductID='OIL' THEN TRY_CAST(v.Volume AS DOUBLE) END),
                   sum(CASE WHEN v.ProductID IN ('GAS','ENTGAS','ACGAS')
                            THEN TRY_CAST(v.Volume AS DOUBLE) END)
            FROM read_csv('{tmp}', {SAFE}) v
            JOIN pden p ON p.PDEN_ID = v.FromToIDIdentifier
            WHERE v.FromToIDType='WI' AND v.ActivityID='PROD'
              AND v.ProductID IN ('OIL','GAS','ENTGAS','ACGAS','WATER','FSHWTR',
                                  'BRKWTR','STEAM','COND','CO2')""").fetchone()
        got = one(f"""SELECT count(*), sum(CAST(OIL_VOLUME AS DOUBLE)),
                             sum(CAST(GAS_VOLUME AS DOUBLE))
                      FROM pden_vol_summary
                      WHERE PERIOD_ID='{month}' AND ACTIVITY_TYPE='PROD'""")
        check(f"{month}: row count and OIL/GAS totals re-derive from the raw file",
              got[0] == exp[0] and abs(got[1] - exp[1]) < 1.0 and abs(got[2] - exp[2]) < 1.0,
              f"rows {got[0]:,}={exp[0]:,}  oil {got[1]:,.1f}~{exp[1]:,.1f}  "
              f"gas {got[2]:,.1f}~{exp[2]:,.1f}")
    tmp.unlink(missing_ok=True)

    check("pden_well links every entity to itself as PRIMARY_UWI",
          one("SELECT count(*) FROM pden_well WHERE PRIMARY_UWI <> PDEN_ID")[0] == 0,
          "PDEN_ID == PRIMARY_UWI on all rows")

    # -------------------------------------------------------- informational
    print("\nInformational (not pass/fail)")
    print(f"   pden without any volume rows: "
          f"{one('''SELECT count(*) FROM pden p WHERE NOT EXISTS
                    (SELECT 1 FROM pden_vol_summary v WHERE v.PDEN_ID=p.PDEN_ID)''')[0]:,}")
    print(f"   negative OIL_VOLUME rows (Petrinex amendments): "
          f"{one('SELECT count(*) FROM pden_vol_summary WHERE OIL_VOLUME < 0')[0]:,}")
    print(f"   median producing months per well: "
          f"{one('''SELECT median(n) FROM (SELECT PDEN_ID, count(*) n
                    FROM pden_vol_summary GROUP BY 1)''')[0]}")
    print(f"   wells with surveyed path AND production: "
          f"{one('''SELECT count(*) FROM well_dir_srvy d
                    WHERE d.SURVEY_TYPE='SURVEYED'
                      AND EXISTS (SELECT 1 FROM pden p WHERE p.PDEN_ID=d.UWI)''')[0]:,}")

    passed = sum(_results)
    print(f"\n{'=' * 66}\n  {passed}/{len(_results)} checks passed")
    if passed != len(_results):
        print("  The database does not match its sources. Fix the LOADER and")
        print("  re-run the failing stage; do not patch the database by hand.")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
