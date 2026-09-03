#!/usr/bin/env python3
"""Generate a minimal PPDM 3.9 schema for DuckDB from the official PG DDL.

    uv run python scripts/build_ppdm_schema.py          # -> sql/ppdm_min.sql
    uv run python scripts/build_ppdm_schema.py --apply  # also builds data/ppdm.duckdb

WHY A GENERATOR RATHER THAN A HAND-WRITTEN SCHEMA
-------------------------------------------------
PPDM 3.9 is 2,688 tables. Every table and column name emitted here is COPIED
from the official DDL inside PPDM39_PG.zip, never retyped or recalled -- a
wrong column name would silently poison every downstream query. Re-run this
script if the subset changes; do not hand-edit sql/ppdm_min.sql.

WHAT "MINIMAL" MEANS HERE
-------------------------
Tables are minimised; COLUMNS ARE NOT. Each table carries its complete PPDM
column list, because DuckDB is columnar and unused NULL columns cost almost
nothing -- whereas dropping columns would break any real PPDM query. So these
19 tables are PPDM-faithful; the 2,669 omitted simply do not exist here.

Primary keys are enforced. Foreign keys are enforced ONLY within the subset
(well_dir_srvy_station -> well_dir_srvy -> well, pden_vol_summary -> pden,
and so on). PPDM's full FK closure from these 19 tables reaches 65 more
tables, which reach more again; enforcing it would mean implementing most of
PPDM to store three sources. Unenforced FKs are written into the SQL as
comments so the omission stays visible rather than silent.
"""

import argparse
import re
import zipfile
from pathlib import Path

TABLES = {
    # --- well identity and geometry --------------------------------------
    "well": "Well header: one row per UWI. Surface/bottom location, TD, dates.",
    "well_alias": "Alternate identifiers -- AER licence number, Petrinex "
                  "WellIdentifier, ST37 UWI label. Keeps source keys joinable "
                  "without polluting WELL.",
    "well_dir_srvy": "Directional survey header: one row per survey per well.",
    "well_dir_srvy_station": "Survey stations -- the ST37 PolyLineZ vertices. "
                             "MD/TVD/lat/long/inclination/azimuth per point.",
    # --- production -------------------------------------------------------
    "pden": "Production entity. Petrinex reports per well event, so "
            "PDEN_SUBTYPE='WELL'.",
    "pden_well": "Links a production entity to its well (PRIMARY_UWI).",
    "pden_vol_summary": "Monthly produced volumes by substance.",
    # --- context ----------------------------------------------------------
    "business_associate": "Operators / licensees.",
    "field": "AER/Petrinex field codes.",
    "pool": "AER/Petrinex pool-deposit codes.",
    # --- reference tables we actually populate ----------------------------
    "r_well_status": "Well status codes.",
    "r_well_class": "Well class codes.",
    "r_dir_srvy_type": "Survey type -- carries ST37 Surveyed vs Calculated.",
    "r_dir_srvy_point_type": "Station point type.",
    "r_activity_type": "PDEN_VOL_SUMMARY.ACTIVITY_TYPE (PROD, INJ, ...).",
    "r_period_type": "PDEN_VOL_SUMMARY.PERIOD_TYPE (MONTH).",
    "r_volume_method": "How the volume was determined.",
    "r_pden_amend_reason": "Amendment reason for restated volumes.",
    "substance": "Product codes -- OIL, GAS, WATER, COND.",
}

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "PPDM39_PG.zip"
OUT = ROOT / "sql" / "ppdm_min.sql"
DB = ROOT / "data" / "ppdm.duckdb"


def _read(zf, suffix):
    name = next(n for n in zf.namelist() if n.upper().endswith(suffix))
    return zf.read(name).decode("latin1")


def parse(zf):
    tab, pk, fk = (_read(zf, s) for s in ("PPDM39_TAB.SQL", "PPDM39_PK.SQL", "PPDM39_FK.SQL"))
    cols = {
        m.group(1).lower(): m.group(2)
        for m in re.finditer(r"CREATE TABLE\s+(\w+)\s*\((.*?)\n\)\s*;", tab, re.S | re.I)
    }
    pks = {
        m.group(1).lower(): (m.group(2), [c.strip() for c in m.group(3).split(",") if c.strip()])
        for m in re.finditer(
            r"ALTER TABLE (\w+) ADD CONSTRAINT (\w+) PRIMARY KEY\s*\((.*?)\)\s*;",
            pk, re.S | re.I)
    }
    fks = {}
    for m in re.finditer(
        r"ALTER TABLE (\w+)\s+ADD CONSTRAINT (\w+) FOREIGN KEY\s*\((.*?)\)"
        r"\s*REFERENCES\s+(\w+)\s*\((.*?)\)", fk, re.S | re.I):
        fks.setdefault(m.group(1).lower(), []).append((
            m.group(2),
            [c.strip() for c in m.group(3).split(",") if c.strip()],
            m.group(4).lower(),
            [c.strip() for c in m.group(5).split(",") if c.strip()],
        ))
    return cols, pks, fks


def pg_to_duckdb(coldef):
    """PPDM's PG types are DuckDB-compatible; normalise NUMERIC -> DECIMAL."""
    return re.sub(r"\bNUMERIC\s*\(", "DECIMAL(", coldef, flags=re.I)


def build():
    with zipfile.ZipFile(ZIP) as zf:
        cols, pks, fks = parse(zf)
    missing = [t for t in TABLES if t not in cols]
    if missing:
        raise SystemExit(f"not present in the PPDM DDL: {missing}")

    out = [
        "-- PPDM 3.9 minimal subset for DuckDB.",
        "-- GENERATED by scripts/build_ppdm_schema.py from PPDM39_PG.zip. Do not hand-edit.",
        f"-- {len(TABLES)} of 2,688 PPDM tables. Column definitions verbatim from the official",
        "-- DDL; primary keys enforced; foreign keys enforced only within the subset.",
        "",
    ]
    for t, why in TABLES.items():
        out.append(f"-- {t}: {why}")
        out.append(f"CREATE TABLE IF NOT EXISTS {t} (")
        out.append(pg_to_duckdb(cols[t].rstrip().rstrip(",")))
        if t in pks:
            out.append(f"\t, CONSTRAINT {pks[t][0]} PRIMARY KEY ({', '.join(pks[t][1])})")
        for name, c, ref, rc in fks.get(t, []):
            if ref in TABLES and ref != t:
                out.append(f"\t, CONSTRAINT {name} FOREIGN KEY ({', '.join(c)}) "
                           f"REFERENCES {ref} ({', '.join(rc)})")
        out.append(");")
        ext = sorted({ref for _, _, ref, _ in fks.get(t, []) if ref not in TABLES})
        if ext:
            out.append(f"-- unenforced PPDM FKs: {t} -> {', '.join(ext)}")
        out.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(TABLES)} tables, {OUT.stat().st_size/1024:.0f} KB)")
    return OUT


def apply_schema(path):
    import duckdb
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))
    # Strip comment lines BEFORE splitting: the header comments contain
    # semicolons, which would otherwise be read as statement terminators.
    body = "\n".join(l for l in path.read_text().splitlines()
                     if not l.lstrip().startswith("--"))
    stmts = [s.strip() for s in body.split(";") if s.strip()]
    pending = stmts
    for _ in range(4):                      # retry: FK targets must exist first
        failed = []
        for s in pending:
            try:
                con.execute(s)
            except Exception:
                failed.append(s)
        pending = failed
        if not pending:
            break
    if pending:
        raise SystemExit(f"{len(pending)} statements failed; first:\n{pending[0][:400]}")
    n = con.execute("SELECT count(*) FROM duckdb_tables()").fetchone()[0]
    c = con.execute("SELECT count(*) FROM duckdb_columns()").fetchone()[0]
    print(f"built {DB.relative_to(ROOT)}: {n} tables, {c} columns")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="also create data/ppdm.duckdb")
    args = ap.parse_args()
    p = build()
    if args.apply:
        apply_schema(p)
