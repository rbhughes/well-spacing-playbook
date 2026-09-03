# Data pipeline: public Alberta well data → PPDM 3.9 subset in DuckDB

Everything the ML work consumes lives in one file, `data/ppdm.duckdb`, built
entirely from free Alberta regulator data by three scripts. This document is
the map: where each byte came from, what was done to it, and how we know the
result is right. Nothing here requires credentials, payment, or manual steps.

```text
  AER ST37 shapefiles ─────────┐  (3-D wellbore geometry, PolyLineZ)
  Petrinex Well Infrastructure ┼──> scripts/fetch_data.py ──> data/raw/alberta/
  Petrinex Volumetrics (x55) ──┘                                   │
                                                                   ▼
  PPDM39_PG.zip (licensed) ──> scripts/build_ppdm_schema.py ──> sql/ppdm_min.sql
                                                                   │
                                            scripts/load_ppdm.py ──┴──> data/ppdm.duckdb
                                            scripts/verify_ppdm.py ───── (proves it)
```

## 1. Sources

| source | what it provides | size | licence |
| --- | --- | --- | --- |
| [AER ST37](https://static.aer.ca/prd/documents/sts/st37/ST37_Shapefiles.zip) "List of Wells" shapefiles | Wellbore geometry as **PolyLineZ (true 3-D)**. `WGGeomSrce` flags each bore `Surveyed` (real directional survey, median 123 stations) or `Calculated` (2-vertex stick) | 534 MB | OGL-Alberta (attribution; do **not** re-host raw) |
| [Petrinex Well Infrastructure](https://www.petrinex.gov.ab.ca/publicdata) | Per-event well headers: `WellIdentifier`, licence, operator, status | 63 MB | Petrinex terms: non-commercial use; Crown copyright acknowledged; commercial use needs prior consent |
| Petrinex Volumetric Data | Monthly OIL/GAS/WATER/COND volume **per well event**. One zip per month; the free archive is a **rolling ~5-year window** (2022-01 .. 2026-07 when fetched, 2026-08-23) | 55 × ~8 MB | same |
| PPDM 3.9 DDL (`PPDM39_PG.zip`, members' download) | The official schema our subset is generated from | 1.7 MB | PPDM IP, all rights reserved — **not committed**; supply your own copy |
| [AGS Geological Framework of Alberta v3](https://gfa-v3-ags-aer.hub.arcgis.com/) | 13 curated surface rasters (formation tops + vertical thicknesses) as GeoTIFF, sampled at every well as the rock-quality control | ~140 MB | AER/AGS — publish derivatives with attribution, do not re-host raw |

Saskatchewan and BC were evaluated and dropped (2026-08-23): SK publishes
2-point 2-D sticks with no survey and no production volumes; BC was only ever a
single-lateral comparison set. Details in `data/README.md`.

## 2. Download — `scripts/fetch_data.py`

```bash
uv run python scripts/fetch_data.py             # all four sources (incl. 'ags')
uv run python scripts/fetch_data.py probe_vol   # re-check the rolling window
```

The volumetric fetch is resumable (skips months already on disk) and tolerates
the window sliding. **The window is the binding constraint on the ML target**:
a well's early, highest-rate production is only observable if it came on
stream after 2022-01, so `config.FIRST_PROD_MIN` is pinned to that date with an
assert.

## 3. Schema — `scripts/build_ppdm_schema.py`

```bash
uv run python scripts/build_ppdm_schema.py --apply    # -> data/ppdm.duckdb (empty)
```

19 of PPDM 3.9's 2,688 tables, 1,222 columns, 31 enforced foreign keys. Three
rules, each chosen deliberately:

- **Tables are minimised; columns are not.** Every kept table carries its full
  PPDM column list — DuckDB is columnar, so unused NULL columns are nearly
  free, while dropped columns would break any real PPDM query.
- **Nothing is typed from memory.** Every identifier is extracted from the
  official DDL by script; the generated SQL is never hand-edited. PPDM's
  column names are full of near-misses (`substance` has no `LONG_NAME`;
  `well` has no `SOURCE`), so generation beats transcription.
- **FKs are enforced only within the subset.** The full FK closure from these
  19 tables reaches 65 more. Unenforced FKs are kept as comments in the SQL so
  the omission is visible.

Tables: `well`, `well_alias`, `well_dir_srvy`, `well_dir_srvy_station`,
`pden`, `pden_well`, `pden_vol_summary`, `business_associate`, `field`,
`pool`, and 9 seeded reference tables.

## 4. Import — `scripts/load_ppdm.py`

```bash
uv run python scripts/load_ppdm.py         # stages 0-7 in order
uv run python scripts/load_ppdm.py 7       # re-run one stage (all idempotent)
```

| stage | target | source | rows |
| --- | --- | --- | --- |
| 0 | 9 reference tables | seeded codes (upserted) | 14 |
| 1 | `business_associate` | Petrinex operators | 683 |
| 2 | `well` — one row per well **event** | Well Infrastructure | 664,075 |
| 3 | `well_alias` — AER licence + Petrinex WellID | Well Infrastructure | 1,328,150 |
| 4 | `well_dir_srvy` — one per ST37 bore (SURVEYED 189,105 / CALCULATED 343,448) | ST37 PolyLineZ | 532,553 |
| 5 | `well_dir_srvy_station` — one per vertex | ST37 PolyLineZ | 24,804,008 |
| 6 | `pden` + `pden_well` — producing OR injecting entities | 55 volumetric months | 176,216 |
| 7 | `pden_vol_summary` — well-month volumes per activity (PROD 7,590,408 + INJ 574,849) | 55 volumetric months | 8,165,257 |
| 8 | `field` + `pool` reference tables | Well Infrastructure | 900 + 9,796 |

### The decisions that matter

**UWI = the Petrinex `WellIdentifier`** (e.g. `100071901015W400`). Production
is the scarcer join, so the key production is published under wins; the ST37
label and AER licence live in `WELL_ALIAS`. Manufacturing that key from the
ST37 label is the trickiest step in the pipeline and has its own section (§5).

**`SOURCE` is per-row provenance**: `AER` for ST37 geometry, `PETRINEX` for
headers and volumes. Neither regulator prescribes a value; PPDM defines the
column as "the agency designated as the source of information for this row".

**CRS: EPSG:3400 → EPSG:4269, recorded in the database.** The shapefiles are
NAD83 / Alberta 10-TM (Forest); coordinates are stored as NAD83 geographic
(*not* WGS84 — they differ by ~1 m in Alberta, which matters for a spacing
model) with `GEOG_COORD_SYS_ID='EPSG:4269'` on every survey header.

**Depths are derived, not copied.** The shapefile Z ordinate is *elevation*
(verified: Z at the first vertex equals KB elevation exactly). PPDM wants
depths, positive down: `STATION_TVD = KBE − Z`, `STATION_TVDSS = −Z`, KBE in
`REPORT_PERM_DATUM_ELEV`. Invariant `TVD = KBE + TVDSS` holds on all 24.2M
rows.

**Injection is loaded alongside production.** Petrinex reports INJ activity
per well-month (water, STEAM — 23.2M m³ in 2024-06 alone — gas, CO₂); a
neighbouring injector supports pressure rather than stealing it, so dropping
INJ (an early version did) discards the opposite-signed half of the
interference physics. `ACTIVITY_TYPE` is part of the PPDM key, so PROD and
INJ rows coexist; steam lands in `WATER_VOLUME` with `PRIMARY_PRODUCT='STEAM'`.

**Both `Surveyed` and `Calculated` bores are loaded**, flagged in
`SURVEY_TYPE`. The database records what the regulator published; filtering is
the cohort query's job — but see the `CALCULATED` depth limitation in §7
before using those bores for anything geometric.

**CSV reading is never lenient.** All Petrinex files are read with
`read_csv(..., all_varchar=true, quote='"', header=true)` and `TRY_CAST`,
never `ignore_errors`. Two properties of the files force this: confidential
values are masked with the literal `***`, and facility names may contain
quoted commas (`"Joffre 8-25,12-20,13-30,13-18-37-26"`) that DuckDB's
auto-detection does not handle — a lenient reader drops such rows without
notice.

## 5. Linking geometry to production (the UWI join)

ST37 geometry and Petrinex production share no common column. ST37 identifies
each bore by its DLS location label; Petrinex keys wells — and every monthly
volume row — by `WellIdentifier`. Both are encodings of the same 16-character
UWI (AER's scheme, included at `data/raw/reference/WCSB_UWI_scheme.txt`), so
the pipeline derives the Petrinex key from the ST37 label
(`st37.uwi_from_st37_label`):

```text
ST37 label:            LE / LSD - SC - TWP - RG W M / ES
                       00 / 07  - 19 - 010 - 15 W 4 / 0

Petrinex identifier:   S  LE  LSD  SC  TWP  RG  WM  ES
                       1  00  07   19  010  15  W4  00    ->  100071901015W400
```

Two components deserve care when reading the label:

**The location exception (`LE`, leading) is alphanumeric**, and its families
carry regulatory meaning:

| LE family | meaning |
| --- | --- |
| `00, 02–99` (01 unused) | conventional oil/gas — Nth well in the legal subdivision |
| `AA–HZ` (no I or O) | oil sands evaluation holes |
| `F*` / `O*` | water source wells |
| `S*` / `W*` | bottomhole in a road allowance south/west of the LSD |

**The event sequence (`ES`, trailing) is the granularity production is
reported at** — Petrinex volumes attach to a specific well event, which is why
`well` holds one row per event rather than per licence. Legs of a multilateral
are grouped by licence (`Well_LicNo`), not by UWI: the UWI describes the
*bottomhole*, and each leg has its own.

Coverage of the derived join, re-checked on every run of `verify_ppdm.py`:

```text
532,623 ST37 features = 532,553 linked to a Petrinex well  (99.99%)
                      +      70 with no Well Infrastructure row (carried as a known remainder)
                      +       0 duplicates, 0 unexplained
```

## 6. Verification — `scripts/verify_ppdm.py`

```bash
uv run python scripts/verify_ppdm.py     # exits non-zero on any failure
```

Every check either **re-derives a number from the raw files independently of
the loader** (full re-parse of all four shapefiles; re-scan of all 55
volumetric months; independent reprojection of sample bores) or asserts an
invariant that must hold if the load is correct (depth algebra on all 24.8M
stations, station contiguity, alias cardinality, month-level volume totals
reconciled against the raw CSVs). Current status: **all checks pass** — run it
rather than trusting this sentence.

## 7. Known limitations — read before modelling

- **`Calculated` bore depths are fabricated.** 99.6% of them have max TVD
  exactly equal to `FTDepth`, which is *measured* depth — a horizontal well
  drawn as a vertical shaft (the apparent "8,126 m well" is this). Use
  `SURVEY_TYPE='SURVEYED'` only. Surveyed depths are sane: median 783 m,
  p99 3,650 m, max 6,033 m — all plausible for Alberta.
- **Production is per well event, never per leg.** Nobody meters individual
  legs; intra-well interference must be inferred.
- **The volumetric window slides.** Re-run `probe_vol` and re-state the window
  alongside any result.
- **A handful of entities (26) have no volume rows** (report-only linkages);
  harmless but present.
- **Licensing:** raw Alberta files must not be re-hosted; Petrinex commercial
  use requires prior consent; the PPDM DDL and the generated `ppdm_min.sql`
  are not committed. A fresh clone needs its own `PPDM39_PG.zip`.

## 8. Reproduce from scratch

```bash
uv sync
uv run python scripts/fetch_data.py            # ~1 GB download
# place PPDM39_PG.zip in the repo root (PPDM members' download)
uv run python scripts/build_ppdm_schema.py --apply
uv run python scripts/load_ppdm.py             # ~10 min
uv run python scripts/verify_ppdm.py           # ~5 min, must exit 0
```

Headline numbers once loaded: **79,655 wells with both a surveyed 3-D path
and production history** (grown from 71,805 by the alphanumeric location-
exception fix in §5 and by admitting injector-only wells). Downstream, after
the volumetric-window, geometry, and data-hygiene gates (storage, thermal,
commingled excluded from labels), **10,200 wells carry a trainable pace
target** — the modelling cohort. The layers between those two numbers are
each documented in `reports/cohort_quality.md`.
