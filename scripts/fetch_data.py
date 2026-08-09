#!/usr/bin/env python3
"""Reproducible public-data fetch. The raw data is gitignored (large, but freely re-downloadable),
so this script IS the provenance — every URL and licence is documented in data/README.md.

    uv run python scripts/fetch_data.py            # all sources
    uv run python scripts/fetch_data.py ab_st37    # one source

Sources (all public, licences in data/README.md):
  ab_st37   Alberta ST37 shapefiles (per-leg bottom holes + well-geometry lines) — free, OGL-Alberta
  ab_infra  Alberta Petrinex Well Infrastructure CSV (per-leg enumeration by UWI)
  sk_nvw    Saskatchewan GeoHub 'Non Vertical Wells' (Boss/Leg/Whipstock, per-leg bottom holes) — paged REST
  bc_survey British Columbia (BCER) directional surveys — single-lateral comparison set
"""

import json
import sys
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SIMPLE = {
    "ab_st37": (
        "https://static.aer.ca/prd/documents/sts/st37/ST37_Shapefiles.zip",
        RAW / "alberta" / "ST37_Shapefiles.zip",
    ),
    "ab_infra": (
        "https://www.petrinex.gov.ab.ca/publicdata/API/Files/AB/Infra/Well%20Infrastructure/CSV",
        RAW / "alberta" / "AB_Well_Infrastructure_CSV.zip",
    ),
    "bc_survey": (
        "https://iris.bc-er.ca/download/dir_survey_csv.zip",
        RAW / "bc" / "bcer_dir_survey_csv.zip",
    ),
}

SK_NVW = (
    "https://gis.saskatchewan.ca/arcgis/rest/services/Economy/Petroleum/MapServer/1/query"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n    -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"    done ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_sk_nvw() -> None:
    """Page the whole 'Non Vertical Wells' layer via the ArcGIS REST API (maxRecordCount 2000)."""
    dest = RAW / "saskatchewan" / "nonvertical_wells.geojson"
    dest.parent.mkdir(parents=True, exist_ok=True)
    features, offset = [], 0
    while True:
        params = {
            "where": "1=1", "outFields": "*", "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 2000, "outSR": 4326,
        }
        r = requests.get(SK_NVW, params=params, timeout=120)
        r.raise_for_status()
        page = r.json().get("features", [])
        if not page:
            break
        features.extend(page)
        offset += len(page)
        print(f"  SK non-vertical wells: {offset} features...")
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"    done ({len(features)} features -> {dest})")


def main(which=None):
    targets = which or (list(SIMPLE) + ["sk_nvw"])
    for name in targets:
        print(f"[{name}]")
        if name == "sk_nvw":
            fetch_sk_nvw()
        elif name in SIMPLE:
            _download(*SIMPLE[name])
        else:
            sys.exit(f"unknown source {name!r}; choices: {list(SIMPLE) + ['sk_nvw']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
