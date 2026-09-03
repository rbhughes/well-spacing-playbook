#!/usr/bin/env python3
"""Reproducible public-data fetch. Alberta only.

The raw data is gitignored (large, but freely re-downloadable), so this script IS the provenance —
every URL and licence is documented in data/README.md.

    uv run python scripts/fetch_data.py            # all sources
    uv run python scripts/fetch_data.py ab_st37    # one source
    uv run python scripts/fetch_data.py ab_vol     # production history (55 monthly files)

Sources (all public, licences in data/README.md):
  ab_st37   Alberta ST37 shapefiles — surveyed 3-D well geometry (PolyLineZ) + bottom/surface
            holes. WGGeomSrce distinguishes 'Surveyed' (real directional survey, median 57
            stations) from 'Calculated' (2-vertex stick, no survey). Only 'Surveyed' is usable
            for closest-approach work.
  ab_infra  Petrinex Well Infrastructure CSV — per-event well headers. Bridges ST37 geometry to
            volumetrics: WellIdentifier here == FromToIDIdentifier in the volumetric files.
  ab_vol    Petrinex Volumetric Data — monthly OIL/GAS/WATER volume per WELL EVENT. Reported at
            the battery, with FromToIDType='WI' naming the producing well. One file per month;
            the free public window is a rolling ~5 years (see VOL_FIRST/VOL_LAST).
"""

import json
import sys
from datetime import date
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
}

# Petrinex publishes one zip per production month and keeps a rolling public window. Verified
# 2026-08-23: 2021-12 and earlier return 404; 2022-01 through 2026-07 return data (~7.9 MB each).
# Re-probe before assuming these bounds still hold -- the window slides forward.
VOL_URL = "https://www.petrinex.gov.ab.ca/publicdata/API/Files/AB/Vol/{month}/CSV"
VOL_FIRST = (2022, 1)
VOL_LAST = (2026, 7)


def _download(url: str, dest: Path, quiet: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"  {url}\n    -> {dest}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if not quiet:
        print(f"    done ({dest.stat().st_size / 1e6:.1f} MB)")


def _months(first: tuple[int, int], last: tuple[int, int]):
    y, m = first
    while (y, m) <= last:
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_ab_vol() -> None:
    """Monthly volumetric files. Skips months already on disk so it is safely resumable."""
    out = RAW / "alberta" / "volumetrics"
    out.mkdir(parents=True, exist_ok=True)
    months = list(_months(VOL_FIRST, VOL_LAST))
    print(f"  {len(months)} months, {VOL_FIRST[0]}-{VOL_FIRST[1]:02d} to "
          f"{VOL_LAST[0]}-{VOL_LAST[1]:02d} (~8 MB each)")
    got = skipped = 0
    for i, month in enumerate(months, 1):
        dest = out / f"Vol_{month}-AB.csv.zip"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            _download(VOL_URL.format(month=month), dest, quiet=True)
            got += 1
        except requests.HTTPError as e:
            # A 404 means the rolling window moved; do not fail the whole run.
            print(f"    {month}: unavailable ({e.response.status_code}) -- skipping")
            dest.unlink(missing_ok=True)
            continue
        if i % 10 == 0 or i == len(months):
            print(f"    {i}/{len(months)} months ({got} fetched, {skipped} already present)")
    total = sum(p.stat().st_size for p in out.glob("*.zip"))
    print(f"    done ({got} fetched, {skipped} skipped, {total / 1e6:.0f} MB in {out})")


def probe_vol_window() -> None:
    """Report the actual public window, since it slides. Run this before a full ab_vol fetch."""
    def has_data(month: str) -> bool:
        with requests.get(VOL_URL.format(month=month), stream=True, timeout=90) as r:
            if r.status_code != 200:
                return False
            return next(r.iter_content(8), b"")[:2] == b"PK"

    today = date.today()
    latest = None
    for back in range(0, 12):
        idx = today.year * 12 + today.month - 1 - back
        month = f"{idx // 12}-{idx % 12 + 1:02d}"
        if has_data(month):
            latest = month
            break
    if latest is None:
        print("  no volumetric months available -- endpoint or network problem")
        return
    # Walk back from the LATEST available month, not from today: the current
    # month is never published yet, so starting there stops the scan at once.
    ly, lm = (int(v) for v in latest.split("-"))
    earliest = latest
    for back in range(1, 96):
        idx = ly * 12 + lm - 1 - back
        month = f"{idx // 12}-{idx % 12 + 1:02d}"
        if not has_data(month):
            break
        earliest = month
    print(f"  public volumetric window: {earliest} .. {latest}")
    print(f"  script constants:         {VOL_FIRST[0]}-{VOL_FIRST[1]:02d} .. "
          f"{VOL_LAST[0]}-{VOL_LAST[1]:02d}")


# --- AGS Geological Framework of Alberta v3: curated surface rasters -------
# Each hub dataset hides a downloadable GeoTIFF behind its description link:
#   DCAT catalog -> Image Service item -> description href -> raster item /data
# Curated to the plays in the cohort; ~8 MB each. Licence: AER/AGS -- publish
# derivatives with attribution, do not re-host raw (same posture as ST37).
AGS_SURFACES = {
    "Elevation Surface Top 01 QNgPg sediment": "top_sediment",
    "Elevation Surface Top 28 K Ca": "top_cardium",
    "Elevation Surface Top 39 K BoI V Pl": "top_viking",
    "Elevation Surface Top 45 K SR": "top_spirit_river",
    "Elevation Surface Top 46 K undiv Mn": "top_mannville",
    "Elevation Surface Top 47 K C excl C w": "top_clearwater",
    "Elevation Surface Top 56_62 T Mo complete": "top_montney",
    "Elevation Surface Base 56_62 T Mo complete": "base_montney",
    "Elevation Surface Top 90 PreC": "top_precambrian",
    # direct thickness grids -- the better rock-volume proxy where published
    "Vertical Thickness 56_62 T Mo complete": "vt_montney",
    "Vertical Thickness 46 K undiv Mn": "vt_mannville",
    "Vertical Thickness 47 K C excl C w": "vt_clearwater",
    "Vertical Thickness 45 K SR": "vt_spirit_river",
    "Vertical Thickness 28 K Ca": "vt_cardium",
}


def fetch_ags():
    import re
    out = RAW / "ags"
    out.mkdir(parents=True, exist_ok=True)
    H = {"User-Agent": "Mozilla/5.0"}
    manifest = {}
    for title, short in AGS_SURFACES.items():
        dest = out / f"{short}.tif"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {short}: already present")
            continue
        q = {"f": "json", "num": 3,
             "q": f'title:"{title}" AND type:"Image Service"'}
        res = requests.get("https://www.arcgis.com/sharing/rest/search",
                           params=q, timeout=60, headers=H).json()["results"]
        hit = next((x for x in res if x["title"] == title), None)
        if hit is None:
            print(f"  {short}: NOT FOUND for {title!r}")
            continue
        item = requests.get(
            f"https://www.arcgis.com/sharing/rest/content/items/{hit['id']}",
            params={"f": "json"}, timeout=60, headers=H).json()
        links = re.findall(r"href=['\"]([^'\"]*item\.html\?id=[0-9a-f]+)['\"]",
                           item.get("description", ""))
        if not links:
            print(f"  {short}: no raster link in description")
            continue
        rid = links[0].split("id=")[-1]
        _download("https://www.arcgis.com/sharing/rest/content/items/"
                  f"{rid}/data", dest, quiet=True)
        manifest[short] = {"title": title, "service_item": hit["id"],
                           "raster_item": rid, "bytes": dest.stat().st_size}
        print(f"  {short}: {dest.stat().st_size/1e6:.1f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  wrote {out}/manifest.json")


def main(which=None):
    targets = which or (list(SIMPLE) + ["ab_vol", "ags"])
    for name in targets:
        print(f"[{name}]")
        if name == "ab_vol":
            fetch_ab_vol()
        elif name == "ags":
            fetch_ags()
        elif name == "probe_vol":
            probe_vol_window()
        elif name in SIMPLE:
            _download(*SIMPLE[name])
        else:
            sys.exit(
                f"unknown source {name!r}; choices: "
                f"{list(SIMPLE) + ['ab_vol', 'ags', 'probe_vol']}"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or None)
