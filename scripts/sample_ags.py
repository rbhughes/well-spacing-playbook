#!/usr/bin/env python3
"""Sample the AGS Geological Framework grids at every well's surface location.

    uv run python scripts/sample_ags.py  ->  data/processed/ags_surfaces.parquet

One row per licence: the elevation/thickness of each fetched surface at the
well. NoData (well outside a unit's extent) stays NaN -- absence of a unit is
information, not a gap to impute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from borehole_geometry import config as C  # noqa: E402

AGS = ROOT / "data" / "raw" / "ags"

con = duckdb.connect(str(ROOT / "data" / "ppdm.duckdb"), read_only=True)
wells = con.execute("""
    SELECT l.well_key,
           median(CAST(s.LONGITUDE AS DOUBLE)) lon,
           median(CAST(s.LATITUDE AS DOUBLE)) lat
    FROM read_parquet('data/processed/legs.parquet') l
    JOIN well_dir_srvy_station s
      ON s.UWI = l.uwi AND s.SOURCE = 'AER' AND s.DEPTH_OBS_NO = 1
    GROUP BY 1
""").df()
print(f"{len(wells):,} wells to sample")

out = wells[["well_key"]].copy()
for tif in sorted(AGS.glob("*.tif")):
    name = tif.stem
    with rasterio.open(tif) as r:
        tf = Transformer.from_crs("EPSG:4269", r.crs, always_xy=True)
        xs, ys = tf.transform(wells.lon.values, wells.lat.values)
        vals = np.array([v[0] for v in r.sample(zip(xs, ys))], dtype=float)
        nod = r.nodata
        if nod is not None:
            vals[np.isclose(vals, nod)] = np.nan
        vals[np.abs(vals) > 1e10] = np.nan
    out[name] = vals
    print(f"  {name:20} coverage {np.isfinite(vals).mean()*100:5.1f}%  "
          f"range [{np.nanmin(vals):8.1f}, {np.nanmax(vals):8.1f}]")

out.to_parquet(C.DATA_PROC / "ags_surfaces.parquet", index=False)
print(f"wrote data/processed/ags_surfaces.parquet ({len(out):,} rows)")
