"""Readers for the AER ST37 shapefiles. No GDAL/geopandas -- these are the two
simplest shapefile formats and the whole point is to avoid a heavy dependency.

Two facts about this data, both verified 2026-08-23 (see data/README.md):

  * The geometry is shapefile type 13, PolyLineZ -- genuinely 3-D.
  * `WGGeomSrce` flags each bore `Surveyed` (a real directional survey, median
    57 stations) or `Calculated` (a 2-vertex stick). The flag matches vertex
    counts exactly, so trust the flag rather than inferring from geometry.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

# The shapefiles' own .prj is "NAD83 / Alberta 10-TM (Forest)", which pyproj
# confirms is exactly EPSG:3400 (CRS.equals() -> True). Use the EPSG code
# rather than a hand-written proj string: a bare "+ellps=GRS80" carries no
# datum, so PROJ cannot apply a datum shift and silently performs a null
# transformation.
SRC_EPSG = 3400          # NAD83 / Alberta 10-TM (Forest) -- the ST37 shapefiles

# Target EPSG:4269 (NAD83 geographic), NOT 4326 (WGS84). The source datum is
# NAD83, and in Alberta NAD83 and WGS84 differ by roughly 1-1.5 m. Asking for
# 4326 from an undatum'd proj string returned bit-identical numbers to 4269 --
# i.e. NAD83 values wearing a WGS84 label. For a project whose whole subject is
# metre-scale spacing between wellbores, an unlabelled 1 m datum error is not
# acceptable. Label the data as what it is; convert deliberately if WGS84 is
# ever needed.
DST_EPSG = 4269          # NAD83 geographic (lat/long)

# Stored in PPDM WELL_DIR_SRVY.GEOG_COORD_SYS_ID, which is defined as the CRS
# "defining the Geodetic Datum of the WELL_DIR_SRVY_STATION LATITUDE and
# LONGITUDE values".
GEOG_COORD_SYS_ID = f"EPSG:{DST_EPSG}"

# Canadian DLS UWI display format: LE/LSD-SEC-TWP-RGEWM/ES
#   LE = location exception (LEADING), ES = event sequence (TRAILING).
# Getting these two the wrong way round drops the ST37<->Petrinex join from
# 100% to 68%.
#
# The location exception is ALPHANUMERIC, not numeric: F1, AA, AB, W0, S0 and
# friends are common (Petrinex alone carries 31,626 wells at exception AA). An
# earlier digits-only pattern silently discarded 62,482 bores -- 11.7% of the
# province -- and the loss was invisible because unparseable labels were simply
# skipped. Caught by verify_ppdm.py's reconciliation, which refuses to lump
# "excluded" and "unexplained" together. Event sequences are all digits
# (checked across every label in ST37).
_LABEL = re.compile(r"^([A-Z0-9]{2})/(\d{2})-(\d{2})-(\d{3})-(\d{2})(W\d)/(\d)$")


def uwi_from_st37_label(label: str) -> str | None:
    """ST37 `UWI_Label` -> the Petrinex `WellIdentifier` used as PPDM UWI.

    '00/07-19-010-15W4/0' -> '100071901015W400'
    """
    m = _LABEL.match(label.strip())
    if not m:
        return None
    le, lsd, sec, twp, rge, mer, es = m.groups()
    return f"1{le}{lsd}{sec}{twp}{rge}{mer}{es:0>2}"


def read_dbf(path: Path) -> list[dict]:
    """Minimal dBase III reader: enough for ST37's 7 character/float fields."""
    with open(path, "rb") as f:
        nrec, hlen, rlen = struct.unpack("<IHH", f.read(32)[4:12])
        fields = []
        while True:
            fd = f.read(32)
            if fd[0:1] in (b"\x0d", b""):
                break
            fields.append((fd[0:11].split(b"\x00")[0].decode("latin1"), fd[16]))
        f.seek(hlen)
        rows = []
        for _ in range(nrec):
            rec = f.read(rlen)
            if not rec or rec[0:1] == b"*":       # deleted record
                continue
            off, row = 1, {}
            for name, size in fields:
                row[name] = rec[off:off + size].decode("latin1").strip()
                off += size
            rows.append(row)
    return rows


def read_polylinez(path: Path):
    """Yield (record_index, [(x, y, z), ...]) for each PolyLineZ feature.

    Shapefile layout per record: XY pairs first, then the Z range and Z array.
    Measures (M) are optional and ignored.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        shape_type = struct.unpack("<i", f.read(100)[32:36])[0]
        if shape_type != 13:
            raise ValueError(f"{path.name}: expected PolyLineZ (13), got {shape_type}")
        idx = 0
        while f.tell() < size:
            head = f.read(8)
            if len(head) < 8:
                break
            _, clen = struct.unpack(">ii", head)
            rec = f.read(clen * 2)
            idx += 1
            if struct.unpack("<i", rec[0:4])[0] == 0:      # null shape
                yield idx, []
                continue
            nparts, npts = struct.unpack("<ii", rec[36:44])
            xy_off = 44 + nparts * 4
            xy = struct.unpack(f"<{npts * 2}d", rec[xy_off:xy_off + npts * 16])
            z_off = xy_off + npts * 16 + 16               # skip zmin/zmax
            zs = struct.unpack(f"<{npts}d", rec[z_off:z_off + npts * 8])
            yield idx, [(xy[2 * i], xy[2 * i + 1], zs[i]) for i in range(npts)]


def wg_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("ST37_WG_*.shp"))
