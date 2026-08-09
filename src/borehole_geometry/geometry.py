"""Pure geometry primitives for reconstructing legs and measuring proximity.

These are the deterministic core of Phases 2 and 4 and carry no I/O, so they are unit-tested
(tests/test_geometry.py). Everything works in a LOCAL equirectangular metre frame built around a
reference latitude — which sidesteps the EPSG:3857 distance-inflation trap (~1.4x at 55 deg N) by
construction: x already carries the cos(latitude) factor, so distances come out in true metres.
"""

from __future__ import annotations

import math

import numpy as np

_EARTH_R_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres. Used for a projection-free check of point distances."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(a))


def latlon_to_local_m(lat, lon, ref_lat: float, ref_lon: float):
    """Project lat/lon (degrees) to a local equirectangular (x_east, y_north) frame in METRES
    around (ref_lat, ref_lon). Accurate to <0.1% over a pad/neighbourhood, and — unlike Web
    Mercator — already in true metres, so no cos(lat) correction is needed downstream."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = math.radians(1.0) * _EARTH_R_M * math.cos(math.radians(ref_lat)) * (lon - ref_lon)
    y = math.radians(1.0) * _EARTH_R_M * (lat - ref_lat)
    return x, y


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass azimuth (0-360, 0 = north) of the vector point1 -> point2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def acute_angle_deg(az_a: float, az_b: float) -> float:
    """Acute angle (0-90) between two azimuths: 0 = parallel, 90 = perpendicular.
    Direction-agnostic, so a leg and its 180-deg reverse read as parallel."""
    d = abs(az_a - az_b) % 180.0
    d = min(d, 180.0 - d)
    return d


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def seg_seg_min_dist(p1, p2, q1, q2) -> float:
    """Minimum distance between two line SEGMENTS p1p2 and q1q2 in a planar metre frame
    (each point an (x, y) pair). This is the leg-to-leg proximity primitive: two laterals are
    segments, and interference scales with their closest approach, not their endpoints' distance."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    if a <= 1e-12 and e <= 1e-12:                 # both degenerate -> point/point
        return float(np.hypot(*(p1 - q1)))
    if a <= 1e-12:                                # first degenerate -> point/segment
        s, t = 0.0, _clamp(f / e)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:                            # second degenerate -> segment/point
            t, s = 0.0, _clamp(-c / a)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a)
    closest_p = p1 + d1 * s
    closest_q = q1 + d2 * t
    return float(np.hypot(*(closest_p - closest_q)))
