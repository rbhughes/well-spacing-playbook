"""Geometry primitives — the deterministic core of Phases 2 & 4. Pure, no I/O."""

import math

import numpy as np

from borehole_geometry.geometry import (
    acute_angle_deg,
    bearing_deg,
    haversine_m,
    latlon_to_local_m,
    seg_seg_min_dist,
)


def test_haversine_known_distance():
    # ~1 deg of latitude ~= 111.2 km
    d = haversine_m(54.0, -114.0, 55.0, -114.0)
    assert abs(d - 111_195) < 500


def test_local_frame_is_true_metres_not_mercator_inflated():
    # 0.01 deg east at 55N. Web-Mercator would inflate by ~1/cos(55) ~= 1.74x; the local frame
    # must return true ground metres (the whole point of avoiding EPSG:3857).
    x, y = latlon_to_local_m(55.0, -113.99, 55.0, -114.0)
    true_m = haversine_m(55.0, -114.0, 55.0, -113.99)
    assert abs(float(x) - true_m) < 1.0
    assert abs(float(y)) < 1e-6


def test_bearing_cardinals():
    assert abs(bearing_deg(54.0, -114.0, 55.0, -114.0) - 0.0) < 0.5      # due north
    assert abs(bearing_deg(54.0, -114.0, 54.0, -113.0) - 90.0) < 0.5     # due east


def test_acute_angle_is_direction_agnostic():
    assert acute_angle_deg(10.0, 10.0) == 0.0
    assert acute_angle_deg(10.0, 190.0) == 0.0        # reverse heading == parallel
    assert abs(acute_angle_deg(0.0, 90.0) - 90.0) < 1e-9
    assert abs(acute_angle_deg(350.0, 20.0) - 30.0) < 1e-9


def test_parallel_segments_distance_is_the_offset():
    # two 1000 m parallel legs offset 300 m apart -> closest approach 300 m
    d = seg_seg_min_dist((0, 0), (1000, 0), (0, 300), (1000, 300))
    assert abs(d - 300.0) < 1e-6


def test_crossing_segments_touch():
    d = seg_seg_min_dist((-10, 0), (10, 0), (0, -10), (0, 10))
    assert d < 1e-6


def test_endpoint_closest_approach():
    # colinear, non-overlapping -> gap between the near endpoints
    d = seg_seg_min_dist((0, 0), (100, 0), (150, 0), (250, 0))
    assert abs(d - 50.0) < 1e-6


def test_degenerate_point_segment():
    d = seg_seg_min_dist((5, 5), (5, 5), (0, 0), (10, 0))
    assert abs(d - 5.0) < 1e-6


def test_fishbone_legs_from_common_heel():
    # two legs fanning from one heel at +/-15 deg; closest approach is the shared heel (~0)
    heel = (0.0, 0.0)
    a = (1000 * math.sin(math.radians(15)), 1000 * math.cos(math.radians(15)))
    b = (1000 * math.sin(math.radians(-15)), 1000 * math.cos(math.radians(-15)))
    assert seg_seg_min_dist(heel, a, heel, b) < 1e-6
    # and they spread apart at the toes
    assert np.hypot(a[0] - b[0], a[1] - b[1]) > 400
