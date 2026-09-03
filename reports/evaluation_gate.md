# Evaluation gate (Phase 9) — the one-shot test-set opening

test wells: 2,624  (first prod >= 2025-01)

SHIP GATE:  model P50 MAE 71.6  vs  B2 MAE 78.1  ->  PASS
pinball on test: 98.7

CALIBRATION (test): fraction below P10 / below P50 / below P90
  overall                    n= 2624    12 /   53 /   88
  UNKNOWN                    n= 1628    11 /   54 /   88
  0176000                    n=   87     5 /   59 /   92
  0280214                    n=   60    32 /   68 /   82
  sparse (<=3 nbrs)          n=  163    20 /   50 /   89
  crowded (>12 nbrs)         n= 2021    11 /   54 /   88

PHYSICS PROBES (400 test wells; ensemble P50, % of peer P50)
  APPROACH  1500->100 m: median P50 drop +0.4 pts; strictly monotone-decreasing wells 48%
            median P50 by dist: 1500m=116  1000m=117  600m=116  300m=116  100m=114
  DEPLETION 0 -> 5M boe at 300 m: median P50 drop -15.7 pts
  ISOLATED  4 wells: max |penalty| when emptying an already-empty set 0.00e+00 (must be 0)
  STEAM     injector vs idle well at 300 m: median P50 delta -12.3 pts (support should not read as theft)
  LEG-THIN  124 fishbones, median P50 keeping 1..4 legs: 114  114  114  115
            marginal gain per extra leg: +0.2  +0.2  +0.4   (diminishing = crowding learned)

GATE VERDICT: approach FAIL | depletion PASS | isolated PASS | ship-gate PASS
=> CONFOUNDING WON on at least one probe: do NOT ship the counterfactual yet
