# RECIPE — Borehole-geometry interference model (step-by-step)

**Goal:** quantify, per well, how much production pace is lost to interference — both **inter-well**
(neighbours draining shared rock) and **intra-well** (a multilateral's own legs draining each
other) — reconstructed from **public** directional data. Two artifacts:

1. `data/processed/spacing_pairs.parquet` + `spacing_wells.parquet` — **deterministic** geometry
   features (no ML; independently useful).
2. `data/processed/interference_scores.parquet` — **model-based** per-well counterfactual:
   predicted pace with actual neighbours/legs vs. with them removed = the **interference penalty**.
   Produced by the PyTorch set-encoder (`model.py`), trained on wells old enough to have a measured
   pace, scored on everything (including wells too young to measure).

This is a public-data reworking of an earlier internal recipe. The two changes that shape
everything: the data is free Alberta regulator data (see `data/README.md`),
and **multilaterals are the point**, not a v1 simplification.

---

## Ground rules (decide first — they shape everything)

- **Multilateral-first.** A "well" may have many legs. Geometry, features, and the counterfactual
  all treat a well as *a set of legs plus a set of neighbours*. This is the distinctive angle.
- **As-of-spud convention.** Every feature describes the world at the well's own spud date: only
  neighbours with `first_prod_date < well.spud_date` count as parents; depletion is cumulative
  production *before* spud. A later well never rewrites an earlier well's row → training rows are
  stable as data grows.
- **Target = pace, well-level.** Production is measured per well (never per leg — nobody meters
  individual legs), so the target is one number per well:
  `pace_pct_of_p50_per_yr = (oil + gas/6.0) / EUR_p50 * 100 / (months_of_data/12)`,
  gated on `months_of_data >= 6`. The `/6.0` gas→BOE convention lives in `config.BOE_GAS_DIVISOR`.
- **EUR is fit, not sourced.** No free source gives a per-well EUR, so we estimate `EUR_p50` with a
  decline-curve fit (`context.py`). This removes the "vendor EUR leakage" caveat of the internal
  version — you own the whole estimate — at the cost of one extra step.
- **Straight-line legs.** Free data gives per-leg **bottom-hole points**, so legs are reconstructed
  as straight kickoff→bottom-hole segments, not curved surveys. For the short, shallow Clearwater
  legs this targets, that's a good approximation (see pitfalls). Curved paths are a paid upgrade.
- **The model must beat a non-set baseline** (Phase 7) or the counterfactual column doesn't ship.

---

## Data sources (all public; licences + fetch in `data/README.md`)

**Alberta only.** Saskatchewan and BC were evaluated on 2026-08-23 and dropped — SK publishes
2-point 2-D sticks with no survey and no production volumes; BC was only a single-lateral
comparison. All three sources below are Alberta and all are free.

- **AER ST37 shapefiles** — the geometry anchor. `ST37_WG_*` is **PolyLineZ (true 3-D)** with
  `WGGeomSrce` flagging each bore `Surveyed` (real directional survey, median 57 stations, max 990)
  or `Calculated` (2-vertex stick, no survey). Filter on that flag; it matches vertex counts
  exactly. In block `TWP_001_025`: 24,439 Surveyed / 89,012 Calculated. Attribution licence —
  publish derivatives, not raw (`data/README.md`).
- **Petrinex Well Infrastructure** — per-event well headers (`WellIdentifier`,
  `WellEventSequence`, `WellLocationException`, licence, field/pool, `LinkedFacilityID`). The
  bridge between ST37 geometry and production.
- **Petrinex Volumetrics** — monthly OIL/GAS/WATER per **well event** → the pace target and
  neighbour depletion. One zip per month; free window is a rolling ~5 years (2022-01..2026-07 as
  of 2026-08-23). Join `WellIdentifier` to `FromToIDIdentifier` where `FromToIDType='WI'`.
  **This window is the binding constraint on the target** — early production is only observable
  for wells that came on stream inside it.

---

## Phase 0 — setup (`config.py`)

This repo is the project. `uv sync`; constants live in `config.py` (provinces, radii, leg caps,
target gates, splits). Torch CPU/MPS is fine — the model is ~tens of thousands of params.

---

## Phase 1 — cohort (`cohort.py` → `data/processed/cohort.parquet`)

One row per candidate well. Join ST37 well geometry to Petrinex Well Infrastructure on
UWI/event, then to Volumetrics on `WellIdentifier` == `FromToIDIdentifier`. Filter to
`WGGeomSrce == 'Surveyed'`, `first_prod_date >= FIRST_PROD_MIN`, has-production, spud date
present. Compute the pace target where the well is mature enough
(`months_of_data >= MIN_MONTHS`), else NULL → score-only.

`FIRST_PROD_MIN` cannot precede the volumetric window (2022-01) — a well that came on earlier has
its early, highest-rate production truncated and its pace is not comparable.

**Checks (keep in the report):** cohort size; % multilateral (≥2 surveyed bores per `Well_LicNo`)
and the bore-count distribution; % with a measurable target; target median + P1/P99 (decide
winsorization now). Play/formation isn't in ST37 — derive it from the Petrinex field/pool codes or
a spatial join to a Clearwater/Mannville polygon.

---

## Phase 2 — legs (`legs.py` → `data/processed/legs.parquet`)

The multilateral core. One row per well carrying its **set of legs**.

1. **Enumerate legs.** Group ST37 bores by `Well_LicNo`; distinguish them by the UWI label's
   the UWI label's **location exception** (leading) and **event sequence** (trailing) --
   `LE/LSD-SEC-TWP-RGEWM/ES`, the Canadian DLS display format. Convert to the Petrinex
   `WellIdentifier` with `st37.uwi_from_st37_label`; the join is 100% when read this way and
   68% when the two fields are swapped, so verify the match rate. Keep only bores with
   `WGGeomSrce == 'Surveyed'`. Verified example: licence `0257776` has 9 surveyed bores across
   events 02/03.
2. **Build legs from the survey.** Each `Surveyed` bore already carries its full 3-D polyline —
   use the vertices, do not approximate with a straight stick. Project to a local metre frame with
   `geometry.latlon_to_local_m` (avoids the EPSG:3857 inflation trap by construction — see
   pitfalls); ST37 ships NAD83 10TM, so reproject rather than assuming lat/long. Record per-leg
   length, azimuth (`geometry.bearing_deg`), heel/toe (x,y), and TVD from the Z ordinate.
3. **Filter stubs** (`length < MIN_LEG_M`) and record `n_legs`, total lateral length, fan spread
   (azimuth range), and mean inter-leg spacing (min `seg_seg_min_dist` among the well's own legs).
4. **VERIFY** leg counts against a few known Clearwater fishbones (Marten Hills / Nipisi); the free
   count is a **floor** (short intra-LSD legs collapse) — log how many wells hit `MAX_OWN_LEGS`.

---

## Phase 3 — candidate pairing (`pairs.py`, part 1)

All (well, neighbour) pairs whose legs come within `NEIGHBOR_RADIUS_M`, without O(N²): emit the
`GRID_CELL_M` grid cells each well's padded bounding box touches, self-join on cell id
(`a<b`), dedupe, then a cheap bbox-distance refine. Output `data/processed/candidate_pairs.parquet`.

---

## Phase 4 — pairwise + intra-well geometry (`pairs.py`, part 2 — numpy)

For each candidate pair, the **minimum leg-to-leg** geometry (min over all leg-pairs of
`geometry.seg_seg_min_dist`): `min_dist_m`, `mean_overlap_frac`, `azimuth_delta_deg`
(`geometry.acute_angle_deg`), vertical offset from TVD. Process **both directions** (overlap is
asymmetric). Also compute each well's **intra-well** leg crowding (its own legs' pairwise spacing)
— this is the fishbone self-interference signal.

Output `data/processed/spacing_pairs.parquet` (directed pairs) + intra-well columns on the well.
**Spot-check:** siblings on one pad should show `min_dist_m` ~200-400 m and `azimuth_delta ~0`; if
everything reads ~1.44× too large, the projection is wrong (see pitfalls).

---

## Phase 5 — context (`context.py`)

1. **Timing:** `days_gap = child.spud - neighbour.first_prod`; classify parent / co-dev.
2. **Neighbour depletion at child spud:** sum monthly production before the child's spud
   (build the needed-pairs list first, join to production — don't scan per pair).
3. **Same-formation / same-play** flags; keep different-formation neighbours within a small vertical
   offset (stacked pay).
4. **Per-well EUR via decline-curve analysis:** fit Arps to each well's monthly production
   (`DCA_MIN_MONTHS` minimum) → `EUR_p50`, completing the pace target. State the fit method + its
   uncertainty in the report.

Output `spacing_pairs_ctx.parquet` + `spacing_wells.parquet`.
**Checkpoint:** Phases 1-5 are the deterministic data product — sanity-review before any ML.

---

## Phase 6 — dataset (`dataset.py`)

One example per well: own static features (total lateral length, n_legs, fan spread, intra-well
crowding, vintage, mean TVD) + embeddings (formation/play/province); the **own-leg set** (≤
`MAX_OWN_LEGS` leg vectors); the **neighbour-well set** (≤ `MAX_NEIGHBOR_WELLS` vectors:
min_dist, overlap, azimuth_delta, days_gap, log1p depletion, same-formation, is_codev, vert_offset).
Target = winsorized pace; NULL-target wells → `score.parquet`.

**Splits (decide skill or fiction):** temporal holdout `first_prod >= TEMPORAL_HOLDOUT` → test;
GroupKFold by pad for train/val (pad siblings leak under random splits). Normalise on train only.

---

## Phase 7 — baselines FIRST (`baselines.py`, scikit-learn)

B0 global-median (MAE floor); B1 ridge on own-well features (geology/completion share); B2
HistGradientBoosting on own-well + **flattened** neighbour/leg aggregates (the honest competitor).
**Gate:** if B2 ≤ B1, the neighbour/leg signal isn't there — investigate Phases 4-5 before any torch.
A quick `GROUP BY spacing_class` median-pace table should already show infill under-pacing.

---

## Phase 8 — the model (`model.py`, `train.py`)

`InterferenceModel`: two `SetEncoder`s (own legs, neighbour wells) with masked attention pooling +
own-static + embeddings → 3 monotone quantiles (P10/P50/P90), ~tens of thousands of params.
**Pinball loss** (robust to the fat pace tail; calibration checkable). Empty set → pooled zeros
(the counterfactual path — must be finite; unit-tested). AdamW, batch 512, early-stop on val
pinball, GroupKFold folds, weight young wells by `min(months,24)/24`. Save one self-contained
`models/interference.pt` (state, norms, vocabs, config, cutoff).

Architecture invariants are unit-tested (`tests/test_model.py`): permutation-invariance, mask
correctness, empty-set finiteness, quantile ordering.

---

## Phase 9 — evaluation gate (`report.py`, part 1)

On the untouched 2023+ test set: **ship gate = model P50 MAE < B2 MAE** (if it only ties, ship B2
— the counterfactual works the same by zeroing aggregates). Calibration (fraction below P10 /
above P90) overall and by play/vintage/spacing_class. **Physics probes** (run the model as a
function): isolated well → penalty ≈ 0; sweep a parent 1500→100 m → penalty rises; sweep depletion
→ penalty rises; **thin a fishbone's legs → intra-well penalty falls.** Probe failures mean
confounding won, whatever the MAE says.

---

## Phase 10 — counterfactual scoring (`score.py`)

For every well: (1) forward pass with actual sets; (2) neighbours emptied → inter-well penalty;
(3) own legs thinned to a reference spacing → intra-well (fishbone) penalty.
`penalty_pct = (1 − pred/pred_counterfactual) * 100` (guard the divide; clip; report negatives).
Output `data/processed/interference_scores.parquet` with `in_training_domain` (play+vintage seen in
training; FALSE ⇒ penalty untrustworthy — never present a penalty without this caveat).

---

## Phase 11 — validation report (`report.py`, part 2 → `reports/interference_report.md`)

Cohort/coverage; baseline ladder (B0→B2→model); calibration + reliability plot; physics-probe
sweeps; **headline exhibits** — penalty vs. nearest-parent distance faceted by formation; the
fishbone curve (intra-well penalty vs. leg count / leg spacing — the novel result); top-20
most-penalised recent wells. Honesty section (verbatim): observational counterfactual not causal;
straight-line-leg and leg-count-floor limits; per-leg production not measured; domain limits.

---

## Phase 12 — operationalisation

Wrap Phases 1-5 + scoring as a rerunnable pipeline (`scripts/fetch_data.py` → the phase modules).
Scoring reruns each data refresh; retraining is **evidence-driven** — evaluate the frozen model on
newly-matured wells, log MAE + calibration drift, retrain only when it degrades past the Phase-9
shipped values by a set margin. Calendar retraining is the stateful-watermark of ML — don't.

---

## Pitfall appendix

- **Projection inflation** — Web Mercator (EPSG:3857) over-reads distance ~1.44× at 55° N.
  `geometry.latlon_to_local_m` gives true metres; use it, don't reproject to 3857.
- **Leg count is a floor** — short legs bottoming in one legal subdivision can collapse to one UWI;
  the location-exception code mitigates but does not eliminate this. A dense engineered fishbone
  may under-resolve. State it; the paid AER surveys resolve it.
- **Only ~21% of ST37 bores are `Surveyed`** (block `TWP_001_025`; higher in the northern,
  horizontal-dominated blocks). Report the surveyed fraction of your cohort — and split it by well
  type, because most `Calculated` bores are verticals that never had a survey to publish, so a bare
  "78% lack surveys" overstates the loss.
- **Do not straight-line a surveyed bore.** ST37 `Surveyed` bores carry the full 3-D polyline
  (median 57 stations); collapsing them to heel-toe sticks throws away the very geometry the
  closest-approach calculation depends on. Straight lines are only a fallback, and `Calculated`
  bores should be excluded rather than approximated.
- **Per-leg production is not measured** — intra-well interference is inferred from cross-well
  variation in leg configs vs. well-level output. Observational, not causal. Say so.
- **Operators drill good rock tightly** — without geology controls the model learns "tight ⇒ good."
  The pace target + play/formation embeddings + location are the defence; the Phase-9 probes detect it.
- **Random splits leak via pads.** GroupKFold by pad or the val MAE is fiction.
- **gas/6.0 BOE convention** — keep it consistent everywhere (`config.BOE_GAS_DIVISOR`).
- **Empty-set forward pass IS the counterfactual** — if untested, the headline number is untested.
- **Alberta raw data is not redistributable** — publish derivatives + attribution; keep `data/`
  gitignored. With Alberta as the only source there is no redistributable anchor, so anything that
  must be reproducible from the repo alone has to go through `fetch_data.py`.
- **The volumetric window slides** — the free Petrinex archive is a rolling ~5 years, so
  `FIRST_PROD_MIN` and any "months of history" gate silently change meaning over time. Re-run
  `fetch_data.py probe_vol` and record the window in the report alongside the results.
