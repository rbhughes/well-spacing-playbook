# Running the playbook with proprietary data

This project's final report names three reasons the model stops short of
prescriptive spacing: a censored depletion window, missing completion
intensity, and geology only at regional resolution. None of these is a
modelling failure — each is a *data* gap, and each maps to a data
category that operators already license (geoSCOUT, Enverus, IHS/S&P) or
hold internally. This page maps gap -> data -> what changes, in impact
order, for a team that wants to rerun this machinery with real inputs.

## 1. Full production history (pre-window cumulative volumes)

**The gap.** The free Petrinex archive is a rolling ~5-year window. A
parent well drilled before the window shows zero observable depletion;
neighbour cumulative-production features are censored for every older
well; the cohort keeps only recent wells (here: 10,200 targets out of
70,852 surveyed producers).

**With full history** (any commercial well database has decades):
depletion kernels un-censor outright, the cohort multiplies, and every
infill event since ~2010 becomes a before/after natural experiment —
the same within-pad, time-varying design that produced this project's
one identified causal result. Highest impact of any single input.
Plug-in point: `context.py` (neighbour as-of features), `cohort.py`
(first-production gate).

## 2. Completion / frac intensity

**The gap.** Proppant tonnage, stage count, fluid volumes, and stage
spacing are absent from public volumetric data. Frac intensity is the
biggest omitted variable twice over: it drives productivity directly,
and it is confounded with spacing — operators pump bigger jobs into
better rock, the same behaviour behind the inverted between-pad slope
(+1.9) this project measured.

**With completions data**: the within-pad interference estimate tightens
substantially, and stage spacing gives the distance channel a physical
frac-hit radius instead of a bare 1/d kernel. Plug-in point: target and
neighbour static features in `dataset.py` (`NBR_KEYS`).

## 3. Per-well formation tops (geology that varies within a pad)

**The gap.** Public AGS grids are regional rasters — constant across a
pad, hence useless to a within-pad identification strategy (this project
measured them as noise in the dev model and dropped them).

**With per-well tops** (picked depths, target-zone thickness, structure):
geology becomes a feature that varies *within* pads — the direct fix for
the good-rock confound rather than a workaround. Plug-in point: replaces
the `geo_*` columns from `sample_ags.py`.

## 4. Pressure surveys (measured depletion)

Initial reservoir pressures and shut-in bottom-hole pressures with dates
convert depletion from an inferred proxy (cumulative volume through a
kernel) into a measured quantity — validation for the depletion channel
at minimum, an auxiliary supervision target at best.

## 5. Initial-potential tests (pre-interference rock quality)

An IP test measures deliverability before any neighbour exists.
Performance-relative-to-own-IP cancels the good-rock term by
construction — a cleaner instrument for the confound than anything
buildable from public data, and a drop-in alternative to the peer-median
benchmark in `cohort.py`.

## 6. Decline-fit EURs and cost data

EUR distributions give a longer-horizon benchmark than 12-month pace;
drilling and completion costs turn "optimal spacing" into the economic
question a drilling manager actually asks. Both are reporting-layer
additions, not pipeline changes.

## What stays the same

Everything this repo exists to demonstrate: the leakage-proof splits,
the physics probes that stopped a model the loss function approved of,
the within-pad natural experiment, the counterfactual domain checks, and
the discipline of letting flattering numbers die in public. Proprietary
data replaces the *inputs*; the *harness* is the point.
