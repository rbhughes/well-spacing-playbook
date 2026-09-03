# well-spacing-playbook

> **Read it as an essay at [spacing.purr.io](https://spacing.purr.io)** —
> the three-number cautionary tale, every one of the 105,724 laterals
> on a map, and the honest boundary where public data runs out.

A complete, honestly-evaluated attempt to answer "how close is too close?"
for horizontal oil wells with machine learning — built entirely on free
public data, taken as far as that data can go, and **documented at the
point where it can go no further**.

Read it two ways:

1. **A cautionary tale.** The naive analysis every spacing study starts
   with gives the *wrong sign*, and this repo shows exactly why, with the
   measurement that proves it.
2. **A playbook.** Every stage — geometry reconstruction, leakage-proof
   dataset design, set-encoder quantile model, physics probes, natural-
   experiment identification, counterfactual scoring — is built so a team
   with proprietary data can drop their inputs into the same machinery.
   What that takes is spelled out in `docs/WITH_PROPRIETARY_DATA.md`.

## The cautionary tale, in three numbers

- **+1.9** — the between-pad relationship between crowding and production:
  *positive*. Crowded wells produce MORE, because operators drill good
  rock tightly. Any regression of performance on spacing across an asset
  inherits this confound and will tell you interference helps.
- **−5.3** [CI −11.4..−0.5] — the same relationship measured *within*
  pads (same rock, cluster-bootstrapped): the true sign of interference,
  recovered only by a natural experiment that holds geology fixed.
- **+16.7 points** — the model's median within-pad crowding penalty on
  its own counterfactuals: economically material, directionally right,
  but too uncertain for prescriptive spacing. The final report says so
  plainly rather than dressing it up.

The full chain of evidence — including the flattering single-seed results
that *shrank* under multi-seed averaging, kept in the record deliberately —
is in `reports/interference_report.md`.

## What survives, and what doesn't

**Delivered:** calibrated P10/P50/P90 first-year forecasts that beat
gradient-boosting baselines on a strict temporal holdout (pinball 71.6 vs
78.1); depletion-aware screening and ranking; a verified 3-D geometry
data product (105,724 legs across 65,082 wells); the within-pad
identification result above.

**Not delivered, provably:** an optimal-spacing prescription. Public data
censors the three inputs that decide it — pre-window depletion history,
frac intensity, and within-pad geology. `docs/WITH_PROPRIETARY_DATA.md`
maps each gap to the data category (which operators and vendors hold)
that closes it.

## How it works

- **Deterministic geometry (no ML):** reconstruct every well's legs in 3-D
  from public directional surveys; measure closest-approach, overlap, and
  orientation between legs and between wells — including a well's *own*
  branches (multilateral self-interference), a setting almost no public
  analysis touches.
- **Learned counterfactual (PyTorch):** two permutation-invariant set
  encoders — own legs, neighbours — predict production pace as calibrated
  quantiles. Emptying or editing a neighbour set and re-running gives the
  interference penalty.
- **Honest by construction:** the model must beat plain baselines or it
  doesn't ship; the temporal test set was opened exactly once; physics
  probes gate every claim (penalty must grow as a neighbour approaches or
  depletes, and vanish for isolated wells); counterfactuals outside the
  training domain are flagged, not quoted.

## Data

100% public Alberta regulator data — AER ST37 (surveyed 3-D wellbore
paths) and Petrinex (monthly volumes per well event) — Alberta-only
because those are the only free Canadian sources carrying both. Raw data
is gitignored (Crown copyright: re-fetch, don't re-host).
`scripts/fetch_data.py` reconstructs everything; the Petrinex acquisition
layer has since been extracted into the standalone
[petrinex-etl](../petrinex-etl) repo, which is where fetch improvements
land now.

## Layout

```text
scripts/                 acquisition & database ops — the provenance layer
  fetch_data.py            all public downloads (ST37, Petrinex, AGS grids)
  build_ppdm_schema.py     PPDM 3.9 subset -> DuckDB (needs PPDM39_PG.zip)
  load_ppdm.py             stages 0-8: raw files -> data/ppdm.duckdb
  verify_ppdm.py           re-derives every count from raw; exits non-zero
  sample_ags.py            AGS grids sampled at wells -> processed parquet
  review_data_product.py   phase 1-5 sanity review -> reports/
  within_pad_contrast.py   the natural experiment -> reports/

src/borehole_geometry/   the pipeline (each phase runs via python -m)
  geometry.py st37.py      pure primitives (unit-tested)
  legs -> cohort -> pairs -> context -> dataset    deterministic data product
  baselines -> train -> report -> score            modelling & evaluation
  model.py                 the architecture (unit-tested); config.py constants

data/    raw/ (all fetches, incl. raw/ags grids) · work/ (purgeable scratch)
         processed/ (pipeline parquets) · ppdm.duckdb — ALL gitignored
models/  fold x seed checkpoints + vocabs/norms — gitignored
reports/ generated .md analyses — TRACKED; these are the deliverables,
         topped by interference_report.md
docs/    DATA_PIPELINE.md (how the data was built and verified)
         WITH_PROPRIETARY_DATA.md (what closes the gaps, if you have it)
         RECIPE.md (the original plan; where reality diverged, the module
         docstrings record why)
```

## Quick start

```bash
uv sync                                   # create the env from uv.lock
uv run pytest                             # geometry + model-architecture tests
uv run python scripts/fetch_data.py       # download the public data into data/raw/
# then follow the phases: docs/RECIPE.md is the map, module docstrings the territory
```

## Status

**Complete.** Data pipeline verified (16/16 reconciliation checks),
15-model ensembles finalized, final report delivered. Maintained as a
reference implementation; not under active development.

## Licence

Code: MIT (see `LICENSE`). Data: public regulator sources, each under its
own licence (`data/README.md`).
