# borehole-geometry-ml

A PyTorch model that estimates how much an oil well's production is reduced by **interference** —
both from neighbouring wells and from the well's own branches — using the **3-D geometry** of how
the wells were drilled. It learns from thousands of real wells, then answers a counterfactual for
each one: *what would this well have produced with more space around it?*

The gap between the two is the **interference penalty**, reported per well.

## Why it's interesting

Modern wells are drilled horizontally and packed close together; some are **multilaterals** — a
single well with many branches ("legs") fanning underground, common in Canadian heavy-oil plays.
Pack the wells (or the legs) too tightly and they drain each other's oil; space them too far apart
and you leave resource behind. Today that trade-off is mostly judged by intuition. This project
turns it into a measured, per-well estimate learned from geometry and production history.

The distinctive part is the **multilateral** angle: the model reasons about a well's *own* legs
competing with each other (fishbone self-interference) as well as its neighbours — a setting
almost no public analysis touches.

## How it works

- **Deterministic geometry (no ML):** reconstruct every well's legs in 3-D from public directional
  data, then measure closest-approach, overlap, and orientation between legs and between wells.
  This stage is a useful dataset on its own.
- **Learned counterfactual (PyTorch):** two permutation-invariant *set encoders* — one over a
  well's own legs, one over its neighbours — predict production **pace** as calibrated P10/P50/P90.
  Emptying a set and re-running the model gives the interference penalty.
- **Honest by construction:** the neural model must beat plain baselines (ridge / gradient
  boosting) or it doesn't ship; it's graded on a future (2023+) hold-out; and it's checked against
  physical sanity probes (penalty must rise as a neighbour gets closer or more depleted, and be
  ~zero for an isolated well).

## Data

100% **public** regulator data — Saskatchewan GeoHub, Alberta ST37, Petrinex, BC Energy Regulator
— under licences that permit this use. Nothing proprietary. The data is gitignored (large but
freely re-downloadable); `scripts/fetch_data.py` reconstructs it and `data/README.md` documents
every source and licence.

## Layout

```
src/borehole_geometry/   cohort - legs - pairs - context - dataset - baselines - model/train - score - report
                         geometry.py + model.py are implemented & unit-tested; the rest are the pipeline
scripts/fetch_data.py    reproducible public-data download
docs/RECIPE.md           the full step-by-step methodology
tests/                   geometry primitives + model invariants (permutation, masking, empty-set)
```

## Quick start

```bash
uv sync                                   # create the env from uv.lock
uv run pytest                             # geometry + model-architecture tests
uv run python scripts/fetch_data.py       # download the public data into data/raw/
# then work through the phases in docs/RECIPE.md
```

## Status

Scaffold + methodology + verified geometry/model core. The pipeline phases (`cohort`..`report`)
are specified in `docs/RECIPE.md` and stubbed for implementation.

## Licence

Code: MIT (see `LICENSE`). Data: public regulator sources, each under its own licence
(`data/README.md`).
