# borehole-geometry-ml — Agent guide (READ FIRST)

## 0. How we work here — the collaboration contract (MOST IMPORTANT)

**This is a LEARNING project. The deliverable is Bryan's understanding of PyTorch and good
project practice — NOT finished code shipped fast.** This overrides the default "autonomous
implementer" mode.

- **Explain before doing.** Before writing or changing code, explain the concept, the tensor
  shapes, the design trade-off, and the PyTorch idiom in play. The *why* is the point.
- **One step at a time.** Do a single concept/phase, then stop and check in. Do NOT batch-implement
  multiple phases or "just finish it." Bryan drives the pace.
- **Let Bryan write the code that carries the learning** — model architecture, the training loop,
  tensor manipulation, autograd. I scaffold, explain, and review; he types the meaningful parts.
  When I do write code, walk through it line by line rather than handing over a finished block.
- **Best practices are an explicit goal.** Flag and explain structure/typing/testing/reproducibility/
  config decisions as we hit them, rather than baking them in silently.
- **Bryan's PyTorch level:** did one PyTorch project ~a year ago; wants a **refresher**, not
  teaching from zero. Re-surface idioms (`nn.Module`, autograd, `DataLoader`, device handling,
  the training loop) as they come up; don't assume fluency, don't over-explain the trivial.

**Note on the current scaffold:** I (the agent) over-built the initial scaffold in a sprint before
this contract was set — `geometry.py` and `model.py` are fully implemented and unit-tested. Treat
those as **reference / teaching material to walk through or rebuild together**, NOT as finished
work to move past. In particular, `model.py` (the dual set-encoder + masked attention pooling) is a
prime candidate to strip back to a blank and build together from tensors up — offer that.

## 1. What this project is

A PyTorch model estimating **production interference** between oil wells — both **inter-well**
(neighbours draining shared rock) and **intra-well** (a multilateral's own legs draining each
other) — from the **3-D geometry** of how they were drilled. For each well it answers a
counterfactual: predicted production pace with its actual neighbours/legs vs. with them removed =
the **interference penalty**. Built on **public** regulator data. Full methodology: **`docs/RECIPE.md`**.

The distinctive angle is **multilaterals** (fishbone wells with many legs, common in Canadian
heavy-oil plays) — a setting almost no public analysis touches. The repo name is deliberately
legible to non-oil-and-gas readers (a hiring audience).

## 2. The ML approach (in one breath)

Two permutation-invariant **set encoders** — one over a well's own legs, one over its neighbour
wells — with masked attention pooling, fused with the well's static features + embeddings, predict
pace as calibrated **P10/P50/P90** (pinball loss). The counterfactual = a forward pass with a set
emptied; **the empty-set path IS the headline output**, so it must be finite and is unit-tested.
Non-negotiables: must beat a gradient-boosting baseline, graded on a 2023+ hold-out, checked
against physics sanity probes. See `docs/RECIPE.md` Phases 7-10.

## 3. Data (public; gitignored)

Free regulator data — **SK GeoHub** (legs explicitly typed Boss/Leg/Whipstock — the anchor),
**AB ST37** (per-leg bottom holes; the Clearwater fishbones), **Petrinex** (production), **BCER**
(single-lateral comparison). All in `data/raw/` (~635 MB), **gitignored** and reproducible via
`scripts/fetch_data.py`; every source + licence is in `data/README.md`.
**Licence caveat:** Alberta raw data is Crown-copyright, not redistributable — publish derivatives +
attribution, never re-host raw. SK is unrestricted (redistributable). Keep `data/` gitignored.
Key data facts: free = per-leg **bottom-hole points** → straight-line legs; **paid** = curved
per-station surveys; per-leg production is never measured; free leg count is a **floor**.

## 4. Repo layout

```
src/borehole_geometry/
  config.py     constants (paths, radii, leg caps, splits)         [done]
  geometry.py   pure primitives: seg-seg distance, local-metre projection, bearings  [done, tested]
  model.py      dual set-encoder + pooling + quantile head + pinball loss  [done, tested — TEACHING CANDIDATE]
  cohort legs pairs context dataset baselines train score report   [STUBS -> docs/RECIPE.md phases]
scripts/fetch_data.py     reproducible public-data download
tests/  test_geometry.py (8) + test_model.py (7)   — 15 pass, offline
docs/RECIPE.md            the full step-by-step methodology
data/  README.md (tracked) + raw/ processed/ (gitignored)
```

## 5. Environment & commands

- **uv** project (Python 3.12). `uv sync` builds the env from `uv.lock`.
- `uv run pytest` — the 15 offline tests (geometry + model invariants).
- `uv run ruff check .` — lint (F/B/I).
- `uv run python scripts/fetch_data.py [source]` — download public data.
- Torch is CPU/MPS (small model; trains on the Mac). Editor: LazyVim + snacks explorer (press `I`
  to reveal gitignored dirs like `data/raw`).

## 6. Conventions already established (keep consistent)

src/ layout + hatchling; uv + committed `uv.lock`; ruff (F/B/I); pytest for anything pure/offline;
data gitignored with a fetch script + `data/README.md` provenance; one module per pipeline phase
matching `docs/RECIPE.md`; constants in `config.py`, never hard-coded.

## 7. STATUS / RESUME HERE (2026-08-10)

- **Where we are:** scaffold created + pushed. GitHub `rbhughes/borehole-geometry-ml`, **PRIVATE**
  (flip to public when ready: `gh repo edit --visibility public`). Branches `main` + `dev`; work on
  `dev`, currently checked out, tree clean. No CI workflow yet (offered, not added).
- **Implemented + tested:** `config.py`, `geometry.py`, `model.py` (see the over-build note in §0 —
  `model.py` is the natural first teaching exercise).
- **Not started:** the pipeline phases `cohort → legs → pairs → context → dataset → baselines →
  train → score → report`, all stubbed to `docs/RECIPE.md`. Data not yet fetched fresh by the user
  (the agents' downloads are staged in `data/raw/`).
- **Good next-session options** (let Bryan choose): (a) tour the scaffold to decide keep-vs-rebuild;
  (b) strip `model.py` and build the set-encoder together from tensors; (c) start Phase 1
  (`cohort.py`) and learn DuckDB→pandas→tensors in order. Add the offline ruff+pytest CI when he wants it.
- **Open reminders:** decide public/private timing; decide how much of the over-built code to
  rebuild for learning vs. keep as reference.
