# well-spacing-playbook — Agent guide (READ FIRST)

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

**Alberta only.** Saskatchewan and BC were evaluated and dropped (2026-08-23): SK publishes
2-point 2-D sticks with no directional survey and no production volumes; BC was only ever a
single-lateral comparison set. Do not reintroduce them.

Free Alberta regulator data — **AER ST37** (surveyed 3-D wellbore geometry + bottom/surface holes),
**Petrinex Well Infrastructure** (per-event well headers, the join bridge), **Petrinex Volumetrics**
(monthly production per well event). All in `data/raw/`, **gitignored** and reproducible via
`scripts/fetch_data.py`; every source + licence is in `data/README.md`.
**Licence caveat:** Alberta raw data is Crown-copyright, not redistributable — publish derivatives +
attribution, never re-host raw. Keep `data/` gitignored.
Key data facts, all verified against the files:
- ST37 well geometry is **PolyLineZ (true 3-D)**. `WGGeomSrce` flags each bore `Surveyed` or
  `Calculated`; only `Surveyed` has a real path (median 57 stations, max 990). `Calculated` is a
  2-vertex stick — for a vertical well that is complete, for a horizontal it means no survey.
  In township block 01-25: 24,439 Surveyed vs 89,012 Calculated.
- Legs are identified by the UWI label, which is the Canadian DLS display format
  `LE/LSD-SEC-TWP-RGEWM/ES`: **leading** pair is the **location exception** `LE`, **trailing**
  digit is the **event sequence** `ES`. Get these the wrong way round and the ST37 <-> Petrinex
  join silently drops from 100% to 68% — it is not an error, just fewer matches, so check the
  match rate rather than trusting the parse. Bores of one multilateral share `Well_LicNo`.
- The UWI used throughout is the **Petrinex `WellIdentifier`**, built from that label by
  `st37.uwi_from_st37_label`: `'1' + LE + LSD + SEC + TWP + RGE + meridian + ES(2)`. The location
  exception is ALPHANUMERIC (`F1`, `AA`, `W0`…) — a digits-only parse silently dropped 11.7% of
  the province. Verified province-wide: 532,553 / 532,623 (99.99%).
- Production is per **well event**, not per leg. Volumetrics report at the battery with
  `FromToIDType='WI'` naming the producing well; join `WellIdentifier` (Well Infrastructure) to
  `FromToIDIdentifier` (Volumetrics).
- The free volumetric window is a **rolling ~5 years** (2022-01..2026-07 as of 2026-08-23), so a
  well's early production is only observable if it came on after the window opens. Run
  `fetch_data.py probe_vol` to re-check; the window slides.

## 4. Repo layout

```text
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

## 7. STATUS / RESUME HERE (2026-09-03)

- **PUBLISHED.** The finished work is committed (single honest commit
  `6e83808`, Bryan's explicit choice) and the repo is PUBLIC. Branch
  flow: work lands on `dev`, merge/push to `main` (default).
- **Essay site** in `site/` (Astro + MapLibre), deployed to Cloudflare
  Pages project `well-spacing-playbook`; canonical URL
  https://spacing.purr.io (Route 53 CNAME done; custom-domain click in
  Cloudflare pending/done — verify TLS if touching this). Map data:
  `site/public/data/legs.json` built from
  `data/processed/legs.parquet` (heel->toe straightened).
- **Guardrail:** `docs/IF_WE_HAD_GEOLOGIC_DATA.md` is private and
  gitignored — never commit or quote it; `WITH_PROPRIETARY_DATA.md`
  is the public version.
- **Learning contract (§0)** governed the build phase; the project is
  complete. For site/publishing maintenance, normal implementer mode
  is fine — but any NEW modeling work with Bryan returns to §0.
- Tests: `uv run --with pytest python -m pytest` (bare `pytest`
  fails to spawn in this venv). 15/15 pass.
