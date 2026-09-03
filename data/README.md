# Data provenance

**Alberta only.** Saskatchewan and BC were evaluated on 2026-08-23 and removed: SK's public
wellbore geometry is 2-point 2-D sticks with no directional survey and no production volumes, and
BC was only ever a single-lateral comparison set. Everything the project needs is in Alberta.

The raw and processed data are **gitignored** (large, and re-downloadable). This file is the
durable record of what they are, where they come from, and the licence for each — run
`uv run python scripts/fetch_data.py` to reconstruct `data/raw/`.

Everything here is **public regulator data** under a licence that permits this use. Nothing
proprietary or third-party-licensed is included.

## Sources

| dir | source | contents | licence |
| --- | --- | --- | --- |
| `raw/alberta/ST37_Shapefiles.zip` | Alberta Energy Regulator ST37 "List of Wells" (shapefiles) | **Well geometry as PolyLineZ (true 3-D)** — `WGGeomSrce` flags each bore `Surveyed` (real directional survey) or `Calculated` (2-vertex stick). Plus bottom-hole and surface-hole points and a production-string status layer. | Open Government Licence – Alberta (attribution) — see caveat below |
| `raw/alberta/AB_Well_Infrastructure_CSV.zip` | Petrinex Alberta public data — Well Infrastructure | Per-event well headers: `WellIdentifier`, `WellEventSequence`, `WellLocationException`, licence, field/pool, status, `LinkedFacilityID`. This is the bridge between ST37 geometry and volumetrics. | Petrinex public data terms |
| `raw/alberta/volumetrics/` | Petrinex Alberta public data — Volumetric Data | Monthly OIL/GAS/WATER/COND volume **per well event**. One zip per production month (~8 MB). Reported at the battery, with `FromToIDType='WI'` naming the producing well. | Petrinex public data terms |
| `raw/reference/` | AER ST37 layout, WCSB UWI scheme | documentation only | as published |

## Verified facts about these files

Measured directly, 2026-08-23. Re-verify rather than trust if the sources are refreshed.

**ST37 well geometry is genuinely 3-D and genuinely surveyed.** Shapefile type 13 (PolyLineZ)
with real Z values. In township block `TWP_001_025` (113,451 features):

| `WGGeomSrce` | features | share | vertices |
| --- | --- | --- | --- |
| `Surveyed` | 24,439 | 21.5% | median 57, mean 76, max 990 |
| `Calculated` | 89,012 | 78.5% | exactly 2 |

The split matches vertex counts exactly, so `WGGeomSrce` is authoritative — filter on it rather
than inferring from geometry. `TWP_001_025` is the oldest, most vertical-dominated block; the
northern blocks are 4x larger and should carry a higher surveyed share.

A `Calculated` 2-vertex line is not corrupt: for a **vertical** well it is complete. For a
horizontal it means the survey was not published. Either way it is unusable for
closest-approach work.

**Legs are identifiable.** The UWI label is `EE/LSD-SEC-TWP-RGEWM/X` — leading **event sequence**
and trailing **location exception**. Bores of one multilateral share `Well_LicNo`. Example: licence
`0257776` carries 9 surveyed bores across events `02` and `03` with exception codes 0/4/8/9 and
2/3/5/6. In `TWP_001_025`, 670 licences have >=3 surveyed bores.

**Production is per well event, not per leg.** Nobody meters individual legs. Volumetrics carry
163,250 distinct producing wells in a single month.

**The volumetric window is a rolling ~5 years.** Verified 2026-08-23: 2021-12 and earlier return
404; 2022-01 .. 2026-07 return data. This is the binding constraint on the target variable — a
well's early production is only observable if it came on stream after the window opens, so the
trainable cohort is effectively wells with first production from 2022-01. Run
`uv run python scripts/fetch_data.py probe_vol` to re-check; the window slides forward monthly.

## Petrinex terms of use (verified 2026-08-24)

From <https://petrinex.ca/terms>, quoted:

> This material, including copyright and marks under the Trade Marks Act (Canada), is owned by the
> **Government of Alberta** and protected by law.
>
> This material may be used, reproduced, stored or transmitted for **non-commercial purposes**.
> However, **Crown copyright is to be acknowledged**. If it is to be used, reproduced, stored or
> transmitted for **commercial purposes, arrange first for consent** by contacting the
> Communications Coordinator.

Two things follow.

**Commercial use needs prior consent.** The project's stated beneficiaries are multilateral
operators; if results are ever used commercially, that consent has to be obtained first. This is a
real constraint on the project's end state, not boilerplate.

**No attribution string is prescribed**, only that Crown copyright be acknowledged. So there is no
mandated value for a provenance field, and the PPDM `SOURCE` values are ours: `PETRINEX` for Well
Infrastructure and Volumetrics, `AER` for ST37. See `config.SOURCE_PETRINEX` / `SOURCE_AER`.

**Note on the linked disclaimer.** The Alberta public-data page links a PDF titled
`sk_public_data_disclaimer.pdf`, and the file really is the *Saskatchewan* Ministry of Energy and
Resources disclaimer -- wrong jurisdiction, apparently a Petrinex site error. It is a liability
disclaimer only. Do not treat it as governing the Alberta data; `petrinex.ca/terms` above is the
operative text.

## The one licence caveat (Alberta)

AER/Petrinex data are Crown copyright. The ST37 spatial data is distributed under OGL-Alberta
(attribution), but the AER product catalogue also carries generic non-commercial-reproduction
language. **Do not re-host the raw Alberta files publicly** — that is why `data/` is gitignored.
Publish code and transformed/aggregated derivatives with attribution; let others pull the raw data
via `fetch_data.py`.

Because Alberta is now the only source, nothing in this repo is redistributable raw. Anything that
must be reproducible from the repo alone has to be reproducible via `fetch_data.py`, not shipped.

## What is free vs. not

Free: **surveyed 3-D wellbore paths** for the 21%+ of bores flagged `Surveyed`, per-event well
headers, and monthly production per well event for the rolling ~5-year window.

Not free: per-station surveys for bores ST37 flags `Calculated` (AER Directional Surveys,
priced per well), and volumetric history older than the public window.

> Superseded note: earlier revisions of this file stated that curved per-station trajectories were
> paid-only in Alberta and that the free data gave only straight-line legs. That is wrong — ST37
> publishes surveyed PolyLineZ traces at median 57 stations. The straight-line approximation is
> not needed for `Surveyed` bores.

## Data we do NOT have that would likely matter (noted 2026-08-27)

The Phase-9 probes showed interference is identifiable through neighbour
DEPLETION but not through bare PROXIMITY -- proximity is chosen because of
rock quality, and nothing in the free data measures rock quality directly.
Three datasets that would plausibly change that:

| dataset | what it would add | status |
|---|---|---|
| **AGS geology grids** (Alberta Geological Survey open data: formation tops, net pay / property maps) | a TRUE rock-quality control, replacing the inferred neighbourhood report card; the single most likely unlock for the distance channel | free; not yet fetched or evaluated |
| **AER pool pressure surveys** | direct measurements of the depletion field, strengthening the channel that already works and validating `nbr_cum_boe_before` as a depletion proxy | published by AER; format/coverage unevaluated |
| **Completion / frac intensity** (stages, tonnage, fluid) | controls the completion-quality confound (better-completed wells also cluster); would sharpen every comparison | not freely available at scale for AB |

## Attribution string to use when publishing derivatives

- Alberta: "Contains information licensed under the Open Government Licence – Alberta" (verify the
  current attribution wording on download).
