# Data provenance

The raw and processed data are **gitignored** (large, and re-downloadable). This file is the
durable record of what they are, where they come from, and the licence for each — run
`uv run python scripts/fetch_data.py` to reconstruct `data/raw/`.

Everything here is **public regulator data** under a licence that permits this use. Nothing
proprietary or third-party-licensed is included.

## Sources

| dir | source | contents | licence |
|---|---|---|---|
| `raw/saskatchewan/` | Saskatchewan GeoHub — Petroleum MapServer, "Non Vertical Wells" (layer 1) | per-**leg** wellbores explicitly typed `Boss`/`Leg`/`Whipstock`, each with its own bottom-hole lat/long (verified: a 34-leg fishbone) | Government of Saskatchewan Standard Unrestricted Use Data Licence v2.0 (commercial + redistribution OK) |
| `raw/alberta/ST37_*` | Alberta Energy Regulator ST37 "List of Wells" (shapefiles) | per-leg bottom-hole points (each leg = its own UWI) + a well-geometry line layer (~78% straight sticks, ~21% surveyed traces) | Open Government Licence – Alberta (attribution) — see note below |
| `raw/alberta/AB_Well_Infrastructure_*` | Petrinex Alberta public data — Well Infrastructure | per-leg enumeration by UWI/event (legal location only, no coordinates) + production linkage | Petrinex public data terms |
| `raw/bc/` | BC Energy Regulator (BCER) IRIS — directional surveys | full per-station curved trajectories, **single-lateral** Montney/Duvernay — the comparison set | BCER Open Data Licence (commercial + redistribution OK) |
| `raw/reference/` | AER ST37 layout, WCSB UWI scheme, licence PDFs | documentation only | as published |

## The one licence caveat (Alberta)

AER/Petrinex data are Crown copyright. The ST37 spatial data is distributed under OGL-Alberta
(attribution), but the AER product catalogue also carries generic non-commercial-reproduction
language. **Do not re-host the raw Alberta files publicly** — that's why `data/` is gitignored.
Publish code and transformed/aggregated derivatives with attribution; let others pull the raw
data via `fetch_data.py`. Saskatchewan and BC are cleaner (their open-data licences permit
redistribution), so the SK data can anchor anything that must be reproducible from the repo alone.

## What's free vs. not

Free everywhere here: per-leg **bottom-hole locations** → straight-line leg geometry (leg count,
length, azimuth, fan pattern), plus monthly production. **Not** free at scale: full per-station
*curved* leg trajectories in Alberta (AER Directional Surveys, $10/well) — the straight-line
approximation is adequate for the short, shallow Clearwater legs this project targets. Per-leg
production is not measured by anyone (production is per well); the leg count from free data is a
**floor** (short legs bottoming in the same legal subdivision can collapse to one UWI).

## Attribution strings to use when publishing derivatives

- Saskatchewan: "Contains information licensed under the Government of Saskatchewan Standard
  Unrestricted Use Data Licence."
- Alberta: "Contains information licensed under the Open Government Licence – Alberta" (verify the
  current attribution wording on download).
- British Columbia: "Contains information licensed under the BC Energy Regulator Open Data Licence."
