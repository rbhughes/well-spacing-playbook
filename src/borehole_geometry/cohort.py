"""Phase 1 — cohort. Build data/processed/cohort.parquet: one row per candidate well from the
free public sources (SK GeoHub 'Non Vertical Wells' + AB ST37), filtered to modern horizontals/
multilaterals with production history. Carries the well-level target `pace_pct_of_p50_per_yr`
(NULL for wells too young to measure — those become score-only). See docs/RECIPE.md Phase 1.
"""


def build_cohort():
    raise NotImplementedError("See docs/RECIPE.md Phase 1.")


if __name__ == "__main__":
    build_cohort()
