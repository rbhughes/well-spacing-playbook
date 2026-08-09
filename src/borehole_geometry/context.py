"""Phase 5 — context. Attach non-geometric context to each pair: timing (days_gap, parent vs
co-dev via spud/first-prod dates), neighbour depletion at the child's spud from Petrinex monthly
production, and same-formation/same-play flags. Also fits per-well EUR via decline-curve analysis
(no free source provides EUR) to complete the pace target. Output spacing_pairs_ctx.parquet +
spacing_wells.parquet. See docs/RECIPE.md Phase 5.
"""


def build_context():
    raise NotImplementedError("See docs/RECIPE.md Phase 5.")


if __name__ == "__main__":
    build_context()
