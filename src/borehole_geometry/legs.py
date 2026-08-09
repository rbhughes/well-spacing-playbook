"""Phase 2 — legs. Reconstruct each well's laterals as STRAIGHT legs (kickoff/surface -> per-leg
bottom-hole) from the free per-leg bottom-hole points: SK explicitly types Boss/Leg/Whipstock;
AB resolves legs as distinct UWIs. Output data/processed/legs.parquet: one row per well with its
set of legs (length, azimuth, heel/toe in local metres, mean TVD). Free data gives straight legs,
not curved surveys, and the leg count is a FLOOR (see docs/RECIPE.md Phase 2 + pitfalls).
"""


def build_legs():
    raise NotImplementedError("See docs/RECIPE.md Phase 2.")


if __name__ == "__main__":
    build_legs()
