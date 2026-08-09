"""Phase 3+4 — candidate pairing and pairwise geometry. Grid-cell hash-join to find wells whose
legs come within NEIGHBOR_RADIUS_M (avoids O(N^2)), then leg-to-leg closest approach, overlap,
and azimuth difference between neighbours — plus INTRA-well leg-to-leg spacing for fishbones.
Uses geometry.seg_seg_min_dist. Output data/processed/spacing_pairs.parquet. See RECIPE.md 3-4.
"""


def build_pairs():
    raise NotImplementedError("See docs/RECIPE.md Phases 3-4.")


if __name__ == "__main__":
    build_pairs()
