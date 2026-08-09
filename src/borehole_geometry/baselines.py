"""Phase 7 — baselines FIRST (scikit-learn, no torch). B0 global-median, B1 ridge on own-well
features, B2 HistGradientBoosting on own-well + flattened neighbour/leg aggregates. The neural
set-encoder must beat B2 on the held-out test set or it doesn't ship (Phase 9 gate). Writes
reports/baselines.md. See docs/RECIPE.md Phase 7.
"""


def run_baselines():
    raise NotImplementedError("See docs/RECIPE.md Phase 7.")


if __name__ == "__main__":
    run_baselines()
