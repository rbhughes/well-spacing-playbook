"""Phase 10 — counterfactual scoring for every well (incl. young, score-only ones). Forward pass
with actual sets, then with the NEIGHBOUR set emptied (inter-well penalty) and with own legs
thinned (intra-well / fishbone penalty). Writes data/processed/interference_scores.parquet with an
`in_training_domain` fail-loud flag. See docs/RECIPE.md Phase 10.
"""


def score():
    raise NotImplementedError("See docs/RECIPE.md Phase 10.")


if __name__ == "__main__":
    score()
