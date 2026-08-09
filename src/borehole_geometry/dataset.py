"""Phase 6 — dataset assembly. One example per well: own static features + categoricals, the set
of its OWN legs, and the set of its NEIGHBOUR wells. Splits: temporal holdout (first_prod >=
TEMPORAL_HOLDOUT) as test; GroupKFold by pad for train/val (pad siblings leak under random
splits). Normalises on the train split only. Writes data/processed/{train,val,test,score}.parquet
and models/norms.json + vocabs. See docs/RECIPE.md Phase 6.
"""


def build_dataset():
    raise NotImplementedError("See docs/RECIPE.md Phase 6.")


if __name__ == "__main__":
    build_dataset()
