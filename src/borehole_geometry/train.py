"""Phase 8 — training loop for model.InterferenceModel. AdamW, batch 512, pinball loss over
P10/P50/P90, early-stop on val pinball, GroupKFold by pad, sample-weight young wells by
min(months,24)/24. Saves models/interference.pt = {state_dict, norms, vocabs, config, cutoff}.
See docs/RECIPE.md Phase 8.
"""


def train():
    raise NotImplementedError("See docs/RECIPE.md Phase 8.")


if __name__ == "__main__":
    train()
