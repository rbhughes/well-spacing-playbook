"""Model architecture invariants (Phase 8). These catch the failure modes that would silently
corrupt the counterfactual: order-dependence, mask leakage, an empty-set NaN, or un-ordered
quantiles. No data or training needed — the module is self-contained."""

import torch

from borehole_geometry.model import InterferenceModel, SetEncoder, pinball_loss


def _model():
    torch.manual_seed(0)
    return InterferenceModel(
        own_static_dim=5, leg_feat_dim=6, neighbor_feat_dim=10,
        n_formation=8, n_play=8, n_province=3,
    ).eval()


def _batch(n_legs, n_nbrs, max_legs=24, max_nbrs=12):
    b = {
        "own_static": torch.randn(1, 5),
        "formation_idx": torch.tensor([1]),
        "play_idx": torch.tensor([2]),
        "province_idx": torch.tensor([0]),
        "legs": torch.randn(1, max_legs, 6),
        "legs_mask": torch.zeros(1, max_legs, dtype=torch.bool),
        "neighbors": torch.randn(1, max_nbrs, 10),
        "neighbors_mask": torch.zeros(1, max_nbrs, dtype=torch.bool),
    }
    b["legs_mask"][0, :n_legs] = True
    b["neighbors_mask"][0, :n_nbrs] = True
    return b


def test_neighbor_permutation_invariance():
    m = _model()
    b = _batch(n_legs=3, n_nbrs=5)
    out1 = m(b)
    perm = torch.randperm(5)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["neighbors"][0, :5] = b["neighbors"][0, :5][perm]
    out2 = m(b2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_masked_padding_does_not_change_output():
    m = _model()
    b = _batch(n_legs=2, n_nbrs=2)
    out1 = m(b)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["neighbors"][0, 5:] = torch.randn(7, 10)   # scribble in masked-out slots
    b2["legs"][0, 10:] = torch.randn(14, 6)
    out2 = m(b2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_empty_sets_are_finite():
    # the counterfactual input: no neighbours and (thinned to) no legs must not NaN
    m = _model()
    out = m(_batch(n_legs=0, n_nbrs=0))
    assert torch.isfinite(out).all()


def test_quantiles_are_ordered():
    m = _model()
    out = m(_batch(n_legs=4, n_nbrs=6))
    assert (out[:, 0] <= out[:, 1]).all() and (out[:, 1] <= out[:, 2]).all()


def test_set_encoder_empty_pools_to_zero():
    enc = SetEncoder(in_dim=6).eval()
    x = torch.randn(1, 5, 6)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    assert torch.equal(enc(x, mask), torch.zeros(1, enc.out_dim))


def test_pinball_positive_and_zero_at_perfect():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    assert pinball_loss(pred, torch.tensor([5.0])) > 0
    assert abs(pinball_loss(torch.tensor([[2.0, 2.0, 2.0]]), torch.tensor([2.0])).item()) < 1e-6
