import pytest
import torch

from flexor.xor_net import XORSpec, random_xor_matrix, taps_to_matrix, xor_taps


def test_fixed_tap_rows_have_exactly_n_tap_ones():
    m = random_xor_matrix(20, 8, n_tap=2, seed=0)
    assert m.shape == (20, 8)
    assert (m.sum(dim=1) == 2).all()


def test_random_fill_has_no_empty_rows():
    for seed in range(20):
        m = random_xor_matrix(10, 3, n_tap=None, seed=seed)
        assert m.any(dim=1).all()


def test_matrix_is_deterministic_per_seed():
    assert torch.equal(random_xor_matrix(20, 8, seed=3), random_xor_matrix(20, 8, seed=3))
    assert not torch.equal(random_xor_matrix(20, 8, seed=3), random_xor_matrix(20, 8, seed=4))


def test_spec_bit_planes_use_distinct_matrices():
    m0, m1 = XORSpec(n_in=8, n_out=20, q=2).matrices()
    assert not torch.equal(m0, m1)


def test_spec_bits_per_weight():
    assert XORSpec(n_in=8, n_out=20, q=2).bits_per_weight == pytest.approx(0.8)


@pytest.mark.parametrize("kwargs", [dict(n_in=0, n_out=10), dict(n_in=11, n_out=10), dict(n_in=4, n_out=10, n_tap=5)])
def test_spec_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        XORSpec(**kwargs)


def test_taps_roundtrip_and_parity():
    # Example network of Fig. 2 / Appendix A, Eq. (7).
    m = torch.tensor(
        [[1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.bool
    )
    taps, parity = xor_taps(m)
    assert taps.shape == (6, 3)
    assert torch.equal(taps_to_matrix(taps, 4), m)
    # (-1)^(n_tap - 1): +1 for 3 taps, -1 for 2 taps.
    assert parity.tolist() == [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
