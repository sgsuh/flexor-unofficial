import itertools

import pytest
import torch

from flexor.ops import SignTanh, xor_decode
from flexor.xor_net import random_xor_matrix, xor_taps


def test_sign_tanh_forward_maps_zero_to_plus_one():
    x = torch.tensor([-2.0, -1e-6, 0.0, 1e-6, 3.0])
    assert SignTanh.apply(x, 10.0).tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]


def test_sign_tanh_backward():
    x = torch.tensor([-0.1, 0.0, 0.05], requires_grad=True)
    s = 10.0
    SignTanh.apply(x, s).sum().backward()
    expected = s * (1 - torch.tanh(x.detach() * s) ** 2)
    assert torch.allclose(x.grad, expected)


def test_two_input_xor_truth_table():
    # Table 4: Boolean 0 <-> -1, 1 <-> +1.
    m = torch.tensor([[1, 1]], dtype=torch.bool)
    taps, parity = xor_taps(m)
    for a, b in itertools.product([0, 1], repeat=2):
        w = torch.tensor([[2.0 * a - 1, 2.0 * b - 1]])
        y = xor_decode(w, taps, parity, 10.0)
        assert y.item() == 2.0 * (a ^ b) - 1


@pytest.mark.parametrize("n_tap", [1, 2, 3, None])
def test_decode_matches_gf2_matmul(n_tap):
    m = random_xor_matrix(20, 8, n_tap=n_tap, seed=1)
    taps, parity = xor_taps(m)
    w = torch.randn(50, 8)
    y = xor_decode(w, taps, parity, 10.0)
    bits = (w >= 0).long()
    ref = (bits @ m.long().T) % 2
    assert torch.equal(y, 2.0 * ref.float() - 1)


def test_gradient_matches_eq6():
    torch.manual_seed(0)
    s = 7.0
    m = random_xor_matrix(10, 6, n_tap=3, seed=2)
    taps, parity = xor_taps(m)
    w = (torch.randn(4, 6) * 0.1).requires_grad_()
    g = torch.randn(4, 10)
    (xor_decode(w, taps, parity, s) * g).sum().backward()

    wd = w.detach()
    sign = torch.where(wd >= 0, 1.0, -1.0)
    dtanh = s * (1 - torch.tanh(wd * s) ** 2)
    expected = torch.zeros_like(wd)
    for r in range(10):
        cols = m[r].nonzero().flatten().tolist()
        n = len(cols)
        for i in cols:
            others = torch.ones(4)
            for j in cols:
                if j != i:
                    others = others * sign[:, j]
            expected[:, i] += g[:, r] * (-1) ** (n - 1) * dtanh[:, i] * others
    assert torch.allclose(w.grad, expected, atol=1e-5)
