import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from flexor import FleXORConv2d, FleXORLinear, FleXORWeight, STanhSchedule, XORSpec, flexor_weights, set_s_tanh


def test_weight_shape_values_and_param_count():
    spec = XORSpec(n_in=8, n_out=20, q=2)
    fw = FleXORWeight((16, 8, 3, 3), spec, alpha_init=0.2)
    w = fw()
    assert w.shape == (16, 8, 3, 3)
    # Two bit planes with alpha=0.2 -> each weight in {-0.4, 0, 0.4}.
    levels = torch.tensor([-0.4, 0.0, 0.4])
    assert torch.isclose(w.detach().reshape(-1, 1), levels).any(dim=1).all()
    n_slices = -(-16 * 8 * 9 // 20)
    assert fw.w_e.shape == (2, n_slices, 8)
    assert fw.alpha.shape == (2, 16)


@pytest.mark.parametrize("n_in", [4, 8, 12, 16, 20])
def test_bits_per_weight(n_in):
    spec = XORSpec(n_in=n_in, n_out=20)
    fw = FleXORWeight((64, 64, 3, 3), spec)
    assert fw.w_e.numel() / fw.numel == pytest.approx(spec.bits_per_weight, rel=1e-3)


def test_conv_matches_functional_conv_with_generated_weight():
    torch.manual_seed(0)
    conv = FleXORConv2d(3, 8, 3, XORSpec(n_in=8, n_out=10), stride=2, padding=1, bias=True)
    x = torch.randn(2, 3, 9, 9)
    ref = F.conv2d(x, conv.weight, conv.bias, stride=2, padding=1)
    assert torch.allclose(conv(x), ref)


def test_gradients_reach_encrypted_weights_and_alpha():
    torch.manual_seed(0)
    model = nn.Sequential(
        FleXORConv2d(3, 8, 3, XORSpec(n_in=8, n_out=10), padding=1),
        nn.Flatten(),
        FleXORLinear(8 * 4 * 4, 5, XORSpec(n_in=4, n_out=10, q=2)),
    )
    model(torch.randn(2, 3, 4, 4)).square().sum().backward()
    for fw in flexor_weights(model):
        assert fw.w_e.grad is not None and fw.w_e.grad.abs().sum() > 0
        assert fw.alpha.grad is not None and fw.alpha.grad.abs().sum() > 0


def test_layers_with_equal_spec_share_xor_matrix():
    spec = XORSpec(n_in=8, n_out=20, seed=5)
    a = FleXORWeight((8, 8, 3, 3), spec)
    b = FleXORWeight((16, 8, 3, 3), spec)
    assert torch.equal(a.taps, b.taps)


def test_set_s_tanh():
    model = nn.Sequential(FleXORLinear(4, 4, XORSpec(n_in=2, n_out=4)), FleXORLinear(4, 4, XORSpec(n_in=2, n_out=4)))
    set_s_tanh(model, 42.0)
    assert all(fw.s_tanh == 42.0 for fw in flexor_weights(model))


def test_s_tanh_schedule():
    sched = STanhSchedule(base=10.0, warmup_start=5.0, warmup_epochs=10, milestones=[150, 175], factor=2.0)
    assert sched.value(0) == pytest.approx(5.0)
    assert sched.value(5) == pytest.approx(7.5)
    assert sched.value(10) == pytest.approx(10.0)
    assert sched.value(149.9) == pytest.approx(10.0)
    assert sched.value(150) == pytest.approx(20.0)
    assert sched.value(180) == pytest.approx(40.0)
    no_warmup = STanhSchedule(base=100.0, factor=1.0, milestones=[10])
    assert no_warmup.value(0) == no_warmup.value(50) == 100.0


def test_kaiming_alpha_init():
    fw = FleXORWeight((16, 8, 3, 3), XORSpec(n_in=8, n_out=20), alpha_init="kaiming")
    assert torch.allclose(fw.alpha, torch.full_like(fw.alpha, (2.0 / 72) ** 0.5))
    with pytest.raises(ValueError):
        FleXORWeight((4, 4), XORSpec(n_in=2, n_out=4), alpha_init="xavier")


def test_xor_modes_share_forward_codes():
    torch.manual_seed(0)
    ref = FleXORWeight((8, 4, 3, 3), XORSpec(n_in=8, n_out=10))
    for mode in ("ste", "analog"):
        fw = FleXORWeight((8, 4, 3, 3), XORSpec(n_in=8, n_out=10), xor_mode=mode)
        fw.w_e.data.copy_(ref.w_e.data)
        assert torch.equal(fw.binary_codes(), ref.binary_codes())
    with pytest.raises(ValueError):
        FleXORWeight((8, 4), XORSpec(n_in=8, n_out=10), xor_mode="bogus")


def test_clip_encrypted_uses_current_s_tanh():
    fw = FleXORWeight((8, 4), XORSpec(n_in=8, n_out=10), s_tanh=10.0)
    fw.w_e.data.normal_()
    fw.clip_encrypted(2.0)
    assert fw.w_e.abs().max() <= 0.2 + 1e-7
    fw.s_tanh = 20.0
    fw.clip_encrypted(2.0)
    assert fw.w_e.abs().max() <= 0.1 + 1e-7
