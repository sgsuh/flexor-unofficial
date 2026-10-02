import pytest
import torch
import torch.nn as nn

from flexor import FleXORConv2d, FleXORLinear, XORSpec, flexor_weights, storage_report
from models import LeNet5, build_model, resnet20, resnet32

SPEC = XORSpec(n_in=8, n_out=20)


def test_lenet5_all_layers_quantized():
    model = LeNet5(spec=SPEC)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)
    assert len(list(flexor_weights(model))) == 4
    assert model.fc1.flexor.shape == (512, 64 * 7 * 7)


def test_lenet5_full_precision():
    model = LeNet5()
    assert len(list(flexor_weights(model))) == 0
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize("fn,depth,fp_params", [(resnet20, 20, 0.27e6), (resnet32, 32, 0.46e6)])
def test_resnet_structure(fn, depth, fp_params):
    fp = fn()
    assert sum(p.numel() for p in fp.parameters()) == pytest.approx(fp_params, rel=0.03)

    model = fn(spec=SPEC)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    # All convs except the first one are FleXOR; first conv and FC stay FP.
    assert len(list(flexor_weights(model))) == depth - 2
    assert isinstance(model.conv1, nn.Conv2d) and isinstance(model.fc, nn.Linear)
    assert not any(isinstance(m, FleXORLinear) for m in model.modules())


def test_resnet_per_stage_specs():
    specs = [XORSpec(n_in=19, n_out=20), XORSpec(n_in=16, n_out=20), XORSpec(n_in=7, n_out=20)]
    model = resnet20(spec=specs)
    for stage, spec in zip((model.layer1, model.layer2, model.layer3), specs):
        convs = [m for m in stage.modules() if isinstance(m, FleXORConv2d)]
        assert len(convs) == 6 and all(c.flexor.spec == spec for c in convs)
    # Paper Table 2: (19, 16, 7)/20 per group averages to ~0.47 bits/weight.
    assert storage_report(model)["quantized_bits_per_weight"] == pytest.approx(0.47, abs=0.01)


def test_flexor_kwargs_are_forwarded():
    model = resnet20(spec=SPEC, flexor_kwargs=dict(alpha_init=0.5, s_tanh=3.0))
    fw = next(flexor_weights(model))
    assert fw.s_tanh == 3.0 and torch.all(fw.alpha == 0.5)


def test_build_model():
    assert isinstance(build_model("lenet5"), LeNet5)
    with pytest.raises(ValueError):
        build_model("vgg")
