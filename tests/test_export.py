import torch
import torch.nn as nn

from flexor import FleXORConv2d, FleXORLinear, XORSpec, export_model, storage_report, verify_export
from flexor.export import pack_bits, unpack_bits


def _model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),  # full-precision first layer
        FleXORConv2d(8, 16, 3, XORSpec(n_in=8, n_out=20, q=1), padding=1, bias=False),
        FleXORConv2d(16, 16, 3, XORSpec(n_in=12, n_out=20, q=2, seed=1), padding=1, bias=False),
        nn.Flatten(),
        FleXORLinear(16 * 4 * 4, 10, XORSpec(n_in=4, n_out=10, n_tap=None)),
    )


def test_pack_unpack_roundtrip():
    bits = torch.randint(0, 2, (1003,))
    assert torch.equal(unpack_bits(pack_bits(bits), 1003), bits.to(torch.uint8))


def test_export_reconstruction_matches_training_forward():
    torch.manual_seed(0)
    model = _model()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    exported = export_model(model)
    assert verify_export(model, exported) < 1e-6
    assert "0.weight" in exported["fp_state"]
    assert not any("flexor" in k for k in exported["fp_state"])


def test_storage_report():
    model = _model()
    report = storage_report(model)
    bpw = {l["name"]: l["bits_per_weight"] for l in report["layers"]}
    assert abs(bpw["1.flexor"] - 0.4) < 1e-2
    assert abs(bpw["2.flexor"] - 1.2) < 1e-2
    assert abs(bpw["4.flexor"] - 0.4) < 1e-2
    assert report["fp_params"] == 3 * 8 * 9 + 8 + 10  # first conv + linear bias
    assert report["quantized_compression"] > 1
