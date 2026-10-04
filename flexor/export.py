"""Export FleXOR models to their compressed inference format.

For inference only sign(w_e) is stored, packed as bits, together with the
scaling factors and the (shared) XOR matrices. Quantized weights are rebuilt
with integer XOR (y = M x mod 2), independently of the training code path, so
``verify_export`` checks that training-time forward matches real decryption.
"""

import math
from dataclasses import asdict
from typing import Dict, List

import torch
import torch.nn as nn

from .layers import FleXORWeight
from .xor_net import taps_to_matrix

_BIT_WEIGHTS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a flat {0, 1} tensor into uint8 (MSB first), zero-padded to 8."""
    bits = bits.flatten().to(torch.uint8).cpu()
    pad = (-bits.numel()) % 8
    bits = torch.cat([bits, bits.new_zeros(pad)]).view(-1, 8)
    return (bits * _BIT_WEIGHTS).sum(dim=1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of :func:`pack_bits`; returns the first ``n`` bits as uint8."""
    bits = (packed.view(-1, 1) & _BIT_WEIGHTS).ne(0).to(torch.uint8)
    return bits.flatten()[:n]


def xor_decrypt_bits(bits: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Boolean decryption y = M x over GF(2).

    Args:
        bits: {0, 1} [n_slices, n_in].
        m: bool [n_out, n_in].

    Returns:
        {0, 1} uint8 [n_slices, n_out].
    """
    return ((bits.long() @ m.long().T) % 2).to(torch.uint8)


def export_layer(fw: FleXORWeight) -> dict:
    q, n_in = fw.spec.q, fw.spec.n_in
    signs = (fw.w_e.detach() >= 0).cpu()  # Boolean 1 <-> +1, 0 <-> -1
    return {
        "shape": fw.shape,
        "spec": asdict(fw.spec),
        "n_slices": fw.n_slices,
        "matrices": torch.stack([taps_to_matrix(fw.taps[i], n_in) for i in range(q)]),
        "packed": [pack_bits(signs[i]) for i in range(q)],
        "alpha": fw.alpha.detach().cpu().float(),
    }


def reconstruct_weight(entry: dict) -> torch.Tensor:
    """Rebuild the dense weight of one exported layer using integer XOR."""
    shape, n_slices = entry["shape"], entry["n_slices"]
    q, n_in = entry["spec"]["q"], entry["spec"]["n_in"]
    numel = math.prod(shape)
    weight = torch.zeros(shape)
    for i in range(q):
        bits = unpack_bits(entry["packed"][i], n_slices * n_in).view(n_slices, n_in)
        y = xor_decrypt_bits(bits, entry["matrices"][i]).flatten()[:numel]
        b = (2.0 * y.float() - 1.0).view(shape)
        weight += entry["alpha"][i].view(shape[0], *([1] * (len(shape) - 1))) * b
    return weight


def _flexor_named(model: nn.Module) -> Dict[str, FleXORWeight]:
    return {name: m for name, m in model.named_modules() if isinstance(m, FleXORWeight)}


def export_model(model: nn.Module) -> dict:
    """Export FleXOR layers in compressed form plus all remaining FP state."""
    flexor = _flexor_named(model)
    prefixes = tuple(f"{name}." for name in flexor)
    fp_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith(prefixes)}
    return {
        "layers": {name: export_layer(fw) for name, fw in flexor.items()},
        "fp_state": fp_state,
    }


@torch.no_grad()
def verify_export(model: nn.Module, exported: dict) -> float:
    """Max abs difference between training-path weights and XOR-decrypted ones."""
    max_diff = 0.0
    for name, fw in _flexor_named(model).items():
        ref = fw().cpu().float()
        rec = reconstruct_weight(exported["layers"][name])
        max_diff = max(max_diff, (ref - rec).abs().max().item())
    return max_diff


def storage_report(model: nn.Module) -> dict:
    """Count storage bits of a FleXOR model.

    Quantized layers store q * n_slices * n_in encrypted bits plus 32-bit
    scaling factors. All other parameters (first/last layers, BN, biases) are
    counted as 32-bit. Shared XOR matrices and BN running stats are ignored.
    """
    flexor = _flexor_named(model)
    prefixes = tuple(f"{name}." for name in flexor)
    layers: List[dict] = []
    for name, fw in flexor.items():
        enc_bits = fw.spec.q * fw.n_slices * fw.spec.n_in
        alpha_bits = 32 * fw.alpha.numel()
        layers.append(
            {
                "name": name,
                "n_weights": fw.numel,
                "encrypted_bits": enc_bits,
                "alpha_bits": alpha_bits,
                "bits_per_weight": enc_bits / fw.numel,
            }
        )

    q_weights = sum(l["n_weights"] for l in layers)
    q_bits = sum(l["encrypted_bits"] + l["alpha_bits"] for l in layers)
    fp_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(prefixes))
    total_bits = q_bits + 32 * fp_params
    return {
        "layers": layers,
        "quantized_weights": q_weights,
        "quantized_bits_per_weight": sum(l["encrypted_bits"] for l in layers) / max(q_weights, 1),
        "quantized_compression": 32 * q_weights / max(q_bits, 1),
        "fp_params": fp_params,
        "model_compression": 32 * (q_weights + fp_params) / max(total_bits, 1),
    }
