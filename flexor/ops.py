"""Differentiable XOR-gate decryption.

Forward follows Eq. (2)/(4): bits live in {-1, +1} (Boolean 0 -> -1) and an
n-input XOR is (-1)^(n-1) * prod(sign(x_j)).
Backward follows the simplified gradient of Eq. (6):
    d/dx_i ~= S * (-1)^(n-1) * (1 - tanh^2(S x_i)) * prod_{j != i} sign(x_j)
which falls out of autograd when each input goes through :class:`SignTanh`
and the outputs are multiplied (Algorithm 1).

Two alternative XOR training schemes from the paper's ablation (Fig. 5) are
selectable with ``mode``:
    "ste":    sign forward, identity backward.
    "analog": tanh(S x) forward and backward (Eq. 3/4), so XOR outputs are
              real numbers in (-1, +1); use binary decoding for inference.
"""

import torch
import torch.nn.functional as F


class SignTanh(torch.autograd.Function):
    """sign(x) forward, S * (1 - tanh^2(S x)) surrogate gradient backward.

    sign(0) is mapped to +1 so outputs are always in {-1, +1}.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, s_tanh: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.s_tanh = s_tanh
        one = torch.ones_like(x)
        return torch.where(x >= 0, one, -one)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        t = torch.tanh(x * ctx.s_tanh)
        return grad_out * ctx.s_tanh * (1.0 - t * t), None


def sign_tanh(x: torch.Tensor, s_tanh: float) -> torch.Tensor:
    return SignTanh.apply(x, s_tanh)


class SignSTE(torch.autograd.Function):
    """sign(x) forward (sign(0) = +1), identity backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(x)
        return torch.where(x >= 0, one, -one)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out


XOR_MODES = ("flexor", "ste", "analog")


def xor_decode(
    w_e: torch.Tensor, taps: torch.Tensor, parity: torch.Tensor, s_tanh: float, mode: str = "flexor"
) -> torch.Tensor:
    """Decrypt encrypted weights through an XOR-gate network.

    Args:
        w_e: real-valued encrypted weights, [n_slices, n_in].
        taps: long [n_out, max_tap] from :func:`flexor.xor_net.xor_taps`.
        parity: float [n_out], (-1)^(n_tap_r - 1) per output.
        s_tanh: steepness of the tanh surrogate gradient.
        mode: XOR training scheme, one of :data:`XOR_MODES`.

    Returns:
        Quantized bits in {-1, +1} ((-1, +1) for "analog"), [n_slices, n_out].
    """
    if mode == "flexor":
        s = sign_tanh(w_e, s_tanh)
    elif mode == "ste":
        s = SignSTE.apply(w_e)
    elif mode == "analog":
        s = torch.tanh(w_e * s_tanh)
    else:
        raise ValueError(f"unknown XOR mode: {mode}")
    s = F.pad(s, (0, 1), value=1.0)  # index n_in is the constant +1 padding input
    return s[:, taps].prod(dim=-1) * parity
