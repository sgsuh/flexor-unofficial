"""FleXOR layers: encrypted weights -> XOR decryption -> binary codes -> scaled weights."""

import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .ops import xor_decode
from .xor_net import XORSpec, xor_taps


class FleXORWeight(nn.Module):
    """Generates a weight tensor of ``shape`` from encrypted weights.

    The weight is ``W = sum_i alpha_i * b_i`` (i over q bit planes), where each
    binary code ``b_i`` is produced by decrypting the slices of ``w_e[i]``
    with the i-th XOR matrix. ``alpha`` holds one scaling factor per output
    channel (``shape[0]``) and bit plane.
    """

    def __init__(
        self,
        shape: Sequence[int],
        spec: XORSpec,
        init_std: float = 1e-3,
        alpha_init: float = 0.2,
        s_tanh: float = 10.0,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.spec = spec
        self.s_tanh = s_tanh
        self.numel = math.prod(self.shape)
        self.n_slices = math.ceil(self.numel / spec.n_out)

        self.w_e = nn.Parameter(torch.randn(spec.q, self.n_slices, spec.n_in) * init_std)
        self.alpha = nn.Parameter(torch.full((spec.q, self.shape[0]), alpha_init))

        matrices = spec.matrices()
        max_tap = max(int(m.sum(dim=1).max()) for m in matrices)
        taps, parity = zip(*(xor_taps(m, max_tap) for m in matrices))
        self.register_buffer("taps", torch.stack(taps))  # [q, n_out, max_tap]
        self.register_buffer("parity", torch.stack(parity))  # [q, n_out]

    def binary_codes(self) -> torch.Tensor:
        """Decrypted binary codes in {-1, +1}, [q, *shape]."""
        planes = [
            xor_decode(self.w_e[i], self.taps[i], self.parity[i], self.s_tanh).reshape(-1)[: self.numel]
            for i in range(self.spec.q)
        ]
        return torch.stack(planes).view(self.spec.q, *self.shape)

    def forward(self) -> torch.Tensor:
        alpha = self.alpha.view(self.spec.q, self.shape[0], *([1] * (len(self.shape) - 1)))
        return (alpha * self.binary_codes()).sum(dim=0)

    def extra_repr(self) -> str:
        s = self.spec
        return (
            f"shape={self.shape}, q={s.q}, n_in={s.n_in}, n_out={s.n_out}, n_tap={s.n_tap}, "
            f"bits/weight={s.bits_per_weight:.3f}"
        )


class FleXORConv2d(nn.Module):
    """Conv2d whose weight is produced by :class:`FleXORWeight`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        spec: XORSpec,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        **weight_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.flexor = FleXORWeight(
            (out_channels, in_channels // groups, *self.kernel_size), spec, **weight_kwargs
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        return self.flexor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.flexor(), self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"
        )


class FleXORLinear(nn.Module):
    """Linear layer whose weight is produced by :class:`FleXORWeight`."""

    def __init__(self, in_features: int, out_features: int, spec: XORSpec, bias: bool = True, **weight_kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flexor = FleXORWeight((out_features, in_features), spec, **weight_kwargs)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        return self.flexor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.flexor(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def flexor_weights(model: nn.Module) -> Iterator[FleXORWeight]:
    """Yield every FleXORWeight module in ``model``."""
    for m in model.modules():
        if isinstance(m, FleXORWeight):
            yield m


def set_s_tanh(model: nn.Module, value: float) -> None:
    """Set S_tanh on every FleXOR layer of ``model``."""
    for m in flexor_weights(model):
        m.s_tanh = value
