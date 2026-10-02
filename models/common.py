"""Layer factories that return FP or FleXOR layers depending on ``spec``."""

from typing import Optional

import torch.nn as nn

from flexor import FleXORConv2d, FleXORLinear, XORSpec


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    spec: Optional[XORSpec],
    stride: int = 1,
    padding: int = 0,
    bias: bool = False,
    flexor_kwargs: Optional[dict] = None,
) -> nn.Module:
    if spec is None:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    return FleXORConv2d(
        in_channels, out_channels, kernel_size, spec, stride=stride, padding=padding, bias=bias,
        **(flexor_kwargs or {}),
    )


def make_linear(
    in_features: int,
    out_features: int,
    spec: Optional[XORSpec],
    bias: bool = True,
    flexor_kwargs: Optional[dict] = None,
) -> nn.Module:
    if spec is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return FleXORLinear(in_features, out_features, spec, bias=bias, **(flexor_kwargs or {}))
