"""CIFAR-10 ResNet-20/32 (He et al., 2016) with option-A (parameter-free) shortcuts.

The first conv and the last FC layer stay full precision (paper, Sec. 4). All
3x3 convs inside residual blocks become FleXOR layers when a spec is given.
``spec`` may be a single XORSpec shared by all stages, or a sequence of three
(one per stage) for layer-group mixed precision (paper, Table 2).
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexor import XORSpec

from .common import make_conv

SpecArg = Union[None, XORSpec, Sequence[Optional[XORSpec]]]


class ShortcutA(nn.Module):
    """Stride-2 subsampling plus zero-padding of new channels."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.pad = out_channels - in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :: self.stride, :: self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, spec: Optional[XORSpec], flexor_kwargs):
        super().__init__()
        kw = dict(flexor_kwargs=flexor_kwargs)
        self.conv1 = make_conv(in_channels, out_channels, 3, spec, stride=stride, padding=1, **kw)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = make_conv(out_channels, out_channels, 3, spec, padding=1, **kw)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutA(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCifar(nn.Module):
    def __init__(
        self,
        depth: int,
        spec: SpecArg = None,
        num_classes: int = 10,
        flexor_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"depth must be 6n+2, got {depth}")
        n = (depth - 2) // 6
        specs = list(spec) if isinstance(spec, (list, tuple)) else [spec] * 3
        if len(specs) != 3:
            raise ValueError(f"expected 3 stage specs, got {len(specs)}")

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        stages, in_ch = [], 16
        for i, (out_ch, stage_spec) in enumerate(zip((16, 32, 64), specs)):
            blocks = []
            for j in range(n):
                stride = 2 if i > 0 and j == 0 else 1
                blocks.append(BasicBlock(in_ch, out_ch, stride, stage_spec, flexor_kwargs))
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def resnet20(**kwargs) -> ResNetCifar:
    return ResNetCifar(20, **kwargs)


def resnet32(**kwargs) -> ResNetCifar:
    return ResNetCifar(32, **kwargs)
