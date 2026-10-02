"""LeNet-5 used in the paper's MNIST experiment (Sec. 3):
32C5-MP2-64C5-MP2-512FC-10SoftMax, every layer followed by an XOR-gate network.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexor import XORSpec

from .common import make_conv, make_linear


class LeNet5(nn.Module):
    def __init__(self, spec: Optional[XORSpec] = None, num_classes: int = 10, flexor_kwargs: Optional[dict] = None):
        super().__init__()
        kw = dict(flexor_kwargs=flexor_kwargs)
        # 'same' padding: 28 -> 14 -> 7 after the two max-pools.
        self.conv1 = make_conv(1, 32, 5, spec, padding=2, bias=True, **kw)
        self.conv2 = make_conv(32, 64, 5, spec, padding=2, bias=True, **kw)
        self.fc1 = make_linear(64 * 7 * 7, 512, spec, **kw)
        self.fc2 = make_linear(512, num_classes, spec, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)
