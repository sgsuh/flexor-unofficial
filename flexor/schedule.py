"""S_tanh schedule (Sec. 4, "Learning rate and S_tanh warmup").

- Warmup: S_tanh rises linearly from ``warmup_start`` to ``base`` over
  ``warmup_epochs`` (same schedule as the learning-rate warmup).
- Decay compensation: each time the learning rate decays (``milestones``),
  S_tanh is multiplied by ``factor`` to cancel the shrinking effect of weight
  decay on encrypted weights. Use ``factor=1.0`` to disable.
"""

from bisect import bisect_right
from typing import Sequence

import torch.nn as nn

from .layers import set_s_tanh


class STanhSchedule:
    def __init__(
        self,
        base: float = 10.0,
        warmup_start: float = 5.0,
        warmup_epochs: float = 0.0,
        milestones: Sequence[float] = (),
        factor: float = 2.0,
    ):
        self.base = base
        self.warmup_start = warmup_start
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.factor = factor

    def value(self, epoch: float) -> float:
        """S_tanh at a (possibly fractional) epoch."""
        if epoch < self.warmup_epochs:
            return self.warmup_start + (self.base - self.warmup_start) * epoch / self.warmup_epochs
        return self.base * self.factor ** bisect_right(self.milestones, epoch)

    def apply(self, model: nn.Module, epoch: float) -> float:
        value = self.value(epoch)
        set_s_tanh(model, value)
        return value
