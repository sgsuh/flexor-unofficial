"""XOR-gate network structure (the binary matrix M in the paper, Sec. 2).

A row r of M (shape [n_out, n_in]) selects which encrypted bits are XORed to
produce quantized bit r, i.e. y = M x over GF(2).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class XORSpec:
    """Configuration of the XOR-gate network(s) used by one FleXOR layer.

    Attributes:
        n_in: number of encrypted bits per slice (XOR inputs).
        n_out: number of quantized bits per slice (XOR outputs).
        q: number of binary-code bit planes; each plane uses its own M.
        n_tap: number of 1's per row of M. ``None`` fills M randomly with
            {0, 1} (used in the paper's MNIST experiment, Fig. 4).
        seed: seed for generating M. Layers with equal specs share the same M.
    """

    n_in: int
    n_out: int
    q: int = 1
    n_tap: Optional[int] = 2
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.n_in <= self.n_out:
            raise ValueError(f"need 0 < n_in <= n_out, got {self.n_in}, {self.n_out}")
        if self.q < 1:
            raise ValueError(f"q must be >= 1, got {self.q}")
        if self.n_tap is not None and not 0 < self.n_tap <= self.n_in:
            raise ValueError(f"need 0 < n_tap <= n_in, got {self.n_tap}")

    @property
    def bits_per_weight(self) -> float:
        return self.q * self.n_in / self.n_out

    def matrices(self) -> List[torch.Tensor]:
        """Return q distinct XOR matrices, each a bool tensor [n_out, n_in]."""
        return [
            random_xor_matrix(self.n_out, self.n_in, self.n_tap, seed=self.seed * 1000 + i)
            for i in range(self.q)
        ]


def random_xor_matrix(n_out: int, n_in: int, n_tap: Optional[int] = 2, seed: int = 0) -> torch.Tensor:
    """Generate a random XOR matrix M (bool, [n_out, n_in]).

    With ``n_tap`` set, every row has exactly ``n_tap`` distinct 1's.
    With ``n_tap=None``, entries are i.i.d. Bernoulli(0.5); all-zero rows are
    resampled so every output depends on at least one input.
    """
    gen = torch.Generator().manual_seed(seed)
    if n_tap is not None:
        cols = torch.rand(n_out, n_in, generator=gen).argsort(dim=1)[:, :n_tap]
        m = torch.zeros(n_out, n_in, dtype=torch.bool)
        m.scatter_(1, cols, True)
        return m

    m = torch.rand(n_out, n_in, generator=gen) < 0.5
    while True:
        empty = ~m.any(dim=1)
        if not empty.any():
            return m
        m[empty] = torch.rand(int(empty.sum()), n_in, generator=gen) < 0.5


def xor_taps(m: torch.Tensor, max_tap: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert M into a gather-friendly form.

    Returns:
        taps: long [n_out, max_tap]; indices of selected inputs per row, padded
            with ``n_in`` (an index that points to a constant +1 input).
        parity: float [n_out]; (-1)^(n_tap_r - 1) for each row r (Eq. 4).
    """
    n_out, n_in = m.shape
    counts = m.sum(dim=1)
    if max_tap is None:
        max_tap = int(counts.max())
    taps = torch.full((n_out, max_tap), n_in, dtype=torch.long)
    for r in range(n_out):
        idx = m[r].nonzero().flatten()
        taps[r, : len(idx)] = idx
    parity = 1.0 - 2.0 * ((counts - 1) % 2).float()
    return taps, parity


def taps_to_matrix(taps: torch.Tensor, n_in: int) -> torch.Tensor:
    """Inverse of :func:`xor_taps`: rebuild bool M [n_out, n_in] from taps."""
    m = torch.zeros(taps.shape[0], n_in + 1, dtype=torch.bool)
    m.scatter_(1, taps.cpu(), True)
    return m[:, :n_in]
