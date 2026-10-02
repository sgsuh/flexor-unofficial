import os

import pytest
import torch

from data import build_loaders

ROOT = os.environ.get("FLEXOR_DATA_ROOT", "/data/torchvision")
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason=f"{ROOT} not found")


@pytest.mark.parametrize(
    "name,n_train,n_test,shape",
    [("mnist", 60000, 10000, (1, 28, 28)), ("cifar10", 50000, 10000, (3, 32, 32))],
)
def test_loaders(name, n_train, n_test, shape):
    train, test = build_loaders(name, ROOT, batch_size=64, test_batch_size=100, num_workers=0)
    assert len(train.dataset) == n_train and len(test.dataset) == n_test
    x, y = next(iter(train))
    assert x.shape == (64, *shape) and y.shape == (64,)
    assert y.dtype == torch.long and 0 <= int(y.min()) and int(y.max()) <= 9
    xt, _ = next(iter(test))
    # Normalized inputs should be roughly zero-mean / unit-variance.
    assert abs(xt.mean().item()) < 0.5 and 0.5 < xt.std().item() < 1.5
