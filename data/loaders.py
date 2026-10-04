"""MNIST / CIFAR-10 data loaders (torchvision datasets)."""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _transforms(name: str, train: bool, augment: bool):
    normalize = [transforms.ToTensor(), transforms.Normalize(*_STATS[name])]
    if name == "cifar10" and train and augment:
        return transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + normalize
        )
    return transforms.Compose(normalize)


def build_datasets(name: str, root: str = "/data/torchvision", augment: bool = True):
    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}.get(name)
    if cls is None:
        raise ValueError(f"unknown dataset: {name}")
    train = cls(root, train=True, transform=_transforms(name, True, augment), download=False)
    test = cls(root, train=False, transform=_transforms(name, False, augment), download=False)
    return train, test


def build_loaders(
    name: str,
    root: str = "/data/torchvision",
    batch_size: int = 128,
    test_batch_size: int = 512,
    num_workers: int = 4,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train, test = build_datasets(name, root, augment)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False, **common)
    return train_loader, test_loader
