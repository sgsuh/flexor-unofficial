"""Download MNIST and CIFAR-10 into the torchvision layout.

Usage (inside the container):
    python scripts/download_data.py --root /data/torchvision
"""

import argparse

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/torchvision")
    args = parser.parse_args()

    for name, cls in [("MNIST", datasets.MNIST), ("CIFAR10", datasets.CIFAR10)]:
        for train in (True, False):
            ds = cls(root=args.root, train=train, download=True)
            split = "train" if train else "test"
            print(f"{name:8s} {split:5s}: {len(ds)} samples")


if __name__ == "__main__":
    main()
