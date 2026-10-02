# flexor-unofficial

Unofficial PyTorch implementation of
[FleXOR: Trainable Fractional Quantization](https://arxiv.org/abs/2009.04126) (NeurIPS 2020).

## Setup

All commands run inside Docker (requires the NVIDIA container runtime).

```bash
docker compose build
scripts/run.sh python scripts/download_data.py   # MNIST + CIFAR-10 (torchvision layout)
scripts/run.sh                                    # interactive shell
```

Datasets are stored on the host at `$FLEXOR_DATA_DIR`
(default: `/home/sgsuh/data/torchvision`) and mounted at `/data/torchvision`.
