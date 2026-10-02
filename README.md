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

## Deviations from the paper

- **LeNet-5 scaling-factor init.** The paper initializes all scaling factors
  to 0.2. LeNet-5 has no BatchNorm, so with ±1 weights each layer amplifies
  activations by ~0.2·sqrt(fan_in) (≈11× for FC1) and training collapses to
  chance accuracy. MNIST configs use `alpha_init: kaiming` (sqrt(2/fan_in));
  ResNet configs keep the paper's 0.2.
