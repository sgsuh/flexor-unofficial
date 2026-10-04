# flexor-unofficial

Unofficial PyTorch implementation of
[FleXOR: Trainable Fractional Quantization](https://arxiv.org/abs/2009.04126) (NeurIPS 2020),
with reproduction experiments on MNIST and CIFAR-10.

FleXOR stores `N_in` encrypted bits per `N_out` weights and decrypts them with a
fixed XOR-gate network `M⊕`, giving fractional bits per weight (`q·N_in/N_out`).
Encrypted weights are trained with a sign forward pass and a `tanh`-based
surrogate gradient (`S_tanh`).

## Setup

All commands run inside Docker (requires the NVIDIA container runtime).

```bash
docker compose build
scripts/run.sh python scripts/download_data.py   # MNIST + CIFAR-10 (torchvision layout)
scripts/run.sh python -m pytest -q               # unit tests
scripts/run.sh                                    # interactive shell
```

Datasets are stored on the host at `$FLEXOR_DATA_DIR`
(default: `/home/sgsuh/data/torchvision`) and mounted at `/data/torchvision`.

## Training

```bash
scripts/run.sh python train.py --config configs/mnist_lenet5.yaml
scripts/run.sh python train.py --config configs/cifar10_resnet32.yaml --set flexor.n_in=12 --name r32_nin12
scripts/run.sh python train.py --config configs/cifar10_resnet32.yaml --set flexor=null --name r32_fp   # FP baseline
scripts/run.sh python train.py --config configs/cifar10_resnet32.yaml --name r32_nin12 --resume
```

`--set key=value` (repeatable, value parsed as YAML) overrides any entry of the
config; see `DEFAULTS` in `train.py`. Useful keys:

| Key | Meaning |
|---|---|
| `flexor.n_in`, `flexor.n_out`, `flexor.q` | XOR network size and number of bit planes |
| `flexor.n_tap` | 1's per row of `M⊕` (`null`: random fill) |
| `flexor.stages` | per-stage spec overrides for ResNet (mixed precision) |
| `flexor.xor_mode` | `flexor` (default), `ste` or `analog` (Fig. 5 ablation) |
| `flexor.clip` | clamp encrypted weights to `±clip/S_tanh` after each step (Fig. 15b), `null` = off |
| `s_tanh.base`, `s_tanh.factor` | `S_tanh` and its multiplier at every LR decay |

Each run writes `runs/<name>/` with `config.yaml`, `metrics.csv`, TensorBoard
logs (`tb/`), `last.pt`, the bit-packed `exported.pt` and `summary.json`
(accuracy, export check, bits/weight and compression ratio).

### Configs

| Config | Recipe |
|---|---|
| `mnist_lenet5.yaml` | LeNet-5, Adam 1e-4, batch 50, 20 epochs, `S_tanh`=100 |
| `cifar10_resnet{20,32}.yaml` | Table 1: 500 epochs, 100-epoch LR/`S_tanh` warmup, LR ×0.5 at 350/400/450 |
| `cifar10_resnet32_200ep.yaml` | Fig. 5/6: 200 epochs, LR 0.1 ×0.5 at 150/175, `S_tanh` ×2 at each decay |
| `cifar10_resnet20_mixed.yaml` | Table 2: per-stage `N_in` = 19/16/7 |

### Sweeps

`scripts/sweep.py` runs the jobs listed in a `sweeps/*.yaml` file one after
another; finished runs are skipped and interrupted ones resumed.
`scripts/summarize.py` collects `runs/*/summary.json` into a Markdown table.

```bash
scripts/run.sh python scripts/sweep.py sweeps/cifar10_200ep.yaml
scripts/run.sh python scripts/summarize.py --match c10_
```

## Code layout

```
flexor/xor_net.py   XORSpec and random XOR matrix M⊕ generation
flexor/ops.py       sign/tanh surrogate (Eq. 2-6), XOR decoding, STE / analog modes
flexor/layers.py    FleXORWeight, FleXORConv2d, FleXORLinear
flexor/schedule.py  S_tanh warmup and decay schedule
flexor/export.py    bit packing, Boolean XOR reconstruction, storage report
data/, models/      MNIST / CIFAR-10 loaders, LeNet-5, CIFAR ResNet-20/32
train.py            training entry point
scripts/            Docker wrapper, data download, sweep runner, summarizer
sweeps/             experiment lists used for the results below
```

## Results

All numbers are best test accuracy (%) of a single run (seed 0), trained on one
RTX 4070 Laptop GPU. Every FleXOR run was re-evaluated from the exported binary
encrypted weights through Boolean XOR (`export_max_diff` = 0 for all runs).

**CIFAR-10 runs use the paper's 200-epoch recipe (Fig. 5/6), not the 500-epoch
recipe behind Tables 1, 2 and 6**, to keep each run at about 40–70 min (ResNet-32
q=1 ≈ 68 min, q=2 ≈ 105 min). Paper numbers are given for reference only.

### MNIST, LeNet-5 (Fig. 4 / Fig. 12) — `sweeps/mnist_lenet5.yaml`

20 epochs. `rand`: randomly filled `M⊕` (as in Fig. 4); `tap2`: `N_tap` = 2.
Full precision: 99.30.

| bits/weight | N_out=10, rand | N_out=10, tap2 | N_out=20, rand | N_out=20, tap2 |
|---|---|---|---|---|
| 0.4 | 98.90 | 96.44 | 98.66 | 97.22 |
| 0.6 | 98.85 | 98.24 | 98.13 | 98.75 |
| 0.8 | 98.77 | 98.01 | 98.26 | 98.87 |

`tap2` at 0.4 bit was still improving at epoch 20 (train acc. 95%).

### CIFAR-10, q=1, N_out=20 (Table 1) — `sweeps/cifar10_200ep.yaml`

| Model | FP | 1.0 bit | 0.8 bit | 0.6 bit | 0.4 bit |
|---|---|---|---|---|---|
| ResNet-20 (ours, 200 ep) | 90.80 | 88.48 | 87.58 | 87.11 | 85.79 |
| ResNet-20 (paper, 500 ep) | 91.87 | 90.44 | 89.91 | 89.16 | 88.23 |
| ResNet-32 (ours, 200 ep) | 91.65 | 89.07 | 89.31 | 88.18 | 87.27 |
| ResNet-32 (paper, 500 ep) | 92.33 | 91.36 | 91.20 | 90.43 | 89.61 |

Quantized models end ~2–2.4 points below the paper. They are still underfitting
at epoch 200 (final train acc. 91.6% for ResNet-32 at 0.4 bit vs. 99.9% for FP),
so the shorter schedule is the likely main cause; this has not been verified
with 500-epoch runs.

### CIFAR-10, mixed precision (Table 2) — `sweeps/cifar10_mixed_200ep.yaml`

| ResNet-20, N_out=20 | bits/weight | Ours (200 ep) | Paper (500 ep) |
|---|---|---|---|
| uniform N_in=8 | 0.40 | 85.79 | 88.23 |
| per-stage N_in=19/16/7 | 0.47 | 86.84 | 89.29 |

The gain from layer-group mixed precision (+1.05 points over uniform 0.4 bit)
matches the paper (+1.06).

### CIFAR-10, q=2 (Table 6) — `sweeps/cifar10_q2_200ep.yaml`

| ResNet-32, N_out=20, N_in=20 | bits/weight | Ours (200 ep) | Paper (500 ep) |
|---|---|---|---|
| q=2 | 2.0 | 90.95 | 92.25 |

### XOR training scheme ablation (Fig. 5) — `sweeps/cifar10_xor_modes_200ep.yaml`

ResNet-32, N_out=10, N_in=8, q=1 (0.8 bit), `S_tanh`=10, 200 epochs (the
paper's own recipe for this figure).

| Scheme | Forward | Backward | Best | Final |
|---|---|---|---|---|
| FleXOR | sign | ∂tanh | **88.72** | 88.32 |
| Analog XOR | binary XOR output (analog XOR via STE) | ∂ of tanh product | 84.72 | 84.21 |
| STE | sign | identity | 83.58 | 81.42 |

As in the paper, FleXOR training is clearly better than both alternatives.

### Not reproduced

- ImageNet (no data), 500-epoch CIFAR-10 runs, Table 5 (N_out=10 sweep),
  Fig. 6 (`S_tanh` sweep) and Fig. 15b (weight clipping; implemented as
  `flexor.clip` but not run).

## Deviations from the paper

- **LeNet-5 scaling-factor init.** The paper initializes all scaling factors
  to 0.2. LeNet-5 has no BatchNorm, so with ±1 weights each layer amplifies
  activations by ~0.2·sqrt(fan_in) (≈11× for FC1) and training collapses to
  chance accuracy. MNIST configs use `alpha_init: kaiming` (sqrt(2/fan_in));
  ResNet configs keep the paper's 0.2.
- **`S_tanh` ×2 at LR decay in the 500-epoch configs.** The paper states this
  only for its 200-epoch recipe; the Table 1 configs enable it as well
  (`s_tanh.factor=1.0` disables it).
- **Analog XOR baseline.** The paper describes it as using Eq. (3) for forward
  and backward with its real-valued XOR outputs "quantized through STE". We
  implement this as a binary XOR output in the forward pass with the gradient
  of the `tanh` product in the backward pass. Using the real-valued output
  directly as the weight during training gives a similar final accuracy
  (84.65) but binary-weight test accuracy stays at 10–25% until `S_tanh`
  saturates the `tanh` late in training.
- **XOR matrix.** Each row of `M⊕` has `N_tap` distinct 1's chosen at random
  (rows may repeat), seeded per bit plane; layers with equal specs share `M⊕`.
- **Conventions.** `sign(0) = +1`; Boolean 1 ↔ +1. Weights are flattened in
  PyTorch order `[C_out, C_in, k, k]` (the paper uses `[k, k, C_in, C_out]`);
  unused output bits of the last slice are dropped.
- **Weight decay** (1e-5) applies to all parameters (encrypted weights, scaling
  factors and BatchNorm).
- **LeNet-5** uses 'same' padding (FC input 64·7·7); the MNIST epoch count (20)
  is our choice since the paper does not state it.
- **Storage report** excludes the shared `M⊕` and BatchNorm running statistics
  and counts remaining full-precision parameters at 32 bits.
