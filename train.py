"""Train FleXOR (or full-precision) models.

Usage (inside the container):
    python train.py --config configs/cifar10_resnet32.yaml
    python train.py --config configs/cifar10_resnet32.yaml --set flexor.n_in=12 --name r32_nin12
    python train.py --config configs/cifar10_resnet32.yaml --set flexor=null   # FP baseline
    python train.py --config configs/cifar10_resnet32.yaml --name r32 --resume
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from data import build_loaders
from flexor import STanhSchedule, XORSpec, export_model, flexor_weights, storage_report, verify_export
from models import build_model

DEFAULTS = {
    "name": None,
    "seed": 0,
    "out_dir": "runs",
    "data": {"name": "cifar10", "root": "/data/torchvision", "batch_size": 128, "test_batch_size": 512,
             "num_workers": 4, "augment": True},
    "model": {"name": "resnet32"},
    # Set to null for a full-precision baseline. ``stages`` (ResNet only) is a
    # list of per-stage overrides merged onto the base spec. ``xor_mode`` is
    # "flexor" | "ste" | "analog" (Fig. 5); ``clip`` (e.g. 2.0) clamps encrypted
    # weights to +-clip/S_tanh after every step (Fig. 15b), null disables it.
    "flexor": {"n_in": 8, "n_out": 20, "q": 1, "n_tap": 2, "seed": 0, "stages": None,
               "init_std": 1.0e-3, "alpha_init": 0.2, "xor_mode": "flexor", "clip": None},
    "optim": {"name": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 1.0e-5},
    "schedule": {"epochs": 200, "warmup_epochs": 0, "milestones": [150, 175], "gamma": 0.5},
    # S_tanh warmup shares ``schedule.warmup_epochs``; ``factor`` is applied at
    # every ``schedule.milestones`` entry.
    "s_tanh": {"base": 10.0, "warmup_start": 5.0, "factor": 2.0},
    "log": {"save_every": 50},
    "debug": {"limit_batches": None},
}


# ----------------------------------------------------------------------------- config

def deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_override(cfg: dict, expr: str) -> None:
    """Apply ``a.b.c=value`` (value parsed as YAML) to ``cfg`` in place."""
    key, _, raw = expr.partition("=")
    if not _:
        raise ValueError(f"override must be key=value: {expr}")
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[leaf] = yaml.safe_load(raw)


def load_config(path: Optional[str], overrides: List[str]) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    for expr in overrides:
        apply_override(cfg, expr)
    if not cfg["name"]:
        cfg["name"] = Path(path).stem if path else "run"
    return cfg


SPEC_KEYS = ("n_in", "n_out", "q", "n_tap", "seed")


def build_spec(fcfg: Optional[dict]) -> Union[None, XORSpec, List[XORSpec]]:
    if not fcfg:
        return None
    base = {k: fcfg[k] for k in SPEC_KEYS}
    if fcfg.get("stages"):
        return [XORSpec(**{**base, **stage}) for stage in fcfg["stages"]]
    return XORSpec(**base)


def build_from_config(cfg: dict) -> nn.Module:
    fcfg = cfg["flexor"]
    kwargs = {"spec": build_spec(fcfg)}
    if fcfg:
        kwargs["flexor_kwargs"] = {"init_std": fcfg["init_std"], "alpha_init": fcfg["alpha_init"],
                                   "s_tanh": cfg["s_tanh"]["base"], "xor_mode": fcfg["xor_mode"]}
    return build_model(cfg["model"]["name"], **kwargs)


# ----------------------------------------------------------------------------- schedules

def lr_factor(epoch: float, warmup_epochs: float, milestones: List[float], gamma: float) -> float:
    """LR multiplier at a fractional epoch: linear warmup from 0, then step decay."""
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return gamma ** bisect_right(milestones, epoch)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    ocfg = cfg["optim"]
    if ocfg["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=ocfg["lr"], momentum=ocfg["momentum"],
                               weight_decay=ocfg["weight_decay"])
    if ocfg["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"])
    raise ValueError(f"unknown optimizer: {ocfg['name']}")


# ----------------------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    correct, loss, n = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    model.train()
    return {"test_loss": loss / n, "test_acc": 100.0 * correct / n}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: dict, resume: bool = False) -> dict:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    ckpt_path = run_dir / "last.pt"
    if run_dir.exists() and not resume and ckpt_path.exists():
        raise FileExistsError(f"{run_dir} already has a checkpoint; use --resume or another --name")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    dcfg, scfg = cfg["data"], cfg["schedule"]
    train_loader, test_loader = build_loaders(
        dcfg["name"], dcfg["root"], dcfg["batch_size"], dcfg["test_batch_size"], dcfg["num_workers"], dcfg["augment"]
    )
    model = build_from_config(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    s_sched = STanhSchedule(cfg["s_tanh"]["base"], cfg["s_tanh"]["warmup_start"], scfg["warmup_epochs"],
                            scfg["milestones"], cfg["s_tanh"]["factor"])
    base_lr = cfg["optim"]["lr"]
    has_flexor = any(True for _ in flexor_weights(model))
    clip = cfg["flexor"]["clip"] if has_flexor else None

    start_epoch, best_acc = 0, 0.0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_acc = ckpt["epoch"], ckpt["best_acc"]
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")

    limit = cfg["debug"]["limit_batches"]
    iters = min(len(train_loader), limit) if limit else len(train_loader)
    writer = SummaryWriter(run_dir / "tb")
    csv_path = run_dir / "metrics.csv"
    fields = ["epoch", "lr", "s_tanh", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc", "time"]
    if not (resume and csv_path.exists()):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    print(f"run: {run_dir} | device: {device} | params: {sum(p.numel() for p in model.parameters()):,}")
    metrics = None
    for epoch in range(start_epoch, scfg["epochs"]):
        t0 = time.time()
        loss_sum, correct, n = 0.0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            if i >= iters:
                break
            e = epoch + i / iters
            lr = base_lr * lr_factor(e, scfg["warmup_epochs"], scfg["milestones"], scfg["gamma"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            s_tanh = s_sched.apply(model, e)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if clip:
                for fw in flexor_weights(model):
                    fw.clip_encrypted(clip)

            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at epoch {e:.3f}")

        metrics = {"train_loss": loss_sum / n, "train_acc": 100.0 * correct / n, **evaluate(model, test_loader, device)}
        best_acc = max(best_acc, metrics["test_acc"])
        row = {"epoch": epoch + 1, "lr": lr, "s_tanh": s_tanh if has_flexor else 0.0, **metrics,
               "best_acc": best_acc, "time": time.time() - t0}
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[k]:.6g}" if isinstance(row[k], float) else row[k] for k in fields])
        for k in ("lr", "s_tanh", "train_loss", "train_acc", "test_loss", "test_acc"):
            writer.add_scalar(k, row[k], epoch + 1)
        print(f"[{epoch + 1:4d}/{scfg['epochs']}] lr={lr:.4g} s_tanh={row['s_tanh']:.3g} "
              f"train {metrics['train_loss']:.4f}/{metrics['train_acc']:.2f}% "
              f"test {metrics['test_loss']:.4f}/{metrics['test_acc']:.2f}% (best {best_acc:.2f}%) {row['time']:.1f}s")

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "best_acc": best_acc, "rng_cpu": torch.get_rng_state(), "config": cfg}
        torch.save(state, ckpt_path)
        if cfg["log"]["save_every"] and (epoch + 1) % cfg["log"]["save_every"] == 0:
            torch.save(state, run_dir / f"epoch{epoch + 1}.pt")
    writer.close()
    if metrics is None:  # resumed a run that had already finished
        s_sched.apply(model, scfg["epochs"])
        metrics = evaluate(model, test_loader, device)

    summary ={"name": cfg["name"], "final_acc": metrics["test_acc"], "best_acc": best_acc}
    if has_flexor:
        exported = export_model(model)
        torch.save(exported, run_dir / "exported.pt")
        report = storage_report(model)
        summary.update(
            export_max_diff=verify_export(model, exported),
            bits_per_weight=report["quantized_bits_per_weight"],
            quantized_compression=report["quantized_compression"],
            model_compression=report["model_compression"],
        )
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry, e.g. --set flexor.n_in=12 (repeatable)")
    parser.add_argument("--name", help="run name (default: config file stem)")
    parser.add_argument("--resume", action="store_true", help="resume from <out_dir>/<name>/last.pt")
    args = parser.parse_args()

    overrides = args.overrides + ([f"name={args.name}"] if args.name else [])
    train(load_config(args.config, overrides), resume=args.resume)


if __name__ == "__main__":
    main()
