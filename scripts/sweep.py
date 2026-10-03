"""Run a list of training jobs sequentially (inside the container).

Usage:
    python scripts/sweep.py sweeps/mnist_lenet5.yaml
    python scripts/sweep.py sweeps/mnist_lenet5.yaml --only nout10 --dry-run

Sweep file format:
    config: configs/mnist_lenet5.yaml   # default config for every run
    set: [schedule.epochs=20]           # overrides applied to every run (optional)
    runs:
      - name: mnist_nout10_nin4_tap2
        set: [flexor.n_out=10, flexor.n_in=4]
      - name: mnist_fp
        config: configs/other.yaml      # per-run config (optional)
        set: [flexor=null]

Runs with ``<out_dir>/<name>/summary.json`` are skipped; runs with only
``last.pt`` are resumed. A failed run is reported and the sweep continues.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_command(sweep: dict, run: dict, out_dir: Path) -> list:
    cmd = [sys.executable, str(REPO_ROOT / "train.py"), "--config", run.get("config", sweep["config"]),
           "--name", run["name"]]
    for expr in sweep.get("set", []) + run.get("set", []):
        cmd += ["--set", expr]
    if (out_dir / run["name"] / "last.pt").exists():
        cmd.append("--resume")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", help="sweep YAML file")
    parser.add_argument("--only", help="run only names containing this substring")
    parser.add_argument("--out-dir", default="runs", help="must match the configs' out_dir")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    out_dir = Path(args.out_dir)
    runs = [r for r in sweep["runs"] if not args.only or args.only in r["name"]]

    done, failed = [], []
    for i, run in enumerate(runs, 1):
        tag = f"[sweep {i}/{len(runs)}] {run['name']}"
        if (out_dir / run["name"] / "summary.json").exists():
            print(f"{tag}: already finished, skipping", flush=True)
            continue
        cmd = build_command(sweep, run, out_dir)
        print(f"{tag}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        status = "ok" if ret == 0 else f"FAILED (exit {ret})"
        print(f"{tag}: {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
        (done if ret == 0 else failed).append(run["name"])

    print(f"[sweep] finished {len(done)}, failed {len(failed)}: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
