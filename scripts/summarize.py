"""Collect ``runs/*/summary.json`` into a Markdown table.

Usage:
    python scripts/summarize.py                 # all runs
    python scripts/summarize.py --match mnist   # names containing "mnist"
"""

import argparse
import json
from pathlib import Path

import yaml

COLUMNS = ["name", "model", "n_out", "n_in", "q", "n_tap", "bits", "epochs", "best_acc", "final_acc",
           "export_diff", "compression"]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-2 else f"{v:.2f}"
    return str(v)


def collect(out_dir: Path, match: str) -> list:
    rows = []
    for summary_path in sorted(out_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        if match and match not in run_dir.name:
            continue
        summary = json.loads(summary_path.read_text())
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        fcfg = cfg.get("flexor") or {}
        rows.append({
            "name": run_dir.name,
            "model": cfg["model"]["name"],
            "n_out": fcfg.get("n_out"),
            "n_in": fcfg.get("n_in"),
            "q": fcfg.get("q"),
            "n_tap": (fcfg.get("n_tap") or "rand") if fcfg else None,
            "bits": summary.get("bits_per_weight", 32.0),
            "epochs": cfg["schedule"]["epochs"],
            "best_acc": summary["best_acc"],
            "final_acc": summary["final_acc"],
            "export_diff": summary.get("export_max_diff"),
            "compression": summary.get("quantized_compression"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--match", default="", help="only runs whose name contains this substring")
    args = parser.parse_args()

    rows = collect(Path(args.out_dir), args.match)
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(fmt(row[c]) for c in COLUMNS) + " |")


if __name__ == "__main__":
    main()
