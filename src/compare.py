"""Phase 5/6: collect every run's metrics into one table and one bar chart per track.

    python -m src.compare --track grading --split val
    python -m src.compare --track segmentation --split val
    python -m src.compare --track grading --split test     # after the winners are frozen

Runs repeated with other seeds (folders named <run>_s43, ...) are grouped: mean +/- std.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import LESIONS, MAIN_METRIC, NUM_GRADES  # noqa: E402

BASELINE = {"grading": "g0_baseline", "segmentation": "s0_baseline"}
DETAIL = {"grading": [f"recall_{c}" for c in range(NUM_GRADES)],
          "segmentation": [f"aupr_{l}" for l in LESIONS]}  # what the thesis story is usually about


def collect(runs_dir, track, split):
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        cfg_file, metrics_file = run / "config.yaml", run / f"{split}_metrics.json"
        if not (cfg_file.exists() and metrics_file.exists()):
            continue
        cfg = yaml.safe_load(cfg_file.read_text())
        if cfg["task"] != track:
            continue
        metrics = json.loads(metrics_file.read_text())
        rows.append({"run": cfg.get("base_run", cfg["run_name"]), "seed": cfg["train"]["seed"],
                     **{k: v for k, v in metrics.items() if isinstance(v, float)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["grading", "segmentation"], required=True)
    ap.add_argument("--split", default="val", help="reads <run>/<split>_metrics.json (val, test, or an external tag)")
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    main_name = MAIN_METRIC[args.track]
    df = collect(args.runs, args.track, args.split)
    if df.empty:
        raise SystemExit(f"No {args.split}_metrics.json found for track '{args.track}' in {args.runs}")

    metric_cols = [c for c in df.columns if c not in ("run", "seed")]
    mean = df.groupby("run")[metric_cols].mean()
    std = df.groupby("run")[metric_cols].std().fillna(0.0)
    mean.insert(0, "n_seeds", df.groupby("run").size())
    mean = mean.sort_values(main_name, ascending=False)
    base = BASELINE[args.track]
    if base in mean.index:
        mean.insert(2, f"{main_name}_vs_baseline", mean[main_name] - mean.loc[base, main_name])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{args.track}_{args.split}"
    mean.round(4).to_csv(out_dir / f"{stem}.csv")

    pd.set_option("display.width", 200)
    print(f"\n{args.track} - {args.split} - sorted by {main_name}")
    show = [main_name] + [c for c in mean.columns if c.endswith("_vs_baseline")]
    print(mean[["n_seeds"] + show].round(4).to_string())
    print(f"\n{'per-class recall' if args.track == 'grading' else 'per-lesion AUPR'} "
          "(look at the rare classes, not just the average):")
    print(mean[[c for c in DETAIL[args.track] if c in mean.columns]].round(3).to_string())

    order = mean.index[::-1].tolist()  # best run on top
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(order) + 1.6))
    ax.errorbar(mean.loc[order, main_name], range(len(order)), xerr=std.loc[order, main_name],
                fmt="o", color="#4c78a8", capsize=3)
    if base in mean.index:
        ax.axvline(mean.loc[base, main_name], ls="--", color="gray", label=f"baseline ({base})")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(f"{main_name}  (mean +/- std over seeds)" if mean["n_seeds"].max() > 1 else main_name)
    ax.set_title(f"{args.track}: {main_name} ({args.split})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    print(f"\nwrote {out_dir / (stem + '.csv')} and {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
