"""External validation (Phase 7): score the FROZEN winner and baseline on a dataset they have never seen.

One command does the whole step for one dataset and one track:
  1. preprocess the external images exactly like DDR (src.prepare ... --external)
  2. evaluate the winner and the baseline, for every seed (src.evaluate --csv ...); runs already scored are skipped
  3. print a mean +/- std table and the drop from the DDR test split, and save it as a CSV

No retraining, no fine-tuning, no threshold tuning. Run from anywhere; it works from the repository root.

Examples
  python scripts/external_eval.py --track grading --name aptos \\
      --src /kaggle/input/aptos2019-blindness-detection \\
      --labels /kaggle/input/aptos2019-blindness-detection/train.csv --winner g4_focal

  python scripts/external_eval.py --track segmentation --name idrid \\
      --src "/kaggle/input/idrid/A. Segmentation" --winner s1_weighted_bce
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LESIONS = ["EX", "HE", "MA", "SE"]
METRICS = {
    "grading": ["qwk", "macro_f1", "accuracy", "auc_referable"] + [f"recall_{c}" for c in range(5)],
    "segmentation": ["mean_aupr"] + [f"aupr_{l}" for l in LESIONS] + ["mean_dice", "mean_iou"] + [f"dice_{l}" for l in LESIONS],
}
MAIN = {"grading": "qwk", "segmentation": "mean_aupr"}
BASELINE = {"grading": "g0_baseline", "segmentation": "s0_baseline"}
PREFIX = {"grading": "grading", "segmentation": "seg"}


def sh(cmd):
    print("$", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}")


def run_dir(runs, base, seed):
    return Path(runs) / (base if seed == 42 else f"{base}_s{seed}")  # seed 42 is the un-suffixed run


def mean_std(values):
    s = pd.Series(values, dtype=float)
    return s.mean(), (s.std() if len(s) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["grading", "segmentation"], required=True)
    ap.add_argument("--name", required=True, help="short dataset name, e.g. aptos, idrid, messidor2 (also the output tag)")
    ap.add_argument("--src", required=True, help="folder with the external images (and masks for segmentation)")
    ap.add_argument("--labels", nargs="+", help="grading: the CSV file(s) holding image names and grades (recommended)")
    ap.add_argument("--winner", required=True, help="frozen winner run name, e.g. g4_focal or s1_weighted_bce")
    ap.add_argument("--baseline", help="baseline run name (default g0_baseline / s0_baseline)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--runs", default="outputs/runs", help="folder with the trained runs")
    ap.add_argument("--out", default="outputs", help="where the result CSV is written")
    args = ap.parse_args()
    os.chdir(ROOT)

    track, name = args.track, args.name
    models = [args.winner, args.baseline or BASELINE[track]]
    csv = Path("data/splits") / f"ext_{name}_{PREFIX[track]}.csv"

    # Fail early if a trained model is missing, before any slow work.
    missing = [str(run_dir(args.runs, m, s) / "best.pt") for m in models for s in args.seeds
               if not (run_dir(args.runs, m, s) / "best.pt").exists()]
    if missing:
        sys.exit("Missing trained models (restore outputs/runs from the notebook output first):\n  " + "\n  ".join(missing))

    # 1. preprocess like DDR, one CSV, no split
    if csv.exists():
        print(f"{csv} exists: reusing the preprocessed external set")
    else:
        cmd = [sys.executable, "-m", "src.prepare", track, "--src", args.src, "--name", name, "--external"]
        if args.labels:
            cmd += ["--labels", *args.labels]
        sh(cmd)
    n_images = len(pd.read_csv(csv))
    print(f"\nexternal set '{name}': {n_images} images")

    # 2. evaluate every frozen model once
    for m in models:
        for s in args.seeds:
            run = run_dir(args.runs, m, s)
            if (run / f"{name}_metrics.json").exists():
                print("already evaluated:", run)
                continue
            sh([sys.executable, "-m", "src.evaluate", "--run", str(run), "--csv", str(csv), "--tag", name])

    # 3. tables
    rows = []
    for m in models:
        ext = [json.loads((run_dir(args.runs, m, s) / f"{name}_metrics.json").read_text()) for s in args.seeds]
        ddr = [json.loads((run_dir(args.runs, m, s) / "test_metrics.json").read_text())
               for s in args.seeds if (run_dir(args.runs, m, s) / "test_metrics.json").exists()]
        for metric in METRICS[track]:
            em, es = mean_std([e[metric] for e in ext if metric in e])
            dm, ds = mean_std([d[metric] for d in ddr]) if ddr else (float("nan"), float("nan"))
            rows.append({"model": m, "metric": metric, "ddr_test_mean": dm, "ddr_test_std": ds,
                         f"{name}_mean": em, f"{name}_std": es, "n_seeds": len(ext)})
    df = pd.DataFrame(rows)
    out = Path(args.out) / f"external_{track}_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)

    pd.set_option("display.width", 200)
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    table = pd.DataFrame({
        "model": df["model"], "metric": df["metric"],
        "DDR test": [fmt(a, b) for a, b in zip(df["ddr_test_mean"], df["ddr_test_std"])],
        name: [fmt(a, b) for a, b in zip(df[f"{name}_mean"], df[f"{name}_std"])],
        "change": (df[f"{name}_mean"] - df["ddr_test_mean"]).map("{:+.3f}".format)})
    print(f"\n{track}: DDR test split vs external '{name}' ({n_images} images), mean ± std over seeds {args.seeds}")
    for m in models:
        print(f"\n{m}")
        print(table[table["model"] == m].drop(columns="model").to_string(index=False))
    main_metric = MAIN[track]
    print(f"\nMain metric ({main_metric}) summary; a large drop is a finding to discuss, not something to hide:")
    for m in models:
        r = df[(df["model"] == m) & (df["metric"] == main_metric)].iloc[0]
        print(f"  {m:20s} DDR test {r['ddr_test_mean']:.3f} -> {name} {r[f'{name}_mean']:.3f}  ({r[f'{name}_mean'] - r['ddr_test_mean']:+.3f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
