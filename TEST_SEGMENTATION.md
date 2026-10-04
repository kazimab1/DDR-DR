# Test the segmentation winner (Track B, Phase 6)

Frozen winner: **s1_weighted_bce** (weighted BCE; provisional in the README until you confirm it in step 0 below).
Baseline for comparison: **s0_baseline**. Both are scored on the held-out DDR **test** split (225 images, full 1024 px), once per run,
for all three seeds (42, 43, 44). The winner must be written in the README "Winners" table and committed **before** you run anything on test.

`evaluate.py` refuses to score a run on test twice (it needs `--force`). Do not use `--force` to change the winner.

## Step 0. Settle the winner (validation only, no test data)

If you trained the optional combined run (`s6_combined`: weighted BCE + lesion-aware crops), compare it with S1 on **validation** first.
This cell only reads validation metrics, so it is safe to run:

```python
import json, yaml
rows = []
for f in glob.glob("outputs/runs/*/config.yaml"):
    cfg = yaml.safe_load(open(f))
    mf = f.replace("config.yaml", "val_metrics.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        rows.append({"run": cfg.get("base_run", cfg["run_name"]), "seed": cfg["train"]["seed"], "mean_aupr": m["mean_aupr"],
                     **{f"aupr_{l}": m[f"aupr_{l}"] for l in ["EX", "HE", "MA", "SE"]}})
df = pd.DataFrame(rows)
ms = lambda x: f"{x.mean():.3f} ± {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"
order = df.groupby("run")["mean_aupr"].mean().sort_values(ascending=False).index
table = df.groupby("run")[["mean_aupr", "aupr_EX", "aupr_HE", "aupr_MA", "aupr_SE"]].agg(ms).loc[order]
table["n_seeds"] = df.groupby("run").size().loc[order]
display(table)
```

* If `s1_weighted_bce` is still first (or the combined run is clearly not better), keep S1 as the winner and **do not test the combined run**.
* If you decide the combined run is the winner, say so in the README first, then replace `"s1_weighted_bce"` below with `"s6_combined"`
  and add seeds 43 and 44 for it (`python -m src.train --config configs/segmentation/combined.yaml --set imbalance.loss=weighted_bce --seed 43 ...`).
  Decide this before testing, never after.

## Setup (Kaggle)

1. Create a **new notebook** by importing `notebooks/kaggle_ddr.ipynb` again (File -> Import Notebook).
   Do not append these cells to your existing notebook and press Run All: in a fresh session it would retrain the seed runs.
2. Session options: **Internet On**, **GPU** (T4 or P100). Evaluation at 1024 px is much faster on GPU.
3. Add Input:
   * the dataset `sunfish141/ddr-segmentation`,
   * **only** the latest output of the notebook that holds your seed runs (**DDR-seg2** in your sidebar), via Add Input -> Notebook Output -> Your Work.
     It contains the splits, the preprocessed images and every run with its `best.pt`.
     Do not also attach older notebook outputs: the restore cell takes the first match.
4. First code cell: `TRACK = "segmentation"`, `QUICK_TEST = False`, `BRANCH = "claude/sweet-sagan-8khcuq"`.
5. Click **Run All**. Check that the restore cell prints `Restoring from /kaggle/input/...`. Finished runs are skipped, so this takes a few minutes
   (the explore step re-runs). If it prints "starting fresh", the output is not attached: stop and fix that.
6. If your seed-run session is still alive, you can add the cells below to it instead.

Sanity check before testing; all six must print `True`:

```python
for name in ["s1_weighted_bce", "s1_weighted_bce_s43", "s1_weighted_bce_s44", "s0_baseline", "s0_baseline_s43", "s0_baseline_s44"]:
    print(name, os.path.exists(f"outputs/runs/{name}/best.pt"))
```

## Cell T1: test split, once per run (safe to re-run after an interruption)

```python
WINNER = "s1_weighted_bce"        # change only if you froze another winner in the README before this step
for name in [WINNER, "s0_baseline"]:
    for suffix in ["", "_s43", "_s44"]:
        run = f"outputs/runs/{name}{suffix}"
        if os.path.exists(f"{run}/test_metrics.json"):
            print("already tested:", run); continue
        sh(f"python -m src.evaluate --run {run} --split test")
```

## Cell T2: mean ± std table on the test split

```python
import json, yaml
rows = []
for f in glob.glob("outputs/runs/*/config.yaml"):
    cfg = yaml.safe_load(open(f))
    mf = f.replace("config.yaml", "test_metrics.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        rows.append({"run": cfg.get("base_run", cfg["run_name"]), "seed": cfg["train"]["seed"],
                     **{k: v for k, v in m.items() if isinstance(v, float)}})
df = pd.DataFrame(rows)
ms = lambda x: f"{x.mean():.3f} ± {x.std():.3f}"
lesions = ["EX", "HE", "MA", "SE"]
aupr = df.groupby("run")[["mean_aupr"] + [f"aupr_{l}" for l in lesions]].agg(ms)
aupr["n_seeds"] = df.groupby("run").size()
print("AUPR (main metric = mean_aupr)")
display(aupr)
other = df.groupby("run")[["mean_dice", "mean_iou"] + [f"dice_{l}" for l in lesions]].agg(ms)
print("Dice and IoU at threshold 0.5")
display(other)
```

## Cell T3: validation vs test (generalisation inside DDR)

```python
for name in [WINNER, "s0_baseline"]:
    val = [json.load(open(f"outputs/runs/{name}{s}/val_metrics.json"))["mean_aupr"] for s in ["", "_s43", "_s44"]]
    test = [json.load(open(f"outputs/runs/{name}{s}/test_metrics.json"))["mean_aupr"] for s in ["", "_s43", "_s44"]]
    print(f"{name}: val mean AUPR {pd.Series(val).mean():.3f}  ->  test mean AUPR {pd.Series(test).mean():.3f}")
```

## Cell T4 (optional): a picture for the thesis

Shows the winner's predictions next to the ground truth on a few test images (seed 42). It reads only the test split; it does not change any result.

```python
import numpy as np, torch, matplotlib.pyplot as plt, random
from src.data import SegmentationDataset
from src.models import build_model
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = yaml.safe_load(open(f"outputs/runs/{WINNER}/config.yaml"))
model = build_model(cfg, pretrained=False).to(dev)
model.load_state_dict(torch.load(f"outputs/runs/{WINNER}/best.pt", map_location=dev)["model"]); model.eval()
ds = SegmentationDataset("data/splits/seg_test.csv")
colors = np.array([[255, 215, 0], [255, 0, 0], [0, 200, 255], [0, 255, 0]])      # EX, HE, MA, SE
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
random.seed(0)
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
for row, i in enumerate(random.sample(range(len(ds)), 3)):
    x, y = ds[i]
    with torch.no_grad():
        p = torch.sigmoid(model(x[None].to(dev)))[0].cpu().numpy() > 0.5
    img = np.clip(x.numpy().transpose(1, 2, 0) * std + mean, 0, 1)
    for col, (name, mask) in enumerate([("ground truth", y.numpy() > 0.5), ("prediction", p)]):
        out = (img * 255).astype(np.uint8).copy()
        for c in range(4):
            out[mask[c]] = colors[c]
        axes[row, col].imshow(out); axes[row, col].axis("off"); axes[row, col].set_title(name)
fig.suptitle("EX yellow  HE red  MA cyan  SE green")
plt.tight_layout(); plt.savefig("outputs/seg_test_examples.png", dpi=120); plt.show()
```

## Save the results

1. **Save Version -> Save & Run All (Commit)**.
2. From the Output tab, download `test_metrics.json` of the six runs, `seg_test_examples.png`, and copy the T2 tables into a text file or screenshot them.
3. Send them to be committed (or commit them yourself) and fill the segmentation test results in the README.

## How to read the result

* Compare the winner with `s0_baseline` **on the test set**. On validation the gain came almost entirely from MA (0.055 -> 0.327), while HE got
  slightly worse (0.566 -> 0.497) and EX did not change. Check whether the same pattern holds on test: that is the finding.
* AUPR is computed from a 10,000-bin probability histogram, so values are exact up to the bin width. Dice and IoU use a fixed threshold of 0.5
  chosen before testing; do not tune it on test.
* MA is tiny, so its scores are the least stable: report its spread across seeds.
* A large drop from validation to test would suggest over-selection on validation and is worth a sentence in the thesis.
* Next: the external check on IDRiD (it has the same four lesion types), see `KAGGLE_GUIDE.md` Step 7.
