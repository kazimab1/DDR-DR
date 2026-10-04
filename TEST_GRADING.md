# Test the grading winner (Track A, Phase 6)

Frozen winner: **g4_focal**. Baseline for comparison: **g0_baseline**. Both are scored on the held-out DDR **test** split, once, for all three seeds
(42, 43, 44). The winner is recorded in the README "Winners" table, which was committed before any test evaluation.

`evaluate.py` refuses to score a run on test twice (it needs `--force`). Do not use `--force` to change the winner.
Segmentation is tested separately, after its winner is final.

## Setup (Kaggle)

1. Create a **new notebook** by importing `notebooks/kaggle_ddr.ipynb` again (File -> Import Notebook).
   Do not append these cells to DDR-Grading and press Run All: in a fresh session it would retrain the seed runs.
2. Session options: **Internet On**, **GPU** (T4 or P100).
3. Add Input:
   * the dataset `mariaherrerot/ddrdataset`,
   * **only** the latest output of your **DDR-Grading** notebook (Add Input -> Notebook Output -> Your Work). It contains the splits, the preprocessed
     images and every run, including seeds 43 and 44, with their `best.pt`. Do not also attach older notebook outputs: the restore cell takes the first match.
4. First code cell: `TRACK = "grading"`, `QUICK_TEST = False`, `BRANCH = "claude/sweet-sagan-8khcuq"`.
5. Click **Run All**. Check that the restore cell prints `Restoring from /kaggle/input/...`. Finished runs are skipped, so this only takes a few minutes
   (the explore step re-runs). If it prints "starting fresh", the output is not attached: stop and fix that.
6. If your DDR-Grading interactive session is still running, you can instead add the cells below to it.

Sanity check before testing:

```python
for name in ["g4_focal", "g4_focal_s43", "g4_focal_s44", "g0_baseline", "g0_baseline_s43", "g0_baseline_s44"]:
    print(name, os.path.exists(f"outputs/runs/{name}/best.pt"))
```
All six must print `True`.

## Cell T1: test split, once per run

```python
for name in ["g4_focal", "g0_baseline"]:
    for suffix in ["", "_s43", "_s44"]:
        sh(f"python -m src.evaluate --run outputs/runs/{name}{suffix} --split test")
```

Run it **once**. If it stops half-way (session limit), run it again: runs that already have a `test_metrics.json` will be refused with a message,
so use the loop below instead, which skips them.

```python
for name in ["g4_focal", "g0_baseline"]:
    for suffix in ["", "_s43", "_s44"]:
        run = f"outputs/runs/{name}{suffix}"
        if os.path.exists(f"{run}/test_metrics.json"):
            print("already tested:", run); continue
        sh(f"python -m src.evaluate --run {run} --split test")
```

## Cell T2: mean +/- std table on the test split

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
cols = ["qwk", "macro_f1", "accuracy", "auc_referable"] + [f"recall_{c}" for c in range(5)]
ms = lambda x: f"{x.mean():.3f} ± {x.std():.3f}"
table = df.groupby("run")[cols].agg(ms)
table["n_seeds"] = df.groupby("run").size()
display(table)
print("Confusion matrix, g4_focal seed 42 (rows = true grade, columns = predicted grade):")
print(pd.DataFrame(json.load(open("outputs/runs/g4_focal/test_metrics.json"))["confusion_matrix"]))
```

## Cell T3: validation vs test (generalisation inside DDR)

```python
for name in ["g4_focal", "g0_baseline"]:
    val = [json.load(open(f"outputs/runs/{name}{s}/val_metrics.json"))["qwk"] for s in ["", "_s43", "_s44"]]
    test = [json.load(open(f"outputs/runs/{name}{s}/test_metrics.json"))["qwk"] for s in ["", "_s43", "_s44"]]
    print(f"{name}: val QWK {pd.Series(val).mean():.3f}  ->  test QWK {pd.Series(test).mean():.3f}")
```

## Save the results

1. **Save Version -> Save & Run All (Commit)**.
2. From the Output tab, download for each of the 6 runs `test_metrics.json`, and the T2 table (copy it into a text file or screenshot it).
3. Send them to be committed (or commit them yourself) and fill the "Test results" section of the README.

## How to read the result

* The test numbers are on the stratified 70/15/15 split (not the official DDR test set), so do not compare them directly with published tables.
* Compare g4_focal with g0_baseline *on the test set*: if the gap is again within about one standard deviation, say that focal loss gave no clear
  QWK gain, and discuss the rare-class recall instead (confusion matrix, Mild and Severe).
* A large drop from validation to test would suggest over-selection on validation and is worth a sentence in the thesis.
* Next: the external check (IDRiD / APTOS / Messidor-2), see `KAGGLE_GUIDE.md` Step 7.
