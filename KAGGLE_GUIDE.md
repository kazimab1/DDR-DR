# Running the whole project on Kaggle notebooks

This is the Kaggle version of [`GUIDE.md`](GUIDE.md). The steps and the science are identical; only setup, data access
and persistence differ. Read `GUIDE.md` for *why* each step exists. This file is about *how to do it on Kaggle*.

> Written from Kaggle's documented behaviour, **not tested on Kaggle**: menu names and limits change, so trust what you see on screen
> over this file if they disagree. The code itself was tested locally on synthetic data (`scripts/smoke_test.sh`).

> **Shortcut:** `notebooks/kaggle_ddr.ipynb` does Steps 1-6 below automatically for one track (set `TRACK` in its first code cell, attach the dataset, Run All).
> This guide explains what it does and covers the manual steps (test split, external sets, seeds).

## What Kaggle changes

| Topic | Kaggle behaviour | What to do |
|---|---|---|
| Data | Datasets you attach are **read-only** under `/kaggle/input/` | No download needed; point `--src` straight at them |
| Writable disk | Only `/kaggle/working/` (limited size, about 20 GB) | Keep the repo, `data/processed` and `outputs` there |
| Persistence | Files in `/kaggle/working` survive only when you **Save Version**; an idle or ended session loses them | Save versions often (see "Surviving session limits") |
| Session length | A GPU session stops after a maximum number of hours, and there is a **weekly GPU quota** | Split the work over several notebooks/sessions; runs resume (finished runs are skipped) |
| Internet | Off by default; turning it on needs a verified phone number | Needed to `git clone`, `pip install`, and download pretrained weights |
| Libraries | torch, numpy, pandas, etc. are preinstalled | Do **not** `pip install -r requirements.txt` (it would replace torch); install only `timm` and `segmentation-models-pytorch` |

## Plan: three notebooks

Run the two tracks in **separate notebooks** so a session limit never costs you both.

| Notebook | Accelerator | Does |
|---|---|---|
| `ddr-0-setup-check` | None (CPU) | Install, unit tests, smoke test. Costs no GPU quota |
| `ddr-grading` | GPU (T4 or P100) | Step 3 (grading) + Track A runs + comparison |
| `ddr-segmentation` | GPU (T4 or P100) | Step 3 (segmentation) + Track B runs + comparison |

The two tracks do not share intermediate files, so they can be done in any order. External sets and the test split are evaluated at the end (see Step 7).

---

## Step 0. Create the notebook and attach data

1. Kaggle -> **Create** -> **New Notebook**.
2. Right panel -> **Session options**: set **Internet: On**; set **Accelerator** (None for `ddr-0-setup-check`, GPU for the others).
3. **Add Input** -> search and add:
   * `mariaherrerot/ddrdataset` (for the grading notebook)
   * `sunfish141/ddr-segmentation` (for the segmentation notebook)
4. Look at what was mounted:
   ```python
   !ls /kaggle/input
   !ls /kaggle/input/ddrdataset | head      # use the folder name you actually see
   ```
   Note the exact folder names; the commands below use `/kaggle/input/ddrdataset` and `/kaggle/input/ddr-segmentation`.

## Step 1. Get the code and install

Cell 1:

```python
!git clone https://github.com/kazimab1/DDR-DR /kaggle/working/DDR-DR
%cd /kaggle/working/DDR-DR
!git checkout claude/sweet-sagan-8khcuq        # drop this line once the branch is merged to main
!pip install -q timm segmentation-models-pytorch pyyaml tqdm
!python -c "import torch, timm, segmentation_models_pytorch as s; print(torch.__version__, torch.cuda.is_available(), timm.__version__, s.__version__)"
```

* `pyyaml`, `scikit-learn`, `matplotlib`, `pandas`, `pillow` come preinstalled. `requirements.txt` pins the versions the project was
  tested with; Kaggle's preinstalled versions are newer or older, which is normally fine. If something breaks, note the versions printed above.
* Keep this `%cd` in the **first cell of every notebook session**: each new session starts from scratch.
* Don't have internet? Upload the repo as a Kaggle Dataset, add it as input, and `cp -r /kaggle/input/<name> /kaggle/working/DDR-DR`.
  You still need internet once per session for the pretrained weights (`pretrained: true` downloads ImageNet weights).

## Step 2. Dry run (in `ddr-0-setup-check`, no GPU)

```python
!python -m pytest tests -q
!bash scripts/smoke_test.sh        # ends with "SMOKE TEST PASSED", a few minutes on CPU
```

The smoke test writes only to `data/smoke/`. Do this once; there is no need to repeat it in the GPU notebooks.

## Step 3. Inspect and explore the data (Phase 1)

```python
!python -m src.prepare inspect --src /kaggle/input/ddrdataset
!python -m src.explore grading --src /kaggle/input/ddrdataset
```
(Segmentation notebook: use `/kaggle/input/ddr-segmentation` and `explore segmentation`.)

Check the `inspect` tables as described in `GUIDE.md` Step 1: about 13.7k grading images with grades 0-5, about 757 segmentation images
with a mask per lesion. If they are empty or wrong, the layout differs from the one `prepare.py` expects: edit `discover_grading()` /
`discover_segmentation()` in `src/prepare.py` (a Kaggle notebook has a file editor, or use `%%writefile` / `sed`) and re-run `inspect`.

Show the figures in the notebook:

```python
from IPython.display import Image, display
display(Image("outputs/figures/grading_class_counts.png"))
display(Image("outputs/figures/grading_samples.png"))
```

Fill in the decision log (see Step 8 for where to keep notes).

## Step 4. Preprocess and freeze the splits (Phase 2)

```python
# grading notebook
!python -m src.prepare grading --src /kaggle/input/ddrdataset

# segmentation notebook
!python -m src.prepare segmentation --src /kaggle/input/ddr-segmentation
```

Output goes to `/kaggle/working/DDR-DR/data/processed/` (images) and `data/splits/` (CSVs). The raw input is never touched.

**Save the splits immediately.** They must never be regenerated, and a lost session would lose them:

1. Click **Save Version** -> **Save & Run All (Commit)** *or* **Quick Save** (see "Surviving session limits"), then
2. download `data/splits/*.csv` from the notebook's **Output** tab and commit them to your GitHub repo from your own computer.

In later sessions you restore them instead of re-running `prepare` with new random state (`prepare.py` refuses to overwrite splits, and the
official-split case is deterministic anyway, but the stratified fallback must stay identical across sessions).

## Step 5. Baseline (Phase 3)

Time one epoch first, into a scratch folder:

```python
!python -m src.train --config configs/grading/g0_baseline.yaml --set train.epochs=1 train.out_dir=outputs/timing train.num_workers=2
```

Use that time to plan: `epochs x number of runs (6 per track)`. Then run the real baseline:

```python
!python -m src.train --config configs/grading/g0_baseline.yaml --set train.num_workers=2
```

Kaggle notes:

* **`train.num_workers=2`**: Kaggle notebooks have few CPU cores. If data loading is the bottleneck, try 4. This setting does not change results,
  so it is safe to pass on the command line for every run (put it in `base.yaml` if you want it permanent).
* **Out of memory:** lower `train.batch_size` **in `configs/<track>/base.yaml`** so all runs of the track share it (grading 8, segmentation 4).
  Don't change it for a single run.
* `train.amp: true` (mixed precision) is already the default and helps on T4/P100.
* A GPU with `T4 x2` is fine; the code uses one GPU.
* Check `outputs/runs/g0_baseline/history.csv`: loss should fall and the validation metric rise.

## Step 6. Imbalance runs, compare, freeze winners (Phases 4-5)

```python
!bash scripts/run_grading_all.sh --set train.num_workers=2      # G0-G5; finished runs are skipped
!python -m src.compare --track grading --split val
```
(Segmentation notebook: `run_seg_all.sh`, `--track segmentation`.)

The script prints one tqdm bar per epoch; to keep the notebook log readable you can append `2>&1 | grep -v "it/s"`... or simply scroll past it.

If a session ends mid-sweep, start a new session, restore `outputs/runs` (see below) and run the same command again: runs that already have a
`val_metrics.json` are skipped, the interrupted one restarts from the beginning.

Seed repeats for the top strategies and the baseline (see `GUIDE.md` Step 6) work the same way:

```python
for s in (43, 44):
    !python -m src.train --config configs/grading/g2_oversample.yaml --seed $s --set train.num_workers=2
    !python -m src.train --config configs/grading/g0_baseline.yaml  --seed $s --set train.num_workers=2
```

Show the comparison chart:

```python
import pandas as pd
display(pd.read_csv("outputs/comparison_grading_val.csv", index_col=0))
display(Image("outputs/comparison_grading_val.png"))
```

Then write the winner and the reason in the README's "Winners" table **before** touching the test set (edit it locally and commit on GitHub).

## Step 7. Test split once, then external sets (Phases 6-7)

Do this at the very end, in a notebook that has the **frozen winners** (`best.pt`) available.

```python
!python -m src.evaluate --run outputs/runs/<winner>    --split test
!python -m src.evaluate --run outputs/runs/g0_baseline --split test
!python -m src.compare --track grading --split test
```

External sets: attach them as inputs (**Add Input**): APTOS 2019 is a Kaggle competition (join it and accept the rules first, then add it as
input); IDRiD and Messidor-2 are not official Kaggle downloads, so upload them yourself as a private Kaggle Dataset (**Datasets -> New Dataset**; mind
the datasets' licences). Then:

```python
!python -m src.prepare grading --src /kaggle/input/aptos2019-blindness-detection --name aptos --external
!python -m src.evaluate --run outputs/runs/<winner> --csv data/splits/ext_aptos_grading.csv --tag aptos
```

Run the same `prepare ... --external` first so the images get exactly the DDR preprocessing. See `GUIDE.md` Step 8 for IDRiD and Messidor-2 details.

---

## Surviving session limits

Everything you need to keep lives in `/kaggle/working/DDR-DR/`: `data/splits/`, `outputs/runs/`, `outputs/comparison_*`, `outputs/figures/`.
`data/processed/` is a cache you can rebuild with `prepare.py`, but it costs time and should not be re-made after the splits are frozen
unless the files are missing.

**Pattern: Save Version, then reuse it as input.**

1. When a run batch is done (or before the session's time is nearly up), **Save Version -> Save & Run All (Commit)**. Whatever is in
   `/kaggle/working` at the end becomes the notebook's **Output**.
   * For a long sweep, put the whole sweep in the notebook cells and use *Save & Run All*: it then runs in the background for the full
     session even if you close the browser, and saves the output at the end. If any cell errors, the commit fails, so test the cells first
     with a short run (`--set train.epochs=1 train.out_dir=outputs/timing`).
2. In the next session, **Add Input -> Notebook Output** and pick your previous notebook. Its files appear read-only under `/kaggle/input/<notebook-slug>/`.
   Check the exact path with `!ls /kaggle/input/<notebook-slug>`.
3. Restore into the fresh clone (adjust the source path to what `ls` showed):
   ```python
   !mkdir -p outputs data/splits data/processed
   !cp -r /kaggle/input/<notebook-slug>/DDR-DR/outputs/runs   outputs/
   !cp -r /kaggle/input/<notebook-slug>/DDR-DR/data/splits/.  data/splits/
   !cp -r /kaggle/input/<notebook-slug>/DDR-DR/data/processed/. data/processed/
   ```
4. Re-run the sweep command; finished runs are skipped.

`best.pt` files are the largest items (U-Net/ResNet34 is much bigger than EfficientNet-B0), so check the output size against the disk limit.
Keep `best.pt` of at least the baseline and the winner of each track: the test and external evaluation need them.

**Backup that does not depend on Kaggle:** from the Output tab, download `data/splits/`, `outputs/runs/*/{config.yaml,history.csv,val_metrics.json,train.log}`
and the comparison CSV/PNG files, and commit them to your repo. The `.pt` weights are git-ignored; keep them as a private Kaggle Dataset.

## Quick checklist

- [ ] Internet on, accelerator set, datasets attached
- [ ] `%cd /kaggle/working/DDR-DR` is the first cell of every session
- [ ] `timm` and `segmentation-models-pytorch` installed, **no** `pip install -r requirements.txt`
- [ ] `inspect` tables look right before `prepare`
- [ ] Splits saved/downloaded right after `prepare`, then committed to GitHub
- [ ] `train.num_workers=2` (or 4) on the command line; batch size changes only in `base.yaml`
- [ ] Save Version after each batch of runs; restore `outputs/runs` and `data/splits` in the next session
- [ ] Winners written in the README before running `--split test`
