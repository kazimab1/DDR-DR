# External validation (Phase 7)

Run the **frozen** winners on datasets they have never seen, to show whether they generalise to other hospitals and cameras.

| Track | Winner | Baseline | Seeds |
|---|---|---|---|
| A, grading | `g4_focal` | `g0_baseline` | 42, 43, 44 |
| B, segmentation | `s1_weighted_bce` | `s0_baseline` | 42, 43, 44 |

**Rules (same spirit as the DDR test split)**
* No retraining, no fine-tuning, no threshold tuning. The models, the 0.5 Dice threshold and the preprocessing are exactly those used on DDR.
* Each run is scored once per external dataset (`evaluate.py` refuses to overwrite `<tag>_metrics.json`).
* Report what you get. A large drop from DDR to external is a finding to discuss, not something to hide.

Everything is done by one script, `scripts/external_eval.py`. It is in the repository, so the Kaggle notebook already has it after `git clone`;
nothing needs to be uploaded separately. For one dataset and one track it: preprocesses the external images like DDR, evaluates the winner and the baseline
for every seed (finished ones are skipped), and prints a mean ± std table with the change from the DDR test split. It writes `outputs/external_<track>_<name>.csv`.

## Which datasets

| Dataset | Track | Labels | Where to get it |
|---|---|---|---|
| **IDRiD** | grading **and** segmentation | grades 0-4 (516 images); masks for EX, HE, MA, SE (81 images) | On Kaggle: `dankok/diabetic-retinopathy-image-dataset` (one dataset, both parts: `Disease_Grading` and `Segmentation`). Original source: IDRiD challenge (India), independent of DDR (China). Strongest choice: same four lesion types as DDR |
| **APTOS 2019** | grading | grades 0-4 | Kaggle competition `aptos2019-blindness-detection` (join it and accept the rules, then Add Input). Only `train.csv` has labels (3,662 images) |
| **Messidor-2** | grading | adjudicated grades 0-4 | Request/download from the dataset owners; labels come in a separate sheet, convert it to a CSV with columns `id_code,diagnosis` |

IDRiD and Messidor-2 are not official Kaggle downloads: upload them yourself as a **private Kaggle Dataset** (Datasets -> New Dataset) and respect the licences.
Start with APTOS (easiest, directly on Kaggle) and IDRiD (covers both tracks); add Messidor-2 only if time allows.

## Setup (Kaggle), once per track

Do the grading and segmentation parts in **separate notebooks**, because your runs live in two different notebook outputs.

1. Create a **new notebook** by importing `notebooks/kaggle_ddr.ipynb` again (File -> Import Notebook).
2. Session options: **Internet On**, **GPU** (T4 or P100).
3. Add Input:
   * the external dataset (APTOS competition, or your private IDRiD / Messidor-2 dataset),
   * the real DDR dataset for this track, as in the normal notebook (`mariaherrerot/ddrdataset` or `sunfish141/ddr-segmentation`),
   * **only** the latest output of the notebook that holds your final runs for this track (grading: **DDR-grade2**, segmentation: **DDR-seg3**).
     It contains the splits, preprocessed images and every run with its `best.pt` and its `test_metrics.json`.
4. First code cell: set `TRACK` to `"grading"` or `"segmentation"`, `QUICK_TEST = False`, `BRANCH = "claude/sweet-sagan-8khcuq"`.
5. **Run All.** Check that the restore cell prints `Restoring from /kaggle/input/...`. Finished runs are skipped, so this takes a few minutes.
   If it prints "starting fresh", the output is not attached: stop and fix that.
6. If the session that produced your test results is still running, you can add the cells below there instead.

Sanity check; all six must print `True` (use `s1_weighted_bce` / `s0_baseline` for segmentation):

```python
for name in ["g4_focal", "g4_focal_s43", "g4_focal_s44", "g0_baseline", "g0_baseline_s43", "g0_baseline_s44"]:
    print(name, os.path.exists(f"outputs/runs/{name}/best.pt"))
```

## Find the external files first

Folder layouts differ, so look before running:

```python
!ls /kaggle/input
!find /kaggle/input/<external-dataset> -maxdepth 3 | head -40
```

## Cells to run

### Grading: APTOS 2019

```python
!python scripts/external_eval.py --track grading --name aptos \
    --src /kaggle/input/aptos2019-blindness-detection \
    --labels /kaggle/input/aptos2019-blindness-detection/train.csv \
    --winner g4_focal
```

`--labels` is required here. The competition folder also contains `sample_submission.csv` with a dummy all-zero `diagnosis` column, and without
`--labels` those unlabeled test images would be silently treated as grade 0.

### Grading: IDRiD (Kaggle dataset `dankok/diabetic-retinopathy-image-dataset`)

Add that dataset as an input. In the notebook it is mounted at `/kaggle/input/diabetic-retinopathy-image-dataset` (check with `!ls /kaggle/input`).
Use only the `Disease_Grading` part, and pin the two label files in `Groundtruths` (training and testing labels) with `--labels`:

```python
base = "/kaggle/input/diabetic-retinopathy-image-dataset/Disease_Grading"
labels = sorted(glob.glob(f"{base}/Groundtruths/*.csv"))
print(labels)                      # expect exactly two CSV files: training labels and testing labels
sh(f"python scripts/external_eval.py --track grading --name idrid --src '{base}' "
   f"--labels {' '.join(repr(l) for l in labels)} --winner g4_focal")
```

All 516 images (413 training + 103 testing in IDRiD's own split) are used as one external test set. The class mix differs a lot from DDR
(roughly: many more Severe and PDR images, only about 25 Mild), so recall for Mild is very noisy here; look at QWK and the referable-DR AUC first.

### Grading: Messidor-2

```python
!python scripts/external_eval.py --track grading --name messidor2 \
    --src /kaggle/input/<messidor2-images> \
    --labels /kaggle/input/<messidor2-labels>/labels.csv \
    --winner g4_focal
```
The CSV needs an image-name column and a grade column (`id_code,diagnosis` works). Images may be `.tif`, `.png` or `.jpg`.

### Segmentation: IDRiD (same Kaggle dataset)

Use the `Segmentation` part of the same dataset. Images are in `Segmentation/Original_Images/{Training Set,Testing Set}` and the masks in
`Segmentation/Segmentation_Groundtruths/{Training Set,Testing Set}/<lesion folder>/IDRiD_xx_MA.tif`. The folder names (Microaneurysms, Haemorrhages,
Hard Exudates, Soft Exudates) are recognised, and the Optic Disc masks are ignored:

```python
sh("python scripts/external_eval.py --track segmentation --name idrid "
   "--src '/kaggle/input/diabetic-retinopathy-image-dataset/Segmentation' --winner s1_weighted_bce")
```

All 81 images (54 training + 27 testing) are used. Soft-exudate masks exist only for the images that contain SE; images without an SE mask file are treated as having no SE.
`prepare` prints how many images have a mask for each lesion: check these counts before trusting the AUPR of a lesion with few examples.

> The dataset `pallavidi/dr-segmentation` is another copy of the IDRiD segmentation images (folders `1. Original images`, `2. All Segmentation Groundtruths`).
> It also works with the same command (`--src` at its `Dr_dataset` folder), but it is the same data, so using both would not add anything.
> Cite the original IDRiD paper: P. Porwal et al., *Indian Diabetic Retinopathy Image Dataset (IDRiD)*, Medical Image Analysis, 2020 (check the dataset page for the exact licence).

Each command prints, for the winner and the baseline, every metric on DDR test and on the external set, with the change, then a summary of the main metric
(QWK for grading, mean AUPR for segmentation). Run one command per dataset.

## Save the results

1. **Save Version -> Save & Run All (Commit)**.
2. From the Output tab, download `outputs/external_<track>_<name>.csv` and the `<name>_metrics.json` of every run, and copy the printed tables into a text file or screenshot them.
3. Send them to be committed (or commit them yourself) and fill the "External results" section of the README.

## How to read the result

* **Compare the winner with the baseline on the external set**, not only with its own DDR score. The question is whether the strategy's advantage survives a change of hospital and camera.
* **Expect a drop.** Domain shift (cameras, image quality, labelling protocol) usually lowers every score. Report it per dataset and discuss likely causes. Do not tune anything to recover it.
* **Grading:** look at per-class recall and the referable-DR AUC (grade >= 2 vs < 2), which is the clinically important decision; QWK can fall while the referral AUC holds up.
* **Segmentation:** IDRiD has few images (81), so AUPR will be noisy; report the spread across seeds. If a lesion is absent from the whole external set its AUPR is NaN and is left out of the mean. MA is the least stable.
* Labelling protocols differ between datasets (for example, what counts as a haemorrhage vs a microaneurysm), which is a limit on how far the numbers can be compared.
* This is the last experiment. After it: write up (see `GUIDE.md` Step 9).
