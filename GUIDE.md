# Step-by-step guide

From an empty machine to the thesis tables. Every command is run from the **repository root**.
Steps 0-3 are shared; Steps 4-6 are done per track (A = grading, B = segmentation); Steps 7-8 happen once, at the end.

```
0 Setup -> 1 Data -> 2 Explore -> 3 Preprocess + freeze splits
        -> 4 Baselines -> 5 Imbalance runs -> 6 Compare on val, freeze winners
        -> 7 Test split (once) -> 8 External check -> 9 Write up
```

---

## Step 0. Set up and check the environment

```bash
git clone https://github.com/kazimab1/DDR-DR && cd DDR-DR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tested with Python 3.11
bash scripts/smoke_test.sh               # ends with "SMOKE TEST PASSED"
python -m pytest tests -q                # all tests pass
```

The smoke test runs the whole pipeline (prepare, 12 training runs, compare, validation/test/external evaluation) on tiny synthetic
images, on CPU, in a few minutes. It writes only to `data/smoke/`. If it fails, fix that before touching real data.

**Where to run the real training.** You need a GPU. Options: a university GPU machine, Google Colab, or a Kaggle notebook
(the datasets are already on Kaggle, so you can attach them instead of downloading). On Colab/Kaggle, clone the repo, `pip install`
the requirements (keep the CUDA build of torch that the platform ships if `pip` wants to replace it), and `cd` into the repo.
Save `outputs/` and `data/splits/` somewhere persistent, because these notebook machines are wiped.

## Step 1. Get the data (Phase 1)

```bash
pip install kaggle                       # and put your API token in ~/.kaggle/kaggle.json
kaggle datasets download -d mariaherrerot/ddrdataset    -p data/raw/ddr_grading      --unzip
kaggle datasets download -d sunfish141/ddr-segmentation -p data/raw/ddr_segmentation --unzip
```

(Or download in the browser and unzip into those two folders.) Never edit anything in `data/raw/`.

Now **look at what you actually got**. The Kaggle layouts are not guaranteed to match the original release:

```bash
python -m src.prepare inspect --src data/raw/ddr_grading
python -m src.prepare inspect --src data/raw/ddr_segmentation
```

`inspect` prints the folder tree, the first lines of any CSV/TXT files, and what the readers find (images per split and grade;
images per split with a mask per lesion). **Check these two tables before going on.** They should show roughly 13.7k grading images
(grades 0-5) and 757 segmentation images, with the train/val/test split if the dataset ships one.

If a table is empty or wrong, your copy uses another layout. All raw-layout knowledge lives in two functions in `src/prepare.py`,
`discover_grading()` and `discover_segmentation()` (plus `read_grade_labels()`). They already understand: `train.txt`/`valid.txt`/`test.txt`
files with `name grade` lines, CSVs with an id column and a grade column, images in folders named `0`-`5`, split folders named
`train`/`valid`/`test`, and masks either in a folder per lesion (`.../MA/img.tif`) or with a suffix (`img_MA.tif`). Adjust one of those functions
and re-run `inspect` until the tables look right. Nothing else in the project depends on the raw layout.

## Step 2. Explore the data (Phase 1, thesis dataset chapter)

```bash
python -m src.explore grading      --src data/raw/ddr_grading
python -m src.explore segmentation --src data/raw/ddr_segmentation
```

Writes to `outputs/figures/`: class counts per split (CSV + bar chart), a sample grid per grade, lesion statistics (images
containing each lesion, share of pixels: this is the pixel imbalance), a mask-overlay sample, and prints image sizes, unreadable files and
exact duplicates. **These charts open your thesis dataset chapter.**

Then fill in the **Decision log** in `README.md` (counts, ungradable, splits, already-preprocessed?, overlap).
Defaults: ungradable class 5 is dropped; if the dataset ships official train/val/test folders they are used as-is, otherwise one
stratified 70/15/15 split with seed 42 is made.

## Step 3. Preprocess and freeze the splits (Phase 2)

```bash
python -m src.prepare grading      --src data/raw/ddr_grading          # -> 512 px
python -m src.prepare segmentation --src data/raw/ddr_segmentation     # -> 1024 px (train on 512 px crops)
git add data/splits && git commit -m "Freeze DDR splits"
```

What it does: crops the black border, pads to a square, resizes (JPEG for photos, PNG for masks, cached in `data/processed/`), drops
ungradable images, and writes `data/splits/grading_{train,val,test}.csv` (`image_path, grade`) and
`seg_{train,val,test}.csv` (`image_path, ex_mask, he_mask, ma_mask, se_mask`; an empty cell means that lesion is absent).
It is safe to re-run after an interruption (finished images are skipped), but it **refuses to overwrite existing split CSVs**: splits are
made once. The segmentation step also prints how many segmentation test images appear in the grading train/val sets.

Open `data/splits/*.csv` and confirm the counts match what you recorded in Step 1.

## Step 4. Baselines (Phase 3)

Train the plain baselines first. They prove the pipeline works end to end and set the number every strategy must beat.

```bash
python -m src.train --config configs/grading/g0_baseline.yaml
python -m src.train --config configs/segmentation/s0_baseline.yaml
```

Each run creates `outputs/runs/<run_name>/` with `config.yaml`, `best.pt`, `history.csv`, `train.log`, `val_metrics.json`.

**Check before moving on:**

* Training loss goes down and the validation metric goes up in `history.csv`. If grading QWK stays near 0 after several epochs,
  something is wrong (paths, labels, learning rate), so do not start the sweep.
* It is normal for the baseline to be good at "No DR" and poor at Severe/Mild (grading), or at MA (segmentation). That is the problem you are studying.
* First time on your GPU? Time one epoch into a scratch folder, so it cannot be mistaken for a finished run:
  `python -m src.train --config configs/grading/g0_baseline.yaml --set train.epochs=1 train.out_dir=outputs/timing`,
  then plan the sweep (6 runs per track).

| Problem | Fix |
|---|---|
| `FileNotFoundError` for an image | run from the repo root; the CSVs hold paths relative to it |
| CUDA out of memory | lower `train.batch_size` (grading 8, segmentation 4) **in `base.yaml`**, so every run of the track keeps the same setting |
| Slow epochs | raise `train.num_workers`; keep `train.amp: true` |
| `Missing key` / shape errors loading a run | the run's `config.yaml` is the source of truth; do not mix runs from different `base.yaml` versions |

## Step 5. Imbalance runs (Phase 4)

```bash
bash scripts/run_grading_all.sh      # G0-G5, one after another
bash scripts/run_seg_all.sh          # S0-S5
```

Finished runs are skipped, so you can stop and resume. **Nothing but the `imbalance:` block differs between runs.** Do not tweak
learning rate or epochs for one strategy only; that would make the comparison about tuning rather than imbalance.

To add a strategy later: add a branch in `imbalance.get_loss` or `get_sampler`, add a YAML file next to the others, and run it.

## Step 6. Compare on validation and freeze one winner per track (Phase 5)

```bash
python -m src.compare --track grading      --split val
python -m src.compare --track segmentation --split val
```

Prints a table sorted by the main metric (QWK or mean AUPR) with the gain over the baseline, plus per-class recall (grading) or
per-lesion AUPR (segmentation). Writes `outputs/comparison_<track>_val.csv` and `.png` (the chart is a dot plot with a dashed baseline line).

1. **Look at the rare classes, not just the average.** The thesis story is usually *which strategy rescued Severe NPDR / MA*.
2. **Check that differences are bigger than noise.** Repeat the top 2-3 strategies **and the baseline** with other seeds:
   ```bash
   for s in 43 44; do
     python -m src.train --config configs/grading/g2_oversample.yaml --seed $s
     python -m src.train --config configs/grading/g0_baseline.yaml  --seed $s
   done
   ```
   Extra seeds go to their own folders (`<run>_s43`), and `compare.py` then reports mean +/- std per strategy. (`SEEDS="42 43 44" bash scripts/run_grading_all.sh` repeats everything.)
3. **Optional combined run**: edit `configs/<track>/combined.yaml` to the best sampler/loss and train it (`python -m src.train --config configs/grading/combined.yaml`).
4. **Pick one winner per track and write down why in the README ("Winners" table) before opening the test set.** Commit it. This is your protection against choosing on test results.

## Step 7. Test split, once (Phase 6)

Only now. Evaluate the winner **and the baseline** of each track so the gain over baseline is also on the test set:

```bash
python -m src.evaluate --run outputs/runs/<winner>      --split test
python -m src.evaluate --run outputs/runs/g0_baseline   --split test
python -m src.evaluate --run outputs/runs/<seg_winner>  --split test
python -m src.evaluate --run outputs/runs/s0_baseline   --split test
python -m src.compare --track grading --split test
python -m src.compare --track segmentation --split test
```

`evaluate.py` refuses to overwrite an existing `test_metrics.json`, so a run is scored on test once (`--force` overrides it; do not use it to
re-pick a winner). Thresholds and preprocessing are exactly those used on validation.

## Step 8. External check (Phase 7)

Evaluate the frozen models on data they have never seen, with **no retraining or fine-tuning**:

| Dataset | Use for | Notes |
|---|---|---|
| IDRiD | both tracks (grading 0-4 and lesion masks) | strongest single choice; registration required |
| APTOS 2019 | Track A | Kaggle competition, accept the rules to download |
| Messidor-2 | Track A | convert the adjudicated grades into a CSV with columns `id_code,diagnosis` |

```bash
# images go through the SAME preprocessing as DDR; --external writes one CSV and does no splitting
python -m src.prepare grading      --src data/raw/aptos --name aptos --external
python -m src.prepare segmentation --src data/raw/idrid --name idrid --external   # IDRiD: point --src at the "A. Segmentation" folder

python -m src.evaluate --run outputs/runs/<winner>     --csv data/splits/ext_aptos_grading.csv --tag aptos
python -m src.evaluate --run outputs/runs/<seg_winner> --csv data/splits/ext_idrid_seg.csv     --tag idrid
python -m src.compare  --track grading --split aptos
```

`prepare.py` finds the grade column in a CSV automatically (`diagnosis`, `grade`, `label`, `level`; IDRiD's
`Retinopathy grade` works too) and recognises IDRiD-style masks (`IDRiD_01_MA.tif`). For IDRiD grading, point `--src` at the
`B. Disease Grading` folder. If a layout is not recognised, run `inspect` on it, like in Step 1.
**A large drop from DDR test to external is a finding to discuss, not a failure to hide.**

## Step 9. Write up

| Thesis section | Source |
|---|---|
| Dataset chapter | `outputs/figures/*` from Step 2, the README decision log |
| Methods: strategies and setup | the two strategy tables in `README.md`, `configs/*/base.yaml` |
| Validation results | `outputs/comparison_*_val.csv/.png`, per-class recall / per-lesion AUPR |
| Learning curves | `outputs/runs/<run>/history.csv` |
| Test results | `outputs/comparison_*_test.csv`, `outputs/runs/<run>/test_metrics.json` (includes the confusion matrix for grading) |
| Generalisation | external `*_metrics.json` and the drop from DDR test |
| Reproducibility appendix | `requirements.txt`, `data/splits/*.csv`, each run's `config.yaml` |

## Reference: what the metrics mean

* **QWK** (grading, picks the winner): agreement between predicted and true grade, penalising far-off mistakes more. 1 = perfect, 0 = chance.
  Reported next to macro-F1, per-class recall, confusion matrix, accuracy, and AUC for *referable DR* (grade >= 2 vs < 2).
  Accuracy alone is misleading here: always predicting "No DR" already scores high.
* **AUPR** (segmentation, mean over EX/HE/MA/SE picks the winner): area under the precision-recall curve over all pixels, the standard
  measure in DDR/IDRiD papers. Reported per lesion with Dice and IoU at threshold 0.5.
* **Validation images are segmented at full 1024 px**; only training uses 512 px crops.

---

## File reference: what each file contains

| File | Contains | Used in step |
|---|---|---|
| `requirements.txt` | Pinned library versions | 0 |
| `configs/<track>/base.yaml` | All shared settings of a track: CSV paths, model, epochs, batch size, lr, seed, imbalance defaults | 4-6 |
| `configs/<track>/g*.yaml`, `s*.yaml` | One run each; lists only the `imbalance:` setting that differs from `base.yaml` | 4-6 |
| `configs/<track>/combined.yaml` | Optional final run (best sampler + best loss); edit before running | 6 |
| `src/utils.py` | Config loading (`inherit:`, `--set` overrides, seed suffix), seeding, logger, shared constants | all |
| `src/prepare.py` | `inspect` / `grading` / `segmentation` commands: find raw files, crop and resize, drop ungradable, write frozen split CSVs, external-set mode | 1, 3, 8 |
| `src/explore.py` | Dataset statistics and thesis figures (class counts, lesion pixel share, sample grids, duplicates, unreadable files) | 2 |
| `src/data.py` | `GradingDataset`, `SegmentationDataset`, augmentations (eval data gets resize + normalise only) | 4-8 |
| `src/imbalance.py` | The strategies: oversampling, undersampling, weighted CE, focal, weighted BCE, Dice, BCE+Dice, Tversky, lesion-aware crops | 5 |
| `src/models.py` | `build_classifier` (EfficientNet-B0) and `build_segmenter` (U-Net/ResNet34) | 4-8 |
| `src/train.py` | One training script for both tracks; validates every epoch, saves `best.pt`, early stopping, writes the run folder | 4-5 |
| `src/evaluate.py` | All metrics; CLI to score a saved run on val, test, or an external CSV (test/external once per run) | 6-8 |
| `src/compare.py` | Collects every run's metrics into one sorted table, CSV and dot-plot per track | 6-7 |
| `scripts/run_grading_all.sh`, `run_seg_all.sh` | Loop over all configs of a track; skip finished runs; `SEEDS="42 43 44"` for repeats | 5 |
| `scripts/smoke_test.sh`, `make_fake_data.py` | End-to-end dry run on synthetic data | 0 |
| `tests/test_core.py` | Unit checks of AUPR, samplers, crops, mask alignment, splits | 0 |
| `data/raw/` | Extracted Kaggle files, never edited (git-ignored) | 1 |
| `data/processed/` | Cropped and resized image cache (git-ignored) | 3 |
| `data/splits/*.csv` | Frozen split lists; commit these | 3 |
| `outputs/figures/` | Exploration charts and tables | 2 |
| `outputs/runs/<run>/` | `config.yaml` (exact config), `best.pt`, `history.csv`, `train.log`, `val_metrics.json`, later `test_metrics.json` and external `<tag>_metrics.json` | 4-8 |
| `outputs/comparison_<track>_<split>.csv/.png` | Comparison tables and charts | 6-7 |
