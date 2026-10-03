# DDR Diabetic Retinopathy: imbalance strategies for grading and lesion segmentation

Masters-thesis code. Two tracks on the DDR dataset, one shared codebase:

| Track | Task | Model | Main metric (picks the winner) |
|---|---|---|---|
| **A** | DR severity grading (5 classes) | EfficientNet-B0 (`timm`) | Quadratic weighted kappa (QWK) |
| **B** | Lesion segmentation (EX, HE, MA, SE) | U-Net + ResNet34 (`segmentation_models_pytorch`) | Mean AUPR over the 4 lesions |

Each track trains a baseline, then **one run per imbalance strategy** where *only* the strategy changes, picks a winner on
the validation split, and evaluates it **once** on the untouched test split and on an external dataset.

> **New here? Follow [`GUIDE.md`](GUIDE.md)**, the step-by-step walkthrough from empty folder to thesis tables.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh      # full pipeline on tiny fake data, CPU, a few minutes: checks your setup
python -m pytest tests -q       # unit checks of the parts that would silently ruin results
```

Then, with the real data in `data/raw/` (see the guide):

```bash
python -m src.prepare grading      --src data/raw/ddr_grading          # preprocess + freeze splits
python -m src.prepare segmentation --src data/raw/ddr_segmentation
bash scripts/run_grading_all.sh                                        # G0-G5
bash scripts/run_seg_all.sh                                            # S0-S5
python -m src.compare --track grading      --split val                 # pick winners on validation
python -m src.compare --track segmentation --split val
python -m src.evaluate --run outputs/runs/<winner> --split test        # once
```

## What is compared

Strategies act on the **training split only**; validation and test keep their natural class balance.

**Track A, grading (class imbalance)**

| Run | Strategy | Implemented in |
|---|---|---|
| G0 | Baseline: plain cross-entropy | |
| G1 | Class-weighted CE (inverse class frequency) | `imbalance.get_loss` |
| G2 | Random oversampling (`WeightedRandomSampler`) | `imbalance.get_sampler` |
| G3 | Random undersampling (majority classes capped at the median class size each epoch) | `imbalance.UnderSampler` |
| G4 | Focal loss (gamma = 2) | `imbalance.FocalLoss` |
| G5 | Oversampling + stronger augmentation for minority classes | `get_sampler` + `data.grading_transform` |

**Track B, segmentation (pixel imbalance)**

| Run | Strategy | Implemented in |
|---|---|---|
| S0 | Baseline: plain BCE per lesion channel | |
| S1 | Weighted BCE (`pos_weight` per lesion from its pixel ratio) | `imbalance.get_loss` |
| S2 | Dice loss | `imbalance.TverskyLoss(0.5, 0.5)` |
| S3 | BCE + Dice | `imbalance.BCEDice` |
| S4 | Tversky loss (alpha 0.3 on false positives, beta 0.7 on misses) | `imbalance.TverskyLoss` |
| S5 | Lesion-aware patch sampling (70% of crops centred on a lesion) | `imbalance.pick_crop` |

Optional last run per track: `configs/<track>/combined.yaml` (best sampler + best loss).
SMOTE-style interpolation is deliberately not used (blending two fundus images does not give a realistic retina).

## Project layout

```
configs/grading|segmentation/   base.yaml + one small YAML per run (only lists what differs)
data/raw/                       extracted Kaggle files, never edited          (git-ignored)
data/processed/                 cropped + resized cache                       (git-ignored)
data/splits/                    frozen split CSVs                             (COMMIT THESE)
src/prepare.py                  inspect raw data, crop/resize, write split CSVs
src/explore.py                  dataset statistics + thesis figures
src/data.py                     GradingDataset, SegmentationDataset, augmentations
src/imbalance.py                samplers, losses, lesion-aware cropping
src/models.py                   build_classifier / build_segmenter
src/train.py                    python -m src.train --config configs/...
src/evaluate.py                 metrics; score a saved run on val / test / external data
src/compare.py                  all runs -> one table + chart per track
src/utils.py                    config loading (inherit), seeding, logging
scripts/                        run_grading_all.sh, run_seg_all.sh, smoke_test.sh, make_fake_data.py
tests/test_core.py              unit checks
outputs/runs/<run_name>/        config.yaml, best.pt, history.csv, val_metrics.json, train.log
```

### Simplifications compared with the architecture guideline

This is a thesis codebase, so it is kept deliberately small. The experimental design (tracks, runs, metrics, rules) is unchanged.

* Flat `src/` instead of sub-packages (`data/`, `imbalance/`, `models/`): same files, fewer imports.
* No Weights & Biases. Every run writes `history.csv` and `train.log` instead.
* No notebooks: `src/explore.py` and `src/compare.py` do the exploration and plotting headless (works on a cluster or Kaggle). Plot from the CSVs in a notebook if you prefer.
* Augmentation uses plain `torchvision` / `numpy` (no extra augmentation library).
* AUPR is computed from a 10,000-bin probability histogram, which is exact up to the bin width and keeps validation fast on 1024 px images (unit-tested against scikit-learn).

## How the rules are enforced

| Rule from the guideline | Where |
|---|---|
| Splits are created once and frozen | `prepare.py` refuses to overwrite `data/splits/*.csv` (needs `--force`) |
| Resampling, weights and pixel ratios come from the training split only | `imbalance.py` only ever sees the training dataset |
| Validation/test never use samplers or augmentation | `data.py`: eval datasets are resize + normalise only |
| Only the imbalance setting changes between runs | `base.yaml` holds everything else; run files only list `imbalance:` |
| Seeds set for Python, NumPy, PyTorch | `utils.seed_everything` (+ seeded sampler and `DataLoader`) |
| Model selection and early stopping use validation | `train.py` never reads a test CSV |
| Test and external sets are evaluated once | `evaluate.py` refuses to overwrite `test_metrics.json` (needs `--force`) |
| Every run folder holds its exact config | `train.py` writes `config.yaml` (including `--set` overrides and seed) |
| Pinned library versions | `requirements.txt` |

## Decision log (fill in during Phase 1, then commit)

- [ ] **Kaggle versions confirmed** (counts below may differ from the published OIA-DDR numbers):
  grading images: ____ (train ____ / val ____ / test ____) &nbsp;|&nbsp; segmentation images: ____ (train ____ / val ____ / test ____)
- [ ] **Ungradable (class 5):** dropped for grading (default in `prepare.py`), so Track A is a 5-class problem. Number removed: ____
- [ ] **Splits:** official folders used as-is / one stratified 70/15/15 split (seed 42). Which one: ____
- [ ] **Already preprocessed?** Grading set image size ____ ; black borders cropped? ____ (`prepare.py` crops again, which is harmless on already-cropped images)
- [ ] **Overlap between tracks:** `prepare.py segmentation` prints how many segmentation *test* images are also in grading train/val: ____ . It only matters if you later combine the tracks.
- [ ] **Preprocessing sizes:** grading 512 px, segmentation 1024 px (train on 512 px crops; MA lesions vanish when shrunk)
- [ ] **`pos_weight` (S1):** softened by `pos_weight_power: 0.5` and capped at 100, because the literal pixel ratio is in the hundreds to thousands. Printed at the start of the S1 run: ____

## Winners (fill in on validation, BEFORE opening the test set)

| Track | Winner | Val main metric | Why (one or two sentences: which strategy rescued the rare class / lesion?) |
|---|---|---|---|
| A, grading | | | |
| B, segmentation | | | |

## Reproducing a table row

Every run folder contains the config that produced it. To redo a run: `python -m src.train --config outputs/runs/<run>/config.yaml --force`.

## Extending

Each extension plugs into one slot; the core pipeline stays untouched.

| Extension | Where it plugs in |
|---|---|
| Stronger backbone (EfficientNet-B3/B4) | `model.name` in `configs/grading/base.yaml` (any `timm` name) |
| New loss or sampler | one branch in `imbalance.get_loss` / `get_sampler` + one YAML file |
| Different segmentation encoder | `model.encoder` in `configs/segmentation/base.yaml` |
| More external test sets | `python -m src.prepare grading --name <set> --external`, then `evaluate.py --csv ... --tag ...` |
| Grad-CAM, ordinal losses, multi-task model, demo app | new script / new branch in `models.py` / separate `app/` folder that imports `src/` |

## Data sources

Both Kaggle sets re-package the OIA-DDR dataset: T. Li et al., *Diagnostic assessment of deep learning algorithms for diabetic
retinopathy screening*, Information Sciences, 2019. Please cite it. Grading: `mariaherrerot/ddrdataset`; segmentation: `sunfish141/ddr-segmentation`.
