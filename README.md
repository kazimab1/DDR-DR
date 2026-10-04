# DDR Diabetic Retinopathy: imbalance strategies for grading and lesion segmentation

Masters-thesis code. Two tracks on the DDR dataset, one shared codebase:

| Track | Task | Model | Main metric (picks the winner) |
|---|---|---|---|
| **A** | DR severity grading (5 classes) | EfficientNet-B0 (`timm`) | Quadratic weighted kappa (QWK) |
| **B** | Lesion segmentation (EX, HE, MA, SE) | U-Net + ResNet34 (`segmentation_models_pytorch`) | Mean AUPR over the 4 lesions |

Each track trains a baseline, then **one run per imbalance strategy** where *only* the strategy changes, picks a winner on
the validation split, and evaluates it **once** on the untouched test split and on an external dataset.

> **New here? Follow [`GUIDE.md`](GUIDE.md)**, the step-by-step walkthrough from empty folder to thesis tables.
> Running on Kaggle notebooks? Use [`KAGGLE_GUIDE.md`](KAGGLE_GUIDE.md) instead.
> Winners frozen? Test the grading winner with [`TEST_GRADING.md`](TEST_GRADING.md).

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

### The two tracks at a glance

| | **Track A: DR grading** | **Track B: Lesion segmentation** |
|---|---|---|
| Task | Classify DR severity (5 classes, grades 0-4) | Pixel masks for 4 lesions: EX, HE, MA, SE |
| Dataset (Kaggle) | `mariaherrerot/ddrdataset` (about 13.7k images; official split 6,835 / 2,733 / 4,105) | `sunfish141/ddr-segmentation` (757 images; official split 383 / 149 / 225) |
| Preprocessing | Crop border, resize to 512 px | Crop border, resize to 1024 px, train on 512 px crops |
| Model (fixed for all runs) | EfficientNet-B0, ImageNet-pretrained (`timm`) | U-Net with ResNet34 encoder, ImageNet-pretrained |
| Output | 5 logits | 4 sigmoid channels |
| Winner picked by (validation) | Quadratic weighted kappa (QWK) | Mean AUPR over the 4 lesions |
| Also reported | Macro-F1, per-class recall, confusion matrix, accuracy, AUC for referable DR (grade >= 2) | Per-lesion AUPR, Dice, IoU |

### Where each model is tested

| Stage | Data | What is evaluated |
|---|---|---|
| Model selection | DDR **validation** split | All runs (G0-G5, S0-S5), optionally repeated with seeds 42/43/44 |
| Final test | DDR **test** split, used once | The winner and the baseline of each track |
| External check | IDRiD (both tracks), APTOS 2019 and Messidor-2 (grading only) | The frozen winners, with no retraining |

### Imbalance strategies

Strategies act on the **training split only**; validation and test keep their natural class balance.
In every run only the imbalance strategy changes (model, image size, epochs, augmentation and seed stay fixed).

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
  grading images: 12,522 after dropping ungradable (train 8,765 / val 1,878 / test 1,879) &nbsp;|&nbsp; segmentation images: 757 (train 383 / val 149 / test 225)
- [ ] **Ungradable (class 5):** dropped for grading (default in `prepare.py`), so Track A is a 5-class problem. Number removed: about 1,151 (13,673 published minus 12,522 kept; confirm in the `prepare` log)
- [ ] **Splits:** official folders used as-is / one stratified 70/15/15 split (seed 42). Which one: **grading = one stratified 70/15/15 split (seed 42)**, because the Kaggle copy has no official split, so grading test numbers are not directly comparable to published results on the official DDR test set; the file names carry no patient ID, so patient-level leakage between splits cannot be ruled out. **Segmentation = official split** (383 / 149 / 225).
- [ ] **Already preprocessed?** Grading set image size ____ ; black borders cropped? ____ (`prepare.py` crops again, which is harmless on already-cropped images)
- [ ] **Overlap between tracks:** `prepare.py segmentation` prints how many segmentation *test* images are also in grading train/val: **194 of 225** (755 of the 757 segmentation images exist in the grading set at all). It only matters if you later combine the tracks.
- [ ] **Preprocessing sizes:** grading 512 px, segmentation 1024 px (train on 512 px crops; MA lesions vanish when shrunk)
- [ ] **`pos_weight` (S1):** softened by `pos_weight_power: 0.5` and capped at 100, because the literal pixel ratio is in the hundreds to thousands. Printed at the start of the S1 run: ____

## Winners (fill in on validation, BEFORE opening the test set)

| Track | Winner | Val main metric | Why (one or two sentences: which strategy rescued the rare class / lesion?) |
|---|---|---|---|
| A, grading | **g4_focal** | QWK 0.904 +/- 0.005 (3 seeds; baseline 0.897 +/- 0.006) | Highest mean validation QWK under the pre-declared rule, but the gain over the baseline is only about one standard deviation. Class weighting (g1) gave the best Severe recall (0.611 vs 0.556) at slightly lower QWK, while focal loss lowered Mild recall (0.355 vs 0.450). The samplers (g2, g3, g5) did not beat the baseline on QWK; g5 almost never predicts Mild or Severe (likely because strong augmentation was applied to the minority classes only, so augmentation itself became a cue; not verified). |
| B, segmentation | **s1_weighted_bce** (provisional until the optional combined run s6 is evaluated) | mean AUPR 0.519 +/- 0.004 (3 seeds; baseline 0.444 +/- 0.029) | Highest mean validation AUPR, by more than one standard deviation over s4. The gain comes from MA (0.327 vs 0.055); weighted BCE lowers HE (0.497 vs 0.566) and leaves EX unchanged. Dice (s2) and BCE+Dice (s3) were below the baseline (single seed). |

### Validation results (mean +/- std over seeds 42, 43, 44)

**Track A, grading** (main metric QWK)

| Run | QWK | Mild recall | Severe recall |
|---|---|---|---|
| g4_focal | 0.904 +/- 0.005 | 0.355 +/- 0.033 | 0.546 +/- 0.042 |
| g0_baseline | 0.897 +/- 0.006 | 0.450 +/- 0.165 | 0.556 +/- 0.056 |
| g1_class_weights | 0.893 +/- 0.009 | 0.479 +/- 0.043 | 0.611 +/- 0.028 |

Single seed (42) only: g2_oversample 0.899, g5_oversample_aug 0.896, g3_undersample 0.890.

**Track B, segmentation** (main metric mean AUPR)

| Run | Mean AUPR | EX | HE | MA | SE |
|---|---|---|---|---|---|
| s1_weighted_bce | 0.519 +/- 0.004 | 0.602 +/- 0.027 | 0.497 +/- 0.033 | 0.327 +/- 0.032 | 0.648 +/- 0.039 |
| s4_tversky | 0.501 +/- 0.014 | 0.589 +/- 0.016 | 0.532 +/- 0.031 | 0.296 +/- 0.004 | 0.587 +/- 0.033 |
| s5_lesion_crops | 0.457 +/- 0.015 | 0.588 +/- 0.021 | 0.530 +/- 0.049 | 0.152 +/- 0.037 | 0.558 +/- 0.019 |
| s0_baseline | 0.444 +/- 0.029 | 0.593 +/- 0.022 | 0.566 +/- 0.057 | 0.055 +/- 0.014 | 0.562 +/- 0.084 |

Single seed (42) only: s2_dice 0.370, s3_bce_dice 0.360.

### Test results, Track A grading (stratified 70/15/15 test split of 1,879 images; mean +/- std over seeds 42, 43, 44)

Winner `g4_focal` and baseline `g0_baseline`, each scored once per run on the test split.

| Metric | g0_baseline | g4_focal (winner) |
|---|---|---|
| QWK (main) | 0.902 +/- 0.008 | **0.911 +/- 0.007** |
| Macro-F1 | 0.734 +/- 0.021 | 0.751 +/- 0.012 |
| Accuracy | 0.872 +/- 0.025 | 0.888 +/- 0.005 |
| AUC, referable DR (grade >= 2) | 0.983 +/- 0.002 | 0.983 +/- 0.003 |
| Recall, No DR (0) | 0.940 +/- 0.049 | 0.968 +/- 0.005 |
| Recall, Mild (1) | **0.435 +/- 0.133** | 0.347 +/- 0.018 |
| Recall, Moderate (2) | 0.866 +/- 0.013 | 0.881 +/- 0.017 |
| Recall, Severe (3) | 0.571 +/- 0.125 | 0.590 +/- 0.016 |
| Recall, PDR (4) | 0.818 +/- 0.026 | 0.830 +/- 0.004 |

Reading: focal loss gives a small gain in QWK (+0.009, about one standard deviation), accuracy and macro-F1, in the same direction as on validation (+0.007), and much lower seed-to-seed variance.
It does not help the rarest classes: Mild recall is lower (0.347 vs 0.435, as on validation) and the Severe difference is within the baseline's spread.
Seed-42 confusion matrix (rows true, columns predicted): Mild images are mostly called No DR (32) or Moderate (31); only 32 of 95 are called Mild.
Validation to test shows no sign of over-selection on validation (g4: 0.904 -> 0.911, baseline: 0.897 -> 0.902).
Class weighting (g1), which gave the best rare-class recall on validation, was not declared as a winner and is therefore not evaluated on the test split.

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
