"""Fast checks of the parts where a silent bug would invalidate the thesis results.

    python -m pytest tests -q          (the end-to-end check is scripts/smoke_test.sh)
"""
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import average_precision_score

from src.evaluate import AUPR_BINS, binned_ap
from src.imbalance import BCEDice, TverskyLoss, UnderSampler, get_sampler, minority_classes, pick_crop
from src.prepare import crop_pad_resize, lesion_of, retina_box


def test_binned_aupr_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.01                       # 1% positives, like a rare lesion
    p = np.clip(0.02 + 0.5 * y * rng.random(len(y)) + 0.05 * rng.random(len(y)), 0, 1)
    idx = torch.as_tensor((p * (AUPR_BINS - 1)).round(), dtype=torch.long)
    t = torch.as_tensor(y)
    pos = torch.bincount(idx[t], minlength=AUPR_BINS)
    neg = torch.bincount(idx[~t], minlength=AUPR_BINS)
    assert binned_ap(pos, neg) == pytest.approx(average_precision_score(y, p), abs=0.01)


def test_oversampler_balances_classes_and_undersampler_caps_them():
    labels = np.array([0] * 600 + [1] * 60 + [2] * 300 + [3] * 20 + [4] * 90)
    drawn = Counter(labels[i] for i in get_sampler("oversample", labels, seed=0))
    assert len(drawn) == 5 and max(drawn.values()) / min(drawn.values()) < 1.5   # ~equal per class
    sampler = get_sampler("undersample", labels, seed=0)
    drawn = Counter(labels[i] for i in sampler)
    assert drawn[0] == drawn[2] == 90 and drawn[3] == 20 and len(sampler) == sum(drawn.values())  # cap = median
    assert list(sampler) != list(sampler)               # a fresh draw every epoch
    assert minority_classes(labels) == [1, 3]           # < median (90)


def test_lesion_aware_crops_contain_lesions():
    random.seed(0)
    masks = np.zeros((4, 256, 256), np.uint8)
    masks[2, 200:203, 30:33] = 1                         # one tiny MA-like blob
    def hit_rate(prob):
        hits = 0
        for _ in range(300):
            y, x = pick_crop(masks, 64, prob)
            hits += masks[:, y:y + 64, x:x + 64].any()
        return hits / 300
    assert hit_rate(1.0) == 1.0 and hit_rate(0.0) < 0.1


def test_tversky_with_equal_weights_is_dice_and_beta_penalises_misses():
    torch.manual_seed(0)
    logits, y = torch.randn(2, 4, 16, 16), (torch.rand(2, 4, 16, 16) < 0.1).float()
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum((0, 2, 3)) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)))).mean()  # textbook soft Dice
    assert TverskyLoss(0.5, 0.5, smooth=1e-6)(logits, y) == pytest.approx(float(dice), abs=1e-4)
    empty = torch.full((1, 4, 8, 8), -9.0)               # predicts nothing
    target = torch.zeros(1, 4, 8, 8); target[:, :, 2:4, 2:4] = 1
    assert TverskyLoss(0.3, 0.7)(empty, target) > TverskyLoss(0.7, 0.3)(empty, target)  # misses cost more
    assert BCEDice()(logits, y) > 0


def test_lesion_name_detection():
    assert lesion_of(Path("seg/train/label/MA/007-1.tif")) == ("MA", "007-1")
    assert lesion_of(Path("IDRiD_01_HE.tif")) == ("HE", "IDRiD_01")
    assert lesion_of(Path("a/3. Hard Exudates/IDRiD_01.tif")) == ("EX", "IDRiD_01")
    assert lesion_of(Path("IDRiD_01_OD.tif")) is None and lesion_of(Path("train/image/007-1.jpg")) is None


def test_crop_resize_keeps_image_and_mask_aligned_and_is_repeatable():
    img = Image.new("RGB", (300, 200), 0)
    img.paste((200, 100, 50), (60, 20, 240, 180))        # retina area inside a black border
    mask = Image.new("L", (300, 200), 0)
    mask.paste(255, (100, 60, 130, 90))
    img.paste((255, 255, 255), (100, 60, 130, 90))       # lesion is visible in the image too
    box = retina_box(img)
    assert box == (60, 20, 240, 180)
    out_img = np.asarray(crop_pad_resize(img, box, 128, Image.BICUBIC).convert("L"))
    out_mask = np.asarray(crop_pad_resize(mask, box, 128, Image.BILINEAR)) >= 64
    bright = out_img > 190                               # the lesion as seen in the image (retina is ~125)
    iou = (bright & out_mask).sum() / (bright | out_mask).sum()
    centre = lambda m: np.argwhere(m).mean(0)
    assert iou > 0.7 and np.abs(centre(bright) - centre(out_mask)).max() < 1.0   # same place, within 1 px
    once = crop_pad_resize(img, box, 128, Image.BICUBIC)
    twice = crop_pad_resize(once, retina_box(once), 128, Image.BICUBIC)   # data that is "already preprocessed"
    assert once.size == twice.size == (128, 128)
    assert np.abs(np.asarray(once, int) - np.asarray(twice, int)).max() <= 2   # processing it again is a no-op


def test_prepare_refuses_to_overwrite_frozen_splits(tmp_path):
    (tmp_path / "ddr_grading").mkdir()
    for g in range(2):
        Image.new("RGB", (40, 40), (90, 40, 20)).save(tmp_path / "ddr_grading" / f"im{g}.jpg")
    (tmp_path / "ddr_grading" / "labels.csv").write_text("id_code,diagnosis\nim0,0\nim1,1\n")
    cmd = [sys.executable, "-m", "src.prepare", "grading", "--src", str(tmp_path / "ddr_grading"), "--size", "32",
           "--out-dir", str(tmp_path / "proc"), "--splits-dir", str(tmp_path / "splits"), "--external"]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode != 0 and "frozen" in second.stderr


def test_stratified_split_is_disjoint_complete_and_keeps_class_proportions(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    counts, rows = [60, 20, 40, 12, 16], ["id_code,diagnosis"]
    for grade, n in enumerate(counts):
        for i in range(n):
            Image.new("RGB", (24, 24), (90, 40, 20)).save(src / f"g{grade}_{i}.png")
            rows.append(f"g{grade}_{i},{grade}")
    (src / "labels.csv").write_text("\n".join(rows) + "\n")
    cmd = [sys.executable, "-m", "src.prepare", "grading", "--src", str(src), "--size", "16",
           "--out-dir", str(tmp_path / "proc"), "--splits-dir", str(tmp_path / "splits")]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    import pandas as pd
    parts = {s: pd.read_csv(tmp_path / "splits" / f"grading_{s}.csv") for s in ("train", "val", "test")}
    paths = [set(df["image_path"]) for df in parts.values()]
    assert sum(map(len, paths)) == sum(counts) == len(set.union(*paths))        # complete, no overlap
    assert abs(len(parts["train"]) / sum(counts) - 0.70) < 0.03
    for df in parts.values():                                                    # every class in every split
        assert sorted(df["grade"].unique()) == [0, 1, 2, 3, 4]
    rare_share = [(df["grade"] == 3).mean() for df in parts.values()]
    assert max(rare_share) - min(rare_share) < 0.03                              # same class mix everywhere


def test_stratified_split_explains_itself_when_a_class_is_too_small(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    rows = ["id_code,diagnosis"] + [f"a{i},0" for i in range(30)] + ["b0,1", "b1,1"]
    for r in rows[1:]:
        Image.new("RGB", (24, 24), (90, 40, 20)).save(src / f"{r.split(',')[0]}.png")
    (src / "labels.csv").write_text("\n".join(rows) + "\n")
    cmd = [sys.executable, "-m", "src.prepare", "grading", "--src", str(src), "--size", "16",
           "--out-dir", str(tmp_path / "proc"), "--splits-dir", str(tmp_path / "splits")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and "need >= 10 images per class" in result.stderr


def test_labels_option_ignores_dummy_label_files_like_aptos_sample_submission(tmp_path):
    src = tmp_path / "aptos"
    (src / "train_images").mkdir(parents=True)
    (src / "test_images").mkdir()
    for name, folder in (("a1", "train_images"), ("a2", "train_images"), ("t1", "test_images"), ("t2", "test_images")):
        Image.new("RGB", (24, 24), (90, 40, 20)).save(src / folder / f"{name}.png")
    (src / "train.csv").write_text("id_code,diagnosis\na1,3\na2,4\n")
    (src / "sample_submission.csv").write_text("id_code,diagnosis\nt1,0\nt2,0\n")   # dummy labels, must not be used
    cmd = [sys.executable, "-m", "src.prepare", "grading", "--src", str(src), "--size", "16", "--external",
           "--out-dir", str(tmp_path / "proc"), "--splits-dir", str(tmp_path / "splits"),
           "--labels", str(src / "train.csv")]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    import pandas as pd
    out = pd.read_csv(next((tmp_path / "splits").glob("ext_*_grading.csv")))
    assert sorted(out["grade"]) == [3, 4] and len(out) == 2
