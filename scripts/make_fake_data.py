"""Generate tiny synthetic 'fundus' datasets so the whole pipeline can be tested without Kaggle.

Four fake raw datasets are written, each with a different folder layout so that every
discovery rule in src/prepare.py is exercised:

    ddr_grading/  train|valid|test folders + train.txt/valid.txt/test.txt   (original DDR style)
    aptos/        one folder + CSV with id_code,diagnosis                    (APTOS style)
    ddr_seg/      <split>/image/*.jpg and <split>/label/<EX|HE|MA|SE>/*.tif  (original DDR style)
    idrid_seg/    images/IDRiD_xx.jpg and masks/IDRiD_xx_<MA|HE|EX|SE>.tif   (IDRiD style)

    python scripts/make_fake_data.py --out data/smoke/raw
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {"EX": (250, 230, 60), "HE": (90, 5, 5), "MA": (140, 10, 10), "SE": (235, 235, 235)}


def fundus(rng, n_dots=0):
    """Dark border + orange disc with a few bright dots (more dots = 'worse' grade)."""
    w, h = rng.randint(150, 190), rng.randint(130, 160)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    r = min(w, h) // 2 - 4
    cx, cy = w // 2, h // 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(170, 80, 30))
    for _ in range(n_dots):
        x, y = cx + rng.randint(-r // 2, r // 2), cy + rng.randint(-r // 2, r // 2)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(240, 220, 120))
    return img


def make_grading(root, counts, rng, official):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, per_class in counts.items():
        folder = root / split if official else root / "images"
        folder.mkdir(parents=True, exist_ok=True)
        for grade, n in enumerate(per_class):
            for i in range(n):
                name = f"{split[:2]}-{grade}-{i:03d}"
                fundus(rng, n_dots=grade * 6).save(folder / f"{name}.jpg")
                rows.append((split, name, grade))
    if official:
        for split in counts:
            (root / f"{split}.txt").write_text("".join(f"{n}.jpg {g}\n" for s, n, g in rows if s == split))
    else:
        (root / "labels.csv").write_text("id_code,diagnosis\n" + "".join(f"{n},{g}\n" for _, n, g in rows))


def make_seg(root, split_sizes, rng, style):
    for split, n_images in split_sizes.items():
        for i in range(n_images):
            name = f"{split[:2]}_{i:03d}" if style == "ddr" else f"IDRiD_{split[:2]}{i:02d}"
            img = fundus(rng)
            w, h = img.size
            masks = {}
            for lesion, size in (("EX", 6), ("HE", 5), ("MA", 3), ("SE", 7)):
                if rng.random() < 0.25 and not (lesion == "MA" and i < 3):
                    continue  # not every lesion is in every image (and MA is always present early on)
                mask = Image.new("L", (w, h), 0)
                d = ImageDraw.Draw(mask)
                for _ in range(rng.randint(2, 6)):
                    x, y = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
                    d.ellipse([x - size // 2, y - size // 2, x + size // 2, y + size // 2], fill=255)
                img.paste(COLORS[lesion], mask=mask)  # lesions are visible in the image, so they are learnable
                masks[lesion] = mask
            if style == "ddr":
                (root / split / "image").mkdir(parents=True, exist_ok=True)
                img.save(root / split / "image" / f"{name}.jpg")
                for lesion, mask in masks.items():
                    (root / split / "label" / lesion).mkdir(parents=True, exist_ok=True)
                    mask.save(root / split / "label" / lesion / f"{name}.tif")
            else:
                (root / "images").mkdir(parents=True, exist_ok=True)
                (root / "masks").mkdir(parents=True, exist_ok=True)
                img.save(root / "images" / f"{name}.jpg")
                for lesion, mask in masks.items():
                    mask.save(root / "masks" / f"{name}_{lesion}.tif")
            if style == "idrid":  # optic-disc masks must be ignored by the reader
                Image.new("L", (w, h), 0).save(root / "masks" / f"{name}_OD.tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/smoke/raw")
    args = ap.parse_args()
    out, rng = Path(args.out), random.Random(0)

    # imbalanced on purpose: No DR dominates, Severe is rare
    make_grading(out / "ddr_grading", {"train": [30, 6, 20, 4, 8], "valid": [8, 3, 5, 3, 3], "test": [8, 3, 5, 3, 3]},
                 rng, official=True)
    make_grading(out / "aptos", {"all": [10, 4, 8, 3, 3]}, rng, official=False)
    make_seg(out / "ddr_seg", {"train": 14, "valid": 5, "test": 5}, rng, style="ddr")
    make_seg(out / "idrid_seg", {"all": 6}, rng, style="idrid")
    print(f"fake raw data written to {out}")


if __name__ == "__main__":
    main()
