"""Phase 1-2: find the raw files, crop + resize them, and freeze the split CSVs.

    python -m src.prepare inspect      --src data/raw/ddr_grading
    python -m src.prepare grading      --src data/raw/ddr_grading
    python -m src.prepare segmentation --src data/raw/ddr_segmentation

    # external test sets (everything goes to one CSV, nothing is split)
    python -m src.prepare grading      --src data/raw/aptos  --name aptos --external
    python -m src.prepare segmentation --src data/raw/idrid  --name idrid --external

The Kaggle folder layouts are not guaranteed, so file discovery is deliberately
tolerant (see discover_grading / discover_segmentation). If your copy looks
different, run `inspect` first and adjust those two functions - nothing else
in the project depends on the raw layout.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import LESIONS

SEED = 42  # fixed: splits are made once and frozen
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEG_IMG_EXT = {".jpg", ".jpeg", ".png"}  # tif files in a seg tree are masks
SPLIT_NAMES = {"train": "train", "training": "train", "valid": "val", "val": "val",
               "validation": "val", "test": "test", "testing": "test"}
LESION_WORDS = {"EX": ("hard exudate",), "HE": ("haemorrhage", "hemorrhage"),
                "MA": ("microaneurysm",), "SE": ("soft exudate",)}
MASK_KEEP = 64  # masks are shrunk bilinearly, then kept where >= 25% (keeps 1-px MA dots)


# --------------------------------------------------------------------------- #
# Discovery: raw folder -> DataFrame
# --------------------------------------------------------------------------- #
def split_from_path(path, root):
    """'train' / 'val' / 'test' if a parent folder is named like one, else None."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return None


def lesion_of(path):
    """(lesion, image_stem) if `path` is a lesion mask, else None.

    Recognises  .../MA/img001.tif  (folder named after the lesion, DDR style)
    and         img001_MA.tif      (file suffix, IDRiD style).
    """
    folder = path.parent.name.lower()
    for code, words in LESION_WORDS.items():
        if folder == code.lower() or any(w in folder for w in words):
            head, _, tail = path.stem.rpartition("_")
            # folder AND suffix (IDRiD: Microaneurysms/IDRiD_01_MA.tif): the image name is without the suffix
            return code, head if head and tail.upper() == code else path.stem
    head, _, tail = path.stem.rpartition("_")
    if head and tail.upper() in LESIONS:
        return tail.upper(), head
    return None


def read_grade_labels(src, label_files=None):
    """Labels from train/valid/test .txt ('name grade' lines) or from CSV files.

    `label_files` (CSV paths) restricts the search to exactly those files. Use it for external sets:
    e.g. APTOS also ships sample_submission.csv with a dummy all-zero `diagnosis` column.
    """
    rows = []
    for f in ([] if label_files else sorted(src.rglob("*.txt"))):
        split = SPLIT_NAMES.get(f.stem.lower())
        if split is None:
            continue
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                rows.append((Path(parts[0]).stem, int(parts[-1]), split))
    if rows:
        return pd.DataFrame(rows, columns=["stem", "grade", "split"])

    frames = []
    for f in (label_files or sorted(src.rglob("*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {str(c).lower().strip(): c for c in df.columns}
        name = next((c for k, c in cols.items()
                     if k in ("id", "id_code") or any(w in k for w in ("image", "file", "name"))), None)
        label = next((c for k, c in cols.items()
                      if any(w in k for w in ("diagnosis", "grade", "label", "level"))), None)
        if name is None or label is None:
            continue
        split_col = next((c for k, c in cols.items() if k in ("split", "set", "subset")), None)
        out = pd.DataFrame({
            "stem": df[name].astype(str).map(lambda s: Path(s.strip()).stem),
            "grade": pd.to_numeric(df[label], errors="coerce"),
            "split": df[split_col].astype(str).str.lower().map(SPLIT_NAMES) if split_col else None,
        }).dropna(subset=["grade"])
        out["grade"] = out["grade"].astype(int)
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def discover_grading(src, label_files=None):
    """-> DataFrame[stem, path, grade, split]; split is None when not official."""
    images = {p.stem: p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXT}
    labels = read_grade_labels(src, label_files)
    if labels is None and label_files:
        sys.exit(f"No image-name + grade columns found in {[str(f) for f in label_files]}.")
    if labels is None:  # last resort: images sitting in folders named 0,1,2,3,4,5
        rows = [(p.stem, int(p.parent.name)) for p in images.values() if p.parent.name.isdigit()]
        labels = pd.DataFrame(rows, columns=["stem", "grade"]).assign(split=None)
    if labels.empty:
        sys.exit(f"No labels found under {src}. Run `inspect` and adapt read_grade_labels().")
    labels = labels.drop_duplicates("stem")
    labels["path"] = labels["stem"].map(images)
    missing = labels["path"].isna().sum()
    if missing:
        print(f"  warning: {missing} labelled names have no image file; skipped")
    labels = labels.dropna(subset=["path"]).copy()
    labels["split"] = [s if isinstance(s, str) else split_from_path(p, src)
                       for s, p in zip(labels["split"], labels["path"])]
    return labels[["stem", "path", "grade", "split"]].reset_index(drop=True)


def discover_segmentation(src):
    """-> DataFrame[stem, path, split, EX, HE, MA, SE]; lesion cells are paths or None."""
    images, masks = {}, {}
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        hit = lesion_of(p)
        if hit and p.suffix.lower() in IMG_EXT:
            masks[(hit[1], hit[0])] = p
        elif p.suffix.lower() in SEG_IMG_EXT and not hit:
            images[p.stem] = p
    rows = []
    for stem, path in images.items():
        found = {code: masks.get((stem, code)) for code in LESIONS}
        if any(found.values()):  # images with no mask at all are not segmentation data
            rows.append({"stem": stem, "path": path, "split": split_from_path(path, src), **found})
    if not rows:
        sys.exit(f"No image+mask pairs found under {src}. Run `inspect` and adapt discover_segmentation().")
    print(f"  {len(rows)} images with masks ({len(images) - len(rows)} images without any mask skipped)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Preprocessing: crop black border -> pad to square -> resize
# --------------------------------------------------------------------------- #
def retina_box(im, thr=10):
    """Bounding box of the non-black area (idempotent on already-cropped images)."""
    gray = np.asarray(im.convert("L")) > thr
    cols = np.flatnonzero(gray.sum(0) > 2)
    rows = np.flatnonzero(gray.sum(1) > 2)
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, im.width, im.height)
    return (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)


def crop_pad_resize(im, box, size, resample):
    im = im.crop(box)
    side = max(im.size)
    canvas = Image.new(im.mode, (side, side), 0)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas.resize((size, size), resample)


def process_image(src, dst, size):
    """Returns the crop box so masks can be cropped identically."""
    im = Image.open(src).convert("RGB")
    box = retina_box(im)
    if not dst.exists():
        crop_pad_resize(im, box, size, Image.BICUBIC).save(dst, quality=95)
    return im.size, box


def process_mask(src, dst, image_size, box, size):
    if dst.exists():
        return
    m = Image.open(src).convert("L")
    m = Image.fromarray(((np.asarray(m) > 0) * 255).astype(np.uint8))
    if m.size != image_size:
        m = m.resize(image_size, Image.NEAREST)
    m = crop_pad_resize(m, box, size, Image.BILINEAR)
    Image.fromarray(((np.asarray(m) >= MASK_KEEP) * 255).astype(np.uint8)).save(dst)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def assign_splits(df, strat=None, external=False):
    if external:
        return df.assign(split="test")
    if df["split"].notna().all():
        print("  using the official train/val/test split found in the raw data")
        return df
    print("  no complete official split -> one stratified 70/15/15 split, seed 42")
    df = df.reset_index(drop=True)
    y = None if strat is None else np.asarray(strat)  # positional, aligned with the reset index
    if y is not None and Counter(y).most_common()[-1][1] < 10:
        sys.exit(f"Class counts {dict(sorted(Counter(y).items()))}: need >= 10 images per class for a stratified "
                 "70/15/15 split. Check the labels, or make the split by hand and write the CSVs yourself.")
    train_idx, rest_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, random_state=SEED,
                                         stratify=None if y is None else y[rest_idx])
    df["split"] = ""
    df.loc[train_idx, "split"], df.loc[val_idx, "split"], df.loc[test_idx, "split"] = "train", "val", "test"
    return df


def csv_paths(args, prefix):
    d = Path(args.splits_dir)
    if args.external:
        return {"test": d / f"ext_{args.name}_{prefix}.csv"}
    return {s: d / f"{prefix}_{s}.csv" for s in ("train", "val", "test")}


def check_frozen(paths, force):
    existing = [p for p in paths.values() if p.exists()]
    if existing and not force:
        sys.exit("Splits are frozen - these files already exist:\n  " + "\n  ".join(map(str, existing))
                 + "\nDelete them deliberately (or pass --force) only if you really want new splits.")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def run_grading(args):
    src = Path(args.src)
    out_csv = csv_paths(args, "grading")
    check_frozen(out_csv, args.force)
    df = discover_grading(src, [Path(f) for f in args.labels] if args.labels else None)
    print(f"  grade counts before filtering: {dict(sorted(Counter(df['grade']).items()))}")
    dropped = (~df["grade"].between(0, 4)).sum()
    df = df[df["grade"].between(0, 4)]  # grade 5 = ungradable -> dropped
    print(f"  dropped {dropped} images outside grades 0-4 (class 5 = ungradable)")

    out_dir = Path(args.out_dir) / f"{args.name}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep, bad = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="preprocess"):
        try:
            process_image(row.path, out_dir / f"{row.stem}.jpg", args.size)
            keep.append(row.Index)
        except Exception as e:  # unreadable / truncated file
            bad.append((row.path, e))
    for path, e in bad:
        print(f"  unreadable, skipped: {path} ({e})")
    df = assign_splits(df.loc[keep].assign(image_path=lambda d: [(out_dir / f"{s}.jpg").as_posix() for s in d["stem"]]),
                       strat=df.loc[keep, "grade"], external=args.external)

    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)
    for split, path in out_csv.items():
        part = df[df["split"] == split][["image_path", "grade"]]
        part.to_csv(path, index=False)
        print(f"  {path}: {len(part)} images, grades {dict(sorted(Counter(part['grade']).items()))}")


def run_segmentation(args):
    src = Path(args.src)
    out_csv = csv_paths(args, "seg")
    check_frozen(out_csv, args.force)
    assert args.size % 32 == 0, "--size must be a multiple of 32 (U-Net requirement)"
    df = discover_segmentation(src)
    out_dir = Path(args.out_dir) / f"{args.name}_seg_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in tqdm(df.to_dict("records"), desc="preprocess"):
        try:
            image_size, box = process_image(r["path"], out_dir / f"{r['stem']}.jpg", args.size)
            row = {"stem": r["stem"], "split": r["split"], "image_path": (out_dir / f"{r['stem']}.jpg").as_posix()}
            for code in LESIONS:  # empty cell = lesion absent in this image (all-zero mask)
                mask_path = r[code]
                row[f"{code.lower()}_mask"] = ""
                if isinstance(mask_path, Path):
                    dst = out_dir / f"{r['stem']}_{code}.png"
                    process_mask(mask_path, dst, image_size, box, args.size)
                    row[f"{code.lower()}_mask"] = dst.as_posix()
            rows.append(row)
        except Exception as e:
            print(f"  unreadable, skipped: {r['path']} ({e})")
    df = assign_splits(pd.DataFrame(rows), external=args.external)

    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)
    cols = ["image_path"] + [f"{c.lower()}_mask" for c in LESIONS]
    for split, path in out_csv.items():
        part = df[df["split"] == split]
        part[cols].to_csv(path, index=False)
        print(f"  {path}: {len(part)} images")

    # The seg images also exist in the grading set: report (not an error while tracks are separate).
    grading = [Path(args.splits_dir) / f"grading_{s}.csv" for s in ("train", "val")]
    if not args.external and all(p.exists() for p in grading):
        seen = {Path(p).stem for g in grading for p in pd.read_csv(g)["image_path"]}
        overlap = sorted(set(df[df["split"] == "test"]["stem"]) & seen)
        print(f"  overlap: {len(overlap)} seg-test images are in grading train/val "
              "(only matters if you later combine the tracks)")


def inspect(args):
    src = Path(args.src)
    print(f"Layout of {src} (file counts per folder):")
    for d in sorted([src] + [p for p in src.rglob("*") if p.is_dir()]):
        depth = len(d.relative_to(src).parts)
        if depth <= 4:
            exts = Counter(f.suffix.lower() for f in d.iterdir() if f.is_file())
            print(f"{'  ' * depth}{d.name}/  {dict(exts) if exts else ''}")
    for f in (sorted(src.rglob("*.csv")) + sorted(src.rglob("*.txt")))[:8]:
        print(f"\n--- {f.relative_to(src)} (first 3 lines)")
        print("\n".join(f.read_text(errors="replace").splitlines()[:3]))
    print("\nWhat the grading reader finds:")
    try:
        df = discover_grading(src)
        print(df.assign(split=df["split"].fillna("none")).groupby(["split", "grade"]).size().unstack(fill_value=0))
    except SystemExit as e:
        print(f"  (nothing: {e})")
    print("\nWhat the segmentation reader finds:")
    try:
        df = discover_segmentation(src)
        print(df.assign(split=df["split"].fillna("none")).groupby("split")[LESIONS].count())
    except SystemExit as e:
        print(f"  (nothing: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inspect", "grading", "segmentation"])
    ap.add_argument("--src", required=True, help="folder with the extracted raw files")
    ap.add_argument("--name", default=None, help="dataset name (default: grading / ddr)")
    ap.add_argument("--size", type=int, default=None, help="output size (default 512 grading, 1024 segmentation)")
    ap.add_argument("--external", action="store_true", help="external test set: one CSV, no split")
    ap.add_argument("--labels", nargs="+", help="grading only: CSV file(s) holding the labels (recommended for external sets)")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--force", action="store_true", help="overwrite frozen split CSVs")
    args = ap.parse_args()
    if args.command == "grading":
        args.name, args.size = args.name or "ddr_grading", args.size or 512
        run_grading(args)
    elif args.command == "segmentation":
        args.name, args.size = args.name or "ddr", args.size or 1024
        run_segmentation(args)
    else:
        inspect(args)


if __name__ == "__main__":
    main()
