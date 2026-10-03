"""Phase 1: dataset statistics and thesis figures (replaces the exploration notebooks).

    python -m src.explore grading      --src data/raw/ddr_grading
    python -m src.explore segmentation --src data/raw/ddr_segmentation

Prints the numbers to record in the README and writes PNG/CSV files to outputs/figures/.
"""
import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.prepare import discover_grading, discover_segmentation  # noqa: E402
from src.utils import LESIONS  # noqa: E402

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR", 5: "Ungradable"}
COLORS = {"EX": (255, 215, 0), "HE": (255, 0, 0), "MA": (0, 200, 255), "SE": (0, 255, 0)}


def file_checks(paths):
    """Image sizes, unreadable files and exact duplicates (same bytes)."""
    sizes, bad, seen, dupes = Counter(), [], {}, []
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                im.verify()
        except Exception:
            bad.append(p)
            continue
        digest = hashlib.md5(Path(p).read_bytes()).hexdigest()
        if digest in seen:
            dupes.append((p, seen[digest]))
        seen[digest] = p
    print(f"image sizes (w, h): {sizes.most_common(6)}{' ...' if len(sizes) > 6 else ''}")
    print(f"unreadable files: {len(bad)}   exact duplicates: {len(dupes)}")
    for p in bad[:5]:
        print("  unreadable:", p)
    for a, b in dupes[:5]:
        print("  duplicate:", a, "==", b)


def grading(src, out):
    df = discover_grading(src)
    df["split"] = df["split"].fillna("unsplit")
    table = df.groupby(["split", "grade"]).size().unstack(fill_value=0)
    table.loc["all"] = table.sum()
    table.columns = [f"{c} {GRADE_NAMES.get(c, '')}".strip() for c in table.columns]
    print("\nimages per split and grade:\n", table.to_string())
    table.to_csv(out / "grading_class_counts.csv")
    file_checks(df["path"])

    counts = df.groupby(["grade", "split"]).size().unstack(fill_value=0)
    ax = counts.plot.bar(stacked=True, figsize=(7, 4), rot=0)
    ax.set_xticklabels([GRADE_NAMES.get(g, g) for g in counts.index])
    ax.set_ylabel("images")
    ax.set_title("DDR grading: class distribution")
    plt.tight_layout()
    plt.savefig(out / "grading_class_counts.png", dpi=150)
    plt.close()

    rng = random.Random(0)
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))
    for g, row in enumerate(axes):
        pool = df[df["grade"] == g]["path"].tolist()
        for ax, p in zip(row, rng.sample(pool, min(4, len(pool)))):
            ax.imshow(Image.open(p).convert("RGB"))
        for i, ax in enumerate(row):
            ax.axis("off")
        row[0].set_title(GRADE_NAMES[g], loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "grading_samples.png", dpi=120)
    plt.close()


def segmentation(src, out):
    df = discover_segmentation(src)
    df["split"] = df["split"].fillna("unsplit")
    print("\nimages per split:", df["split"].value_counts().to_dict())
    file_checks(df["path"])

    n_images, share = {}, {}
    pos, total = Counter(), 0
    for r in df.to_dict("records"):
        w, h = Image.open(r["path"]).size
        total += w * h
        for lesion in LESIONS:
            if r[lesion] is not None:
                m = np.asarray(Image.open(r[lesion]).convert("L")) > 0
                pos[lesion] += int(m.sum())
                n_images[lesion] = n_images.get(lesion, 0) + int(m.any())
    stats = pd.DataFrame({"images_with_lesion": [n_images.get(l, 0) for l in LESIONS],
                          "pixel_share_%": [100 * pos[l] / total for l in LESIONS]}, index=LESIONS)
    print("\nlesion statistics (this is the pixel imbalance):\n", stats.round(4).to_string())
    stats.to_csv(out / "seg_lesion_stats.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    stats["images_with_lesion"].plot.bar(ax=axes[0], rot=0, title="images containing the lesion")
    stats["pixel_share_%"].plot.bar(ax=axes[1], rot=0, logy=True, title="share of all pixels (%)")
    plt.tight_layout()
    plt.savefig(out / "seg_lesion_stats.png", dpi=150)
    plt.close()

    rng = random.Random(0)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, r in zip(axes, rng.sample(df.to_dict("records"), min(4, len(df)))):
        img = np.asarray(Image.open(r["path"]).convert("RGB")).copy()
        for lesion in LESIONS:
            if r[lesion] is not None:
                m = np.asarray(Image.open(r[lesion]).convert("L").resize(img.shape[1::-1])) > 0
                img[m] = COLORS[lesion]
        ax.imshow(img)
        ax.axis("off")
    axes[0].set_title("EX yellow  HE red  MA cyan  SE green", loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "seg_samples.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", choices=["grading", "segmentation"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (grading if args.track == "grading" else segmentation)(Path(args.src), out)
    print(f"\nfigures and tables written to {out}/")


if __name__ == "__main__":
    main()
