"""Metrics for both tracks, plus a CLI to score a saved run on val / test / an external set.

    python -m src.evaluate --run outputs/runs/g2_oversample --split val
    python -m src.evaluate --run outputs/runs/g2_oversample --split test        # once, winners only
    python -m src.evaluate --run outputs/runs/g2_oversample --csv data/splits/ext_aptos_grading.csv --tag aptos
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from src.data import GradingDataset, SegmentationDataset
from src.models import build_model
from src.utils import LESIONS, MAIN_METRIC, NUM_GRADES, get_device

AUPR_BINS = 10000  # AUPR is computed from a probability histogram: exact up to bin width, O(1) memory


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


# --------------------------------------------------------------------------- #
# Track A
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_grading(model, loader, device, amp=False):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with autocast(device, amp):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    probs, y = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    pred = probs.argmax(1)
    classes = list(range(NUM_GRADES))
    m = {
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }
    referable = y >= 2  # moderate NPDR or worse
    m["auc_referable"] = roc_auc_score(referable, probs[:, 2:].sum(1)) if 0 < referable.sum() < len(y) else float("nan")
    for c, r in zip(classes, recall_score(y, pred, labels=classes, average=None, zero_division=0)):
        m[f"recall_{c}"] = r
    m = {k: float(v) for k, v in m.items()}
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=classes).tolist()
    return m


# --------------------------------------------------------------------------- #
# Track B
# --------------------------------------------------------------------------- #
def binned_ap(pos, neg):
    """Average precision from per-bin counts of positive / negative pixels (bins ascend in probability)."""
    tp, fp = pos.flip(0).double().cumsum(0), neg.flip(0).double().cumsum(0)  # threshold descending
    if tp[-1] == 0:
        return float("nan")  # lesion absent from the whole split
    precision, recall = tp / (tp + fp).clamp(min=1), tp / tp[-1]
    prev = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float(((recall - prev) * precision).sum())


@torch.no_grad()
def eval_segmentation(model, loader, device, threshold=0.5, amp=False):
    model.eval()
    n = len(LESIONS)
    pos = torch.zeros(n, AUPR_BINS, dtype=torch.long, device=device)
    neg = torch.zeros_like(pos)
    tp, fp, fn = (torch.zeros(n, dtype=torch.double, device=device) for _ in range(3))
    for x, y in loader:
        x, y = x.to(device), y.to(device) > 0.5
        with autocast(device, amp):
            logits = model(x)
        p = torch.sigmoid(logits.float())
        for c in range(n):
            idx = (p[:, c] * (AUPR_BINS - 1)).round().long().flatten()
            target = y[:, c].flatten()
            pos[c] += torch.bincount(idx[target], minlength=AUPR_BINS)
            neg[c] += torch.bincount(idx[~target], minlength=AUPR_BINS)
        pred = p > threshold
        tp += (pred & y).sum((0, 2, 3))
        fp += (pred & ~y).sum((0, 2, 3))
        fn += (~pred & y).sum((0, 2, 3))
    tp, fp, fn = tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):  # lesion absent -> nan, ignored in the mean
        dice, iou = 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)
    m = {}
    for c, lesion in enumerate(LESIONS):
        m[f"aupr_{lesion}"] = binned_ap(pos[c], neg[c])
        m[f"dice_{lesion}"], m[f"iou_{lesion}"] = float(dice[c]), float(iou[c])
    for name in ("aupr", "dice", "iou"):
        m[f"mean_{name}"] = float(np.nanmean([m[f"{name}_{l}"] for l in LESIONS]))
    return m


def run_eval(model, loader, cfg, device):
    amp = cfg["train"].get("amp", False)
    if cfg["task"] == "grading":
        return eval_grading(model, loader, device, amp)
    return eval_segmentation(model, loader, device, cfg["eval"]["threshold"], amp)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def make_eval_loader(cfg, csv):
    if cfg["task"] == "grading":
        ds = GradingDataset(csv, cfg["data"]["image_size"])
    else:
        ds = SegmentationDataset(csv)
    bs = cfg["eval"]["batch_size"]
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg["train"]["num_workers"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder, e.g. outputs/runs/g2_oversample")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--csv", help="evaluate on this split CSV instead (external dataset)")
    ap.add_argument("--tag", help="name for the output file with --csv (default: CSV file name)")
    ap.add_argument("--force", action="store_true", help="re-run a test evaluation that already exists")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    prefix = "grading" if cfg["task"] == "grading" else "seg"
    if args.csv:
        csv, tag = args.csv, args.tag or Path(args.csv).stem
    else:
        csv, tag = Path(cfg["data"]["val_csv"]).parent / f"{prefix}_{args.split}.csv", args.split
    out = run / f"{tag}_metrics.json"
    if tag != "val" and out.exists() and not args.force:
        raise SystemExit(f"{out} exists. The test / external sets are evaluated once; use --force to override.")

    device = get_device()
    model = build_model(cfg, pretrained=False).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = run_eval(model, make_eval_loader(cfg, csv), cfg, device)
    metrics = {"main_metric": MAIN_METRIC[cfg["task"]], "best_epoch": ckpt["epoch"], "csv": str(csv), **metrics}
    out.write_text(json.dumps(metrics, indent=2))
    main_name = metrics["main_metric"]
    print(f"{run.name} on {tag}: {main_name} = {metrics[main_name]:.4f}  -> {out}")


if __name__ == "__main__":
    main()
