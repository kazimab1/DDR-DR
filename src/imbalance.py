"""The imbalance strategies: samplers, losses and lesion-aware cropping.

Every strategy is chosen by name from the `imbalance:` block of a config, so
adding a new one means adding one branch here and one YAML file.
All statistics (class counts, pixel ratios) come from the TRAINING split only.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler

from src.utils import NUM_GRADES


# --------------------------------------------------------------------------- #
# Track A samplers (class imbalance)
# --------------------------------------------------------------------------- #
class UnderSampler(Sampler):
    """Each epoch, draw at most `cap` images per class (majority classes are cut down)."""

    def __init__(self, labels, cap, seed):
        self.by_class = [np.flatnonzero(labels == c) for c in np.unique(labels)]
        self.cap = cap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return sum(min(len(idx), self.cap) for idx in self.by_class)

    def __iter__(self):
        picked = [self.rng.choice(idx, min(len(idx), self.cap), replace=False) for idx in self.by_class]
        order = np.concatenate(picked)
        self.rng.shuffle(order)
        return iter(order.tolist())


def get_sampler(kind, labels, seed):
    if kind == "none":
        return None
    counts = np.bincount(labels, minlength=NUM_GRADES)
    if kind == "oversample":  # every class is drawn equally often (with replacement)
        weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        gen = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=gen)
    if kind == "undersample":  # cap at the median class size
        return UnderSampler(labels, cap=int(np.median(counts)), seed=seed)
    raise ValueError(f"unknown sampler '{kind}' (none | oversample | undersample)")


def minority_classes(labels):
    """Classes smaller than the median class size (they get stronger augmentation in G5)."""
    counts = np.bincount(labels, minlength=NUM_GRADES)
    return [int(c) for c in np.flatnonzero(counts < np.median(counts))]


# --------------------------------------------------------------------------- #
# Track B sampler (pixel imbalance): crop position
# --------------------------------------------------------------------------- #
def pick_crop(masks, crop, lesion_prob):
    """Top-left corner of a square crop from masks of shape (4, H, W).

    With probability `lesion_prob` the crop is centred (with jitter) on a random pixel of
    a randomly chosen lesion type that is present, so rare lesions (MA) are seen as often
    as common ones. Otherwise the crop position is uniform.
    """
    _, h, w = masks.shape
    if random.random() < lesion_prob:
        present = [c for c in range(masks.shape[0]) if masks[c].any()]
        if present:
            ys, xs = np.nonzero(masks[random.choice(present)])
            k = random.randrange(len(ys))
            jitter = crop // 4
            y0 = ys[k] - crop // 2 + random.randint(-jitter, jitter)
            x0 = xs[k] - crop // 2 + random.randint(-jitter, jitter)
            return int(np.clip(y0, 0, h - crop)), int(np.clip(x0, 0, w - crop))
    return random.randint(0, h - crop), random.randint(0, w - crop)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Multi-class focal loss: (1 - p_t)^gamma * CE. Easy examples contribute less."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        return (-((1 - logp.exp()) ** self.gamma) * logp).mean()


class TverskyLoss(nn.Module):
    """1 - TP / (TP + alpha*FP + beta*FN), per lesion channel. alpha = beta = 0.5 is Dice."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        tp = (p * target).sum((0, 2, 3))  # sums over the whole batch, one value per lesion
        fp = (p * (1 - target)).sum((0, 2, 3))
        fn = ((1 - p) * target).sum((0, 2, 3))
        return (1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)).mean()


class BCEDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce, self.dice = nn.BCEWithLogitsLoss(), TverskyLoss(0.5, 0.5)

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)


def get_loss(cfg, train_ds, log=print):
    kind = cfg["imbalance"]["loss"]
    if cfg["task"] == "grading":
        counts = np.bincount(train_ds.labels, minlength=NUM_GRADES)
        if kind == "ce":
            return nn.CrossEntropyLoss()
        if kind == "weighted_ce":  # inverse class frequency, normalised to mean 1
            weight = torch.as_tensor(len(train_ds.labels) / (NUM_GRADES * counts), dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight)
        if kind == "focal":
            return FocalLoss(cfg["imbalance"].get("focal_gamma", 2.0))
        raise ValueError(f"unknown grading loss '{kind}' (ce | weighted_ce | focal)")

    imb = cfg["imbalance"]
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted_bce":  # pos_weight from each lesion's pixel ratio in the training masks
        ratio = train_ds.lesion_pixel_ratio()
        raw = (1 - ratio) / np.maximum(ratio, 1e-9)  # background pixels per lesion pixel
        # The raw ratio is often 100-5000 and destabilises training, so it is softened
        # (power 0.5 = square root; 1.0 = literal ratio) and capped. Order between lesions is kept.
        pos_weight = torch.as_tensor(raw ** imb.get("pos_weight_power", 0.5), dtype=torch.float32)
        pos_weight = pos_weight.clamp(max=imb.get("pos_weight_max", 100.0))
        log(f"lesion pixel share (EX,HE,MA,SE): {(ratio * 100).round(3)} %  raw ratio: {raw.round(0)}  "
            f"pos_weight used: {pos_weight.numpy().round(1)}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(-1, 1, 1))
    if kind == "dice":
        return TverskyLoss(0.5, 0.5)
    if kind == "bce_dice":
        return BCEDice()
    if kind == "tversky":  # beta > alpha: a missed lesion pixel costs more than a false alarm
        return TverskyLoss(imb.get("tversky_alpha", 0.3), imb.get("tversky_beta", 0.7))
    raise ValueError(f"unknown segmentation loss '{kind}' (bce | weighted_bce | dice | bce_dice | tversky)")
