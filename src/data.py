"""Datasets and augmentations for both tracks.

Validation / test data only ever gets resize + normalise (no augmentation).
"""
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.imbalance import pick_crop
from src.utils import LESIONS

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet stats (pretrained encoders)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Track A: grading
# --------------------------------------------------------------------------- #
def grading_transform(size, train=False, strong=False):
    finish = [T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())]
    if not train:
        return T.Compose([T.Resize((size, size))] + finish)
    if strong:  # used for minority classes in G5
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1.2)),
               T.ColorJitter(0.4, 0.4, 0.3, 0.05),
               T.RandomApply([T.GaussianBlur(5)], p=0.3)]
    else:
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
               T.ColorJitter(0.2, 0.2)]
    return T.Compose([T.Resize((size, size))] + aug + finish)


class GradingDataset(Dataset):
    """Rows of `image_path, grade`. Images in `strong_classes` get the stronger augmentation."""

    def __init__(self, csv, size, train=False, strong_classes=()):
        df = pd.read_csv(csv)
        self.paths = df["image_path"].tolist()
        self.labels = df["grade"].to_numpy(dtype=np.int64)
        self.strong_classes = set(strong_classes)
        self.normal_tf = grading_transform(size, train)
        self.strong_tf = grading_transform(size, train, strong=True) if train else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        label = int(self.labels[i])
        tf = self.strong_tf if label in self.strong_classes else self.normal_tf
        return tf(image), label


# --------------------------------------------------------------------------- #
# Track B: segmentation
# --------------------------------------------------------------------------- #
def augment(img, masks):
    """Flips + 90-degree turns (exact for masks) and a brightness/contrast change on the image only."""
    if random.random() < 0.5:
        img, masks = img[:, ::-1], masks[:, :, ::-1]
    if random.random() < 0.5:
        img, masks = img[::-1], masks[:, ::-1]
    k = random.randrange(4)
    img, masks = np.rot90(img, k, (0, 1)), np.rot90(masks, k, (1, 2))
    x = img.astype(np.float32)
    mean = x.mean()
    x = (x - mean) * random.uniform(0.85, 1.15) + mean * random.uniform(0.85, 1.15)
    return np.clip(x, 0, 255), np.ascontiguousarray(masks)


class SegmentationDataset(Dataset):
    """Rows of `image_path, ex_mask, he_mask, ma_mask, se_mask` (empty cell = lesion absent).

    Training returns a random `crop_size` crop (lesion-centred with probability `lesion_prob`);
    otherwise the full image is returned.
    """

    def __init__(self, csv, train=False, crop_size=512, lesion_prob=0.0):
        self.df = pd.read_csv(csv, keep_default_na=False)
        self.train, self.crop_size, self.lesion_prob = train, crop_size, lesion_prob

    def __len__(self):
        return len(self.df)

    def load_masks(self, row):
        w, h = Image.open(row["image_path"]).size
        masks = np.zeros((len(LESIONS), h, w), dtype=np.uint8)
        for c, lesion in enumerate(LESIONS):
            path = row[f"{lesion.lower()}_mask"]
            if path:
                masks[c] = np.asarray(Image.open(path)) > 0
        return masks

    def lesion_pixel_ratio(self):
        """Share of pixels that are lesion, per lesion, over the whole (training) set."""
        pos, total = np.zeros(len(LESIONS)), 0
        for _, row in self.df.iterrows():
            m = self.load_masks(row)
            pos += m.sum((1, 2))
            total += m.shape[1] * m.shape[2]
        return pos / total

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        masks = self.load_masks(row)
        if self.train:
            crop = min(self.crop_size, *img.shape[:2])
            y0, x0 = pick_crop(masks, crop, self.lesion_prob)
            img, masks = img[y0:y0 + crop, x0:x0 + crop], masks[:, y0:y0 + crop, x0:x0 + crop]
            img, masks = augment(img, masks)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(masks.astype(np.float32))
