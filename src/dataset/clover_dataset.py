"""Clover segmentation dataset.

Images are clover-field photographs (800x600 RGB JPEGs).
Masks are JPEG-encoded binary masks where white = four-leaf clover; because of
JPEG compression they contain artifacts near 0/255, so they are binarized with
a configurable threshold on load.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_mask_binary(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a mask and binarize to 0/255 uint8 (0=bg, 255=fg)."""
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"failed to read mask: {path}")
    return (arr > threshold).astype(np.uint8) * 255


def resize_mask_binary(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation.

    Standard interpolations (bilinear etc.) introduce intermediate labels, so
    nearest is required to keep the mask binary.
    """
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def find_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = []
    for img in imgs:
        m = masks.get(img.stem)
        if m is None:
            raise FileNotFoundError(f"no mask found for image {img}")
        pairs.append((img, m))
    return pairs


class CloverDataset(Dataset):
    """Image + binary-mask pairs with optional albumentations transform."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, transform=None, mask_threshold: int = 127):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.pairs = find_pairs(self.image_dir, self.mask_dir)
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = load_mask_binary(mask_path, self.mask_threshold)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        # image -> float32 tensor [0,1] in [C,H,W] (albumentations ToTensorV2 path)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        else:
            image = image.float() / 255.0
        # mask: ToTensorV2 may already return a uint8 tensor; normalize to [0,1]
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
        else:
            mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # (1,H,W) to match model logits
        return image, mask


class PairIndexDataset(Dataset):
    """Wraps another dataset but also returns (img_path, mask_path) for visualization."""

    def __init__(self, base: CloverDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, mask = self.base[idx]
        return image, mask, self.base.pairs[idx]