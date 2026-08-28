"""DataLoader construction with a deterministic train/validation split.

The test set is never touched here; the validation split is carved from the
training set with a fixed seed so it is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from src.augmentations import get_train_transform, get_val_transform
from src.dataset.clover_dataset import CloverDataset
from src.utils.seed import worker_init_fn


def build_train_val_loaders(cfg, limit: int = 0):
    train_base = CloverDataset(
        image_dir=cfg.train_data_dir,
        mask_dir=cfg.train_mask_dir,
        transform=get_train_transform(cfg),
        mask_threshold=cfg.mask_binarize_threshold,
    )
    val_base = CloverDataset(
        image_dir=cfg.train_data_dir,
        mask_dir=cfg.train_mask_dir,
        transform=get_val_transform(cfg),
        mask_threshold=cfg.mask_binarize_threshold,
    )
    assert train_base.pairs == val_base.pairs, "train/val datasets must share ordering"

    n = len(train_base)
    if limit:
        n = min(limit, n)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(cfg.val_fraction * n)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, len(train_idx), len(val_idx)


def build_test_loader(cfg):
    ds = CloverDataset(
        image_dir=cfg.test_data_dir,
        mask_dir=cfg.test_mask_dir,
        transform=get_val_transform(cfg),
        mask_threshold=cfg.mask_binarize_threshold,
    )
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
    )