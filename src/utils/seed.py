"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def set_num_threads(cfg: object) -> None:
    """Limit threads on CPU to the configured worker count (avoids oversubscription)."""
    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, cfg.num_workers))