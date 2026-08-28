"""Centralized configuration for the clover segmentation project.

All tunable parameters live in one place. Config can be built from a YAML
file and/or overridden via command line. Paths are relative to the project
root unless absolute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- data ---
    data_root: str = "training"
    train_images: str = "TrainImages"
    train_masks: str = "TrainLabel"
    test_images: str = "TestImages"
    test_masks: str = "TestLabels"
    mask_binarize_threshold: int = 127

    # --- input ---
    image_size: tuple[int, int] = (640, 480)  # (W, H), preserves 800x600 aspect ratio

    # --- data split ---
    val_fraction: float = 0.10
    seed: int = 42

    # --- training ---
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine | step | none
    lr_scheduler_step_size: int = 20
    lr_scheduler_gamma: float = 0.5
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 0.0  # 0 = disabled
    save_every: int = 5  # checkpoint every N epochs (0 = only best/last)

    # --- model ---
    encoder: str = "mobilenet_v3_small"
    encoder_pretrained: bool = True
    decoder_channels: int = 96
    output_stride: int = 32  # 32 uses full encoder; 16 drops last downsample

    # --- loss ---
    loss: str = "bce_dice"  # bce | dice | bce_dice | focal
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- post-processing / eval ---
    threshold: float = 0.5
    min_cc_area: int = 8  # drop connected components below this (eval/inference only)

    # --- augmentation (flags; probabilities live in src/augmentations.py) ---
    augment: bool = True
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    rotate_limit: int = 15
    scale_limit: float = 0.1
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gamma_limit: tuple[float, float] = (80.0, 120.0)
    blur_p: float = 0.1

    # --- paths ---
    checkpoint_dir: str = "checkpoints"
    experiment_dir: str = "experiments"
    experiment_name: str = "clover_unet"
    resume: str = ""  # path to checkpoint to resume from

    # --- reproducibility ---
    deterministic: bool = True

    # --- runtime ---
    device: str = "auto"  # auto | cuda | cpu

    # ------------------------------------------------------------------ utils
    @property
    def exp_dir(self) -> Path:
        return Path(self.experiment_dir) / self.experiment_name

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.experiment_name

    @property
    def train_data_dir(self) -> Path:
        return (Path(self.data_root) / self.train_images).resolve()

    @property
    def train_mask_dir(self) -> Path:
        return (Path(self.data_root) / self.train_masks).resolve()

    @property
    def test_data_dir(self) -> Path:
        return (Path(self.data_root) / self.test_images).resolve()

    @property
    def test_mask_dir(self) -> Path:
        return (Path(self.data_root) / self.test_masks).resolve()

    def resolve_path(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else PROJECT_ROOT / pp

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {fld.name for fld in fields(cls)}
        cfg = cls()
        for k, v in raw.items():
            if k not in known:
                raise ValueError(f"unknown config key '{k}' in {path}")
            setattr(cfg, k, v)
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def update_from_dict(self, overrides: dict) -> None:
        known = {fld.name for fld in fields(self)}
        for k, v in overrides.items():
            if k not in known:
                raise ValueError(f"unknown config key '{k}'")
            setattr(self, k, v)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)