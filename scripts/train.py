#!/usr/bin/env python
"""Train the four-leaf clover segmentation model.

Usage:
    python scripts/train.py                       # defaults from Config
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --epochs 80 --batch-size 8 --image-size 768,576
    python scripts/train.py --dry-run             # 2 batches, 1 epoch, no checkpoint save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import Config
from src.dataset.build import build_train_val_loaders
from src.models.unet_mobilenet import build_model
from src.training.losses import build_loss
from src.training.trainer import run_training
from src.utils.seed import get_device, set_num_threads, set_seed


def parse_image_size(s: str) -> tuple[int, int]:
    w, h = s.split(",")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train clover segmentation model")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--image-size", type=parse_image_size, help="W,H e.g. 640,480")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--output-stride", type=int, choices=[16, 32])
    ap.add_argument("--loss", choices=["bce", "dice", "bce_dice", "focal"])
    ap.add_argument("--experiment-name")
    ap.add_argument("--resume")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="train on a random subset of N images (testing)")
    ap.add_argument("--dry-run", action="store_true", help="run 1 epoch on 2 batches to sanity-check the pipeline")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    from dataclasses import fields as _fields
    known = {f.name for f in _fields(Config)}
    cfg.update_from_dict({k: v for k, v in vars(args).items() if v is not None and k in known})
    if args.no_amp:
        cfg.amp = False
    dry_run = args.dry_run

    device = get_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)
    set_num_threads(cfg)
    print(f"device: {device} | torch {torch.__version__} | "
          f"cuda {torch.cuda.get_device_name(0) if device.type=='cuda' else 'n/a'}")

    if dry_run:
        cfg.epochs = 1
        cfg.batch_size = 2
        cfg.save_every = 0
        cfg.experiment_name = cfg.experiment_name + "_dryrun"

    model = build_model(cfg).to(device)
    print(f"model: {cfg.encoder} U-Net | params {model.num_params:,} "
          f"({model.num_params*4/1e6:.1f} MB fp32)")

    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma)

    if cfg.resume:
        from src.training.trainer import load_checkpoint
        load_checkpoint(Path(cfg.resume), model, optimizer, scheduler, device)
        print(f"resumed from {cfg.resume}")

    train_loader, val_loader, n_tr, n_val = build_train_val_loaders(
        cfg, limit=12 if dry_run else args.limit)
    print(f"train images: {n_tr} | val images: {n_val} | image size {cfg.image_size} | batch {cfg.batch_size}")

    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(cfg.exp_dir / "config.json")
    cfg.to_yaml(cfg.exp_dir / "config.yaml")

    run_training(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device)


if __name__ == "__main__":
    main()