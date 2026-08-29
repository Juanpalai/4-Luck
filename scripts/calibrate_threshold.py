#!/usr/bin/env python
"""Calibrate the prediction threshold on the validation split.

0.5 is arbitrary. This sweeps thresholds and reports metrics on validation
(never the test set) so the operating point can be chosen for the desired
precision/recall trade-off — important because the dataset has almost no
negative examples, so false-positive control relies on the threshold.

Usage:
    python scripts/calibrate_threshold.py --checkpoint checkpoints/clover_res768/best.pt \
        --config configs/exp_res768.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset.build import build_train_val_loaders
from src.evaluation.metrics import SegmentationMetrics
from src.models.unet_mobilenet import build_model
from src.training.trainer import load_checkpoint
from src.utils.seed import get_device, set_seed

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="experiments/threshold_calibration.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)

    model = build_model(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, device=device)

    _, val_loader, _, _ = build_train_val_loaders(cfg)

    # collect raw logits to sweep thresholds without re-running inference
    logits_all, masks_all = [], []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="val", leave=False, mininterval=5.0):
            logits_all.append(model(images.to(device)).float().cpu())
            masks_all.append(masks.cpu())
    logits = torch.cat(logits_all)
    masks = torch.cat(masks_all)

    results = []
    for t in THRESHOLDS:
        m = SegmentationMetrics(threshold=t)
        m.update(logits, masks)
        s = m.summary()
        results.append({"threshold": t, **s})
        print(f"t={t}  IoU {s['iou']:.4f}  Dice {s['dice']:.4f}  P {s['precision']:.3f}  "
              f"R {s['recall']:.3f}  F1 {s['f1']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()