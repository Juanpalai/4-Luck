#!/usr/bin/env python
"""Run single-image inference with a trained checkpoint.

Usage:
    python scripts/predict.py --checkpoint checkpoints/clover_unet/best.pt \
        --image training/TrainImages/1_000001.jpg --save output.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentations import get_val_transform
from src.config import Config
from src.models.unet_mobilenet import build_model
from src.training.trainer import load_checkpoint
from src.utils.seed import get_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default="prediction.png")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.threshold is not None:
        cfg.threshold = args.threshold
    device = get_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)

    model = build_model(cfg).to(device)
    state = load_checkpoint(Path(args.checkpoint), model, device=device)
    print(f"loaded {args.checkpoint} (epoch {state.get('epoch')})")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_val_transform(cfg)
    x = transform(image=img, mask=np.zeros(img.shape[:2], np.uint8))["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
    prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    pred = (prob > cfg.threshold).astype(np.uint8) * 255
    prob = (prob * 255).astype(np.uint8)

    # upscale prediction back to native resolution (nearest to stay binary)
    h, w = img.shape[:2]
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img.copy()
    overlay[pred_full > 0] = (255, 0, 0)
    overlay = (0.6 * img + 0.4 * overlay).astype(np.uint8)

    out = Path(args.save)
    cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    fg_pct = 100.0 * pred_full.sum() // 255 / (h * w)
    print(f"saved overlay -> {out} | predicted foreground {fg_pct:.3f}% "
          f"(prob mean {prob.mean():.1f} max {prob.max()})")


if __name__ == "__main__":
    main()