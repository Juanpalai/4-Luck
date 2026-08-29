#!/usr/bin/env python
"""Error analysis for the clover segmentation model.

For every test image it detects ground-truth clover components (connected
components in the binary mask) and predicted components, then classifies
each detection:
  - detected: predicted component overlapping a GT component (IoU >= min_iou)
  - false positive: predicted component matching no GT
  - false negative: GT component with no overlapping prediction

Detection rate is broken down by GT component size (max dimension) to expose
whether small clovers are being missed. Also reports per-image IoU distribution.

Usage:
    python scripts/analyze_errors.py --checkpoint checkpoints/clover_unet/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset.build import build_test_loader
from src.models.unet_mobilenet import build_model
from src.training.trainer import load_checkpoint
from src.utils.seed import get_device, set_seed

MIN_CC_AREA = 5
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 4096)]


def components(binary: np.ndarray) -> list[np.ndarray]:
    """Return list of masks, one per connected component (min area filtered)."""
    if binary.max() == 0:
        return []
    num, labels, stat, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, num):
        if stat[i][4] < MIN_CC_AREA:
            continue
        out.append(labels == i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/clover_unet/best.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.1, help="component matching IoU")
    ap.add_argument("--out", default="experiments/error_analysis.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.threshold = args.threshold
    device = get_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)

    model = build_model(cfg).to(device)
    load_checkpoint(Path(args.checkpoint), model, device=device)
    loader = build_test_loader(cfg)

    per_image_iou = []
    per_image_det = []
    gt_components_by_size = {b: [0, 0] for b in SIZE_BINS}  # detected / total
    false_positives = 0
    n_pred = 0
    n_gt = 0
    details = []

    with torch.no_grad():
        sample_idx = 0
        for images, masks in tqdm(loader, desc="test", leave=False, mininterval=5.0):
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
            masks = masks.numpy()  # (B,1,H,W)

            for b in range(images.size(0)):
                gt = (masks[b, 0] > 0.5).astype(np.uint8)
                pred = (probs[b, 0] > args.threshold).astype(np.uint8)

                # pixel IoU
                inter = int(((gt > 0) & (pred > 0)).sum())
                union = int(((gt > 0) | (pred > 0)).sum())
                per_image_iou.append(inter / union if union else 1.0)

                gt_comps = components(gt)
                pred_comps = components(pred)
                n_gt += len(gt_comps)
                n_pred += len(pred_comps)

                matched_gt = np.zeros(len(gt_comps), dtype=bool)
                matched_pred = np.zeros(len(pred_comps), dtype=bool)
                for pi, pm in enumerate(pred_comps):
                    for gi, gm in enumerate(gt_comps):
                        iou = (pm & gm).sum() / (pm | gm).sum()
                        if iou >= args.min_iou and not matched_gt[gi] and not matched_pred[pi]:
                            matched_gt[gi] = True
                            matched_pred[pi] = True
                            break
                for gi, gm in enumerate(gt_comps):
                    max_dim = int(max((gm.any(0)).sum(), (gm.any(1)).sum()))
                    for lo, hi in SIZE_BINS:
                        if lo <= max_dim < hi:
                            gt_components_by_size[(lo, hi)][1] += 1
                            if matched_gt[gi]:
                                gt_components_by_size[(lo, hi)][0] += 1
                            break
                false_positives += int(matched_pred.sum() == 0 and len(pred_comps) > 0)  # images with stray FP
                fp_this = int((~matched_pred).sum())
                details.append(
                    {
                        "image": loader.dataset.pairs[sample_idx][0].name if sample_idx < len(loader.dataset.pairs) else "?",
                        "iou": round(per_image_iou[-1], 3),
                        "gt": len(gt_comps),
                        "pred": len(pred_comps),
                        "fn": int((~matched_gt).sum()),
                        "fp": fp_this,
                    }
                )
                sample_idx += 1

    per_image_iou = np.asarray(per_image_iou)
    report = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "min_iou": args.min_iou,
        "n_images": len(per_image_iou),
        "n_gt_components": n_gt,
        "n_pred_components": n_pred,
        "global_false_positives": false_positives,
        "per_image_iou": {
            "mean": round(float(per_image_iou.mean()), 3),
            "median": round(float(np.median(per_image_iou)), 3),
            "q25": round(float(np.percentile(per_image_iou, 25)), 3),
            "q75": round(float(np.percentile(per_image_iou, 75)), 3),
        },
        "detection_rate_by_gt_size": {
            f"{lo}-{hi}px": {"detected": d, "total": t, "rate": round(d / t, 3) if t else None}
            for (lo, hi), (d, t) in gt_components_by_size.items()
        },
        "worst_images": sorted(details, key=lambda d: d["iou"])[:10],
        "best_images": sorted(details, key=lambda d: d["iou"], reverse=True)[:5],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k not in ("worst_images", "best_images")}, indent=2))
    print("worst images:", [(d["image"], d["iou"]) for d in report["worst_images"]])
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()