#!/usr/bin/env python
"""Evaluate a trained model on the held-out test set.

Supports PyTorch checkpoints and exported ONNX models (fp32/fp16/int8).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/clover_unet/best.pt
    python scripts/evaluate.py --backend onnx --onnx checkpoints/clover_res768/best_on640x480.onnx
    python scripts/evaluate.py --checkpoint <path> --threshold 0.5 --save-vis 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset.build import build_test_loader
from src.evaluation.metrics import SegmentationMetrics
from src.models.unet_mobilenet import build_model
from src.training.losses import build_loss
from src.training.trainer import load_checkpoint
from src.utils.seed import get_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    ap.add_argument("--checkpoint", help="torch checkpoint (backend=torch)")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--onnx", help="path to .onnx model (backend=onnx)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image-size", help="W,H override e.g. 640,480")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--save-vis", type=int, default=0, help="save N prediction overlays")
    ap.add_argument("--out-dir", default="experiments/evaluation")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in {"config", "checkpoint", "save_vis", "out_dir",
                                                "backend", "onnx"}}
    cfg.update_from_dict(overrides)
    if args.image_size:
        w, h = args.image_size.split(",")
        cfg.image_size = (int(w), int(h))

    device = get_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)

    session = None
    if args.backend == "onnx":
        if not args.onnx:
            raise SystemExit("--onnx is required with --backend onnx")
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        # the ONNX model's input size is fixed; recover it from "name_onWxH[_quant].onnx"
        m = re.search(r"_on(\d+)x(\d+)", args.onnx)
        if not m:
            raise SystemExit(f"cannot infer input size from ONNX filename: {args.onnx}")
        w, h = int(m.group(1)), int(m.group(2))
        cfg.image_size = (w, h)
        cfg.batch_size = 1  # ONNX models are exported with a fixed batch of 1
        print(f"onnx session loaded: {args.onnx}")
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --backend torch")
        model = build_model(cfg).to(device)
        state = load_checkpoint(Path(args.checkpoint), model, device=device)
        print(f"loaded {args.checkpoint} (epoch {state.get('epoch')}, "
              f"val IoU {state.get('val_metrics', {}).get('iou', 'n/a')})")

    loader = build_test_loader(cfg)
    metrics = SegmentationMetrics(threshold=cfg.threshold)
    criterion = build_loss(cfg) if args.backend == "torch" else None
    total_loss, n = 0.0, 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="test", mininterval=5.0):
            if args.backend == "torch":
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                total_loss += loss.item() * images.size(0)
                n += images.size(0)
            else:
                input_name = session.get_inputs()[0].name
                if session.get_inputs()[0].type == "tensor(float16)":
                    feed = images.half().numpy()
                else:
                    feed = images.numpy()
                out = session.run(None, {input_name: feed})[0]
                logits = torch.from_numpy(np.asarray(out)).float()
            metrics.update(logits.float(), masks.float())

            if saved < args.save_vis:
                for i in range(images.size(0)):
                    if saved >= args.save_vis:
                        break
                    save_overlay(images[i], masks[i], logits[i], cfg, out_dir, saved, 0)
                    saved += 1

    summary = metrics.summary()
    if args.backend == "torch":
        summary["loss"] = round(total_loss / n, 4)
    summary["threshold"] = cfg.threshold
    summary["image_size"] = cfg.image_size
    summary["model"] = str(args.checkpoint or args.onnx)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"visualizations -> {out_dir}")


@torch.no_grad()
def save_overlay(image, mask, logits, cfg, out_dir: Path, idx: int, epoch: int) -> None:
    """Render original | ground truth | prediction | overlay for one sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # un-normalize image
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = image.cpu().numpy() * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

    prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    pred = (prob > cfg.threshold).astype(np.uint8)

    gt = mask.squeeze(0).cpu().numpy()
    gt = (gt > 0.5).astype(np.uint8)

    def overlay(a):
        o = a.copy()
        o[pred > 0] = (255, 0, 0)
        return (0.6 * a + 0.4 * o).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(img); axes[0].set_title("image")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("ground truth")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title(f"prediction (t={cfg.threshold})")
    axes[3].imshow(overlay(img)); axes[3].set_title("overlay")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_{idx:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()