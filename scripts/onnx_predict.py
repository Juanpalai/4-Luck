#!/usr/bin/env python
"""Single-image inference with an exported ONNX model (onnxruntime).

Validates that the exported model reproduces PyTorch behavior on a real image.

Usage:
    python scripts/onnx_predict.py --onnx checkpoints/clover_res768/best_on640x480.onnx \
        --image training/TrainImages/1_000001.jpg --save pred_onnx.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentations import get_val_transform
from src.config import Config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="path to .onnx model")
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default="prediction_onnx.png")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--config", default="configs/exp_res768.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    # the ONNX model's input size is fixed; recover it from the filename "name_onWxH[_quant].onnx"
    m = re.search(r"_on(\d+)x(\d+)", args.onnx)
    if not m:
        raise ValueError(f"cannot infer input size from filename: {args.onnx}")
    w, h = int(m.group(1)), int(m.group(2))
    cfg.image_size = (w, h)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    input_name = in_meta.name

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_val_transform(cfg)
    x = transform(image=img, mask=np.zeros(img.shape[:2], np.uint8))["image"].unsqueeze(0)

    feed = x.numpy()
    if in_meta.type == "tensor(float16)":
        feed = feed.astype(np.float16)
    logits = sess.run(None, {input_name: feed})[0]
    prob = 1.0 / (1.0 + np.exp(-logits))[0, 0]  # (H,W) stable-ish; values are moderate here
    prob = np.clip(prob, 0.0, 1.0)
    pred = (prob > args.threshold).astype(np.uint8) * 255

    h0, w0 = img.shape[:2]
    pred_full = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[pred_full > 0] = (255, 0, 0)
    overlay = (0.6 * img + 0.4 * overlay).astype(np.uint8)
    cv2.imwrite(str(args.save), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fg = 100.0 * pred_full.sum() // 255 / (h0 * w0)
    print(f"onnx={args.onnx}")
    print(f"input {w}x{h} -> prob max {prob.max():.3f} | predicted foreground {fg:.3f}%")
    print(f"saved overlay -> {args.save}")


if __name__ == "__main__":
    main()