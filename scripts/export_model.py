#!/usr/bin/env python
"""Export a trained checkpoint to ONNX and verify output parity with PyTorch.

Exports at a fixed input size and batch 1 (mobile-friendly). HardSwish/Resize/
BatchNorm ops are all supported by onnxruntime (CPU and Android). Optionally
exports fp16 (half) or applies onnxruntime dynamic int8 quantization.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/clover_res768/best.pt \
        --config configs/exp_res768.yaml
    python scripts/export_model.py --checkpoint ... --input-size 640,480 --fp16
    python scripts/export_model.py --checkpoint ... --int8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.models.unet_mobilenet import build_model
from src.training.trainer import load_checkpoint

ONNX_OPSET = 17


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (avoids overflow on large negative logits)."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/clover_res768/best.pt")
    ap.add_argument("--config", default="configs/exp_res768.yaml")
    ap.add_argument("--input-size", default=None, help="W,H override e.g. 640,480 (default: from config)")
    ap.add_argument("--output", default=None, help="output .onnx path")
    ap.add_argument("--fp16", action="store_true", help="export in fp16 (half weights)")
    ap.add_argument("--int8", action="store_true", help="apply onnxruntime dynamic int8 quantization")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.input_size:
        w, h = args.input_size.split(",")
        cfg.image_size = (int(w), int(h))
    model = build_model(cfg)
    state = load_checkpoint(Path(args.checkpoint), model, device="cpu")
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {state.get('epoch')}, val IoU "
          f"{state.get('val_metrics', {}).get('iou', 'n/a')})")

    W, H = cfg.image_size
    ckpt_stem = Path(args.checkpoint).stem  # e.g. "best"
    out_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{ckpt_stem}_on{ W}x{H}.onnx"

    dummy = torch.randn(1, 3, H, W)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed input size (batch 1) for mobile
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter: self-contained file, exact parity
    )

    # ---- parity check (torch fp32 vs onnx fp32), in probability space ----
    with torch.no_grad():
        torch_prob = torch.sigmoid(model(dummy)).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input": dummy.numpy()})[0]
    ort_prob = _sigmoid(ort_logits)
    diff = np.abs(ort_prob - torch_prob).max()
    print(f"exported -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB, opset {ONNX_OPSET})")
    print(f"parity torch<->onnx fp32: max prob abs diff = {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError("ONNX export deviates too much from PyTorch")

    # ---- fp16 (weights cast to half) ----
    if args.fp16:
        from onnxconverter_common import float16  # available with onnxconverter-common? fallback below
        fp16_path = out_path.with_name(out_path.stem + "_fp16.onnx")
        try:
            fp16_model = float16.convert_float_to_float16(onnx.load(str(out_path)))
            onnx.save(fp16_model, str(fp16_path))
        except ImportError:
            # manual fp16: cast all float weights/initializers and tensor dtypes
            m = onnx.load(str(out_path))
            for init in m.graph.initializer:
                if init.data_type == onnx.TensorProto.FLOAT:
                    init.data_type = onnx.TensorProto.FLOAT16
                    from onnx import numpy_helper
                    init.raw_data = numpy_helper.from_array(
                        numpy_helper.to_array(init).astype(np.float16)).raw_data
            for node in m.graph.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.FLOAT:
                        attr.t.data_type = onnx.TensorProto.FLOAT16
            onnx.save(m, str(fp16_path))
        onnx.checker.check_model(onnx.load(str(fp16_path)))
        print(f"fp16 -> {fp16_path} ({fp16_path.stat().st_size/1e6:.2f} MB)")

    # ---- int8 (dynamic quantization) ----
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = out_path.with_name(out_path.stem + "_int8.onnx")
        quantize_dynamic(str(out_path), str(int8_path), weight_type=QuantType.QInt8)
        print(f"int8 -> {int8_path} ({int8_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()