#!/usr/bin/env python
"""Benchmark inference latency/FPS of exported ONNX models and PyTorch.

Measures end-to-end inference time (excluding preprocessing) with warmup.
Covers torch CUDA (reference), torch CPU, and onnxruntime CPU for fp32/fp16/int8
at any exported resolution.

Usage:
    python scripts/benchmark_onnx.py --checkpoint checkpoints/clover_res768/best.pt \
        --config configs/exp_res768.yaml
    python scripts/benchmark_onnx.py --onnx checkpoints/clover_res768/best_on640x480_int8.onnx
    python scripts/benchmark_onnx.py --all --out experiments/onnx_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.models.unet_mobilenet import build_model
from src.training.trainer import load_checkpoint

REPS = 30
WARMUP = 5


def bench_cuda(cfg, ckpt, reps=REPS) -> dict:
    model = build_model(cfg).to("cuda").eval()
    load_checkpoint(Path(ckpt), model, device="cuda")
    W, H = cfg.image_size
    x = torch.randn(1, 3, H, W, device="cuda")
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
    return {"latency_ms": (time.perf_counter() - t0) / reps * 1e3}


def bench_torch_cpu(cfg, ckpt, reps=REPS) -> dict:
    torch.set_num_threads(4)
    model = build_model(cfg).eval()
    load_checkpoint(Path(ckpt), model, device="cpu")
    W, H = cfg.image_size
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
    return {"latency_ms": (time.perf_counter() - t0) / reps * 1e3}


def bench_onnx(onnx_path, reps=REPS) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    shape = in_meta.shape  # [1,3,H,W]
    dtype = np.float16 if in_meta.type == "tensor(float16)" else np.float32
    x = np.random.rand(*shape).astype(dtype)
    input_name = in_meta.name
    for _ in range(WARMUP):
        sess.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {input_name: x})
    return {"latency_ms": (time.perf_counter() - t0) / reps * 1e3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="torch checkpoint")
    ap.add_argument("--config", default="configs/exp_res768.yaml")
    ap.add_argument("--onnx", help="single onnx model to benchmark")
    ap.add_argument("--all", action="store_true", help="benchmark every exported onnx under the checkpoint dir")
    ap.add_argument("--out", default="experiments/onnx_benchmark.json")
    args = ap.parse_args()

    results = []
    ckpt_dir = Path(args.checkpoint).parent if args.checkpoint else Path("checkpoints/clover_res768")

    if args.checkpoint:
        cfg = Config.from_yaml(args.config)
        for name, fn in [("torch_cuda", bench_cuda), ("torch_cpu", bench_torch_cpu)]:
            if name == "torch_cuda" and not torch.cuda.is_available():
                continue
            m = fn(cfg, args.checkpoint)
            m.update({"backend": name, "size_mb": Path(args.checkpoint).stat().st_size / 1e6,
                      "input": f"{cfg.image_size[0]}x{cfg.image_size[1]}"})
            results.append(m)
            print(m)

    onnx_files = [Path(args.onnx)] if args.onnx else sorted(ckpt_dir.glob("*.onnx"))
    for f in onnx_files:
        m = bench_onnx(str(f))
        m.update({"backend": f"onnx_{f.stem}", "size_mb": f.stat().st_size / 1e6,
                  "input": re.search(r"_on(\d+)x(\d+)", f.name).group(0)[1:]})
        m["fps"] = round(1000 / m["latency_ms"], 1)
        results.append(m)
        print(f"{f.name}: {m['latency_ms']:.1f} ms/image ({m['fps']} fps), {m['size_mb']:.2f} MB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()