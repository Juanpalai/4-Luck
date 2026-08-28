# Four-Leaf Clover Detection

End-to-end four-leaf clover detection: **semantic segmentation** model trained
on clover-field photos with binary masks (white = four-leaf clover), targeting
eventual real-time mobile inference.

> Current status: **Phase 1 complete** — dataset analyzed, training pipeline
> implemented and tested. Training not yet run (run it yourself, see below).

## Dataset

```
training/
├── TrainImages/   1000 photos (800x600 RGB JPEG)
├── TrainLabel/    1000 binary masks (800x600, JPEG-encoded)
├── TestImages/    500 photos
└── TestLabels/    500 masks
```

Masks are JPEG files: values are near-black/white but **not exactly binary** due
to compression. They are binarized with a threshold (127) on load.

### Key analysis findings (see `experiments/dataset_analysis/`)

| | train | test |
|---|---|---|
| images / masks | 1000 / 1000 | 500 / 500 |
| size / mode | 800x600 RGB | 800x600 RGB |
| correspondence | 100% matched | 100% matched |
| empty masks (no clover) | 1 (0.1%) | 2 (0.4%) |
| foreground (median) | 0.33% | 0.71% |
| clover components | 1565 | 926 |
| comp. area px (q05 / med / q95) | 96 / 1017 / 3864 | 75 / 1711 / 7319 |
| comp. max-dim px (q05 / med / q95) | 15 / 43 / 88 | 16 / 59 / 124 |

Negative examples (images without any clover) are almost absent (0.1–0.4%), so
the model cannot learn "no clover" from the training distribution alone. This is
documented as a known limitation; candidate fixes: add negative-field photos,
or rely on a low-probability post-processing threshold for false-positive
control.

## Training resolution

Native images are 800x600. Targets are small: median clover ~43px across.
Clover size at each candidate input resolution (longest side resized to R):

| input | scale | typical (med) max-dim | small (q05) max-dim | typical area |
|---|---|---|---|---|
| 256 | 0.32 | 13.8 px | 4.8 px | 104 px² |
| 384 | 0.48 | 20.6 px | 7.2 px | 234 px² |
| 512 | 0.64 | 27.5 px | 9.6 px | 417 px² |
| **640x480** | **0.80** | **34.4 px** | **12.0 px** | **651 px²** |
| 768 | 0.96 | 41.3 px | 14.4 px | 937 px² |

**Recommended: 640x480** — preserves the 4:3 aspect ratio (no distortion or
padding), keeps the smallest clovers ≥12 px in feature-relevant size, and is a
realistic target for future mobile inference. 256/384 make small clovers
<10 px, which a lightweight segmentation network cannot resolve reliably.

**Crops/tiling are NOT necessary**: clovers are a small fraction of the image
but median ~43 px at native is comfortably larger than the ~12 px threshold
where segmentation degrades. Tiling would add little; it mainly helps when
targets are <1% of the image *and* very small, which is not the case here.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
# CUDA build (if you installed torch CPU above, reinstall for your GPU):
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Usage

### 1. Analyze the dataset (done, reproducible)

```bash
python scripts/analyze_dataset.py --data-root training --output-dir experiments/dataset_analysis
```

### 2. Train

```bash
python scripts/train.py                                  # defaults (configs/default.yaml)
python scripts/train.py --epochs 80 --batch-size 8       # override flags
python scripts/train.py --config my_config.yaml          # custom config
python scripts/train.py --dry-run                        # sanity check: 1 epoch, 2 batches
```

Checkpoints: `checkpoints/clover_unet/{best,last,epoch_N}.pt`. Best model is
selected by validation IoU. History/logs go to `experiments/clover_unet/`.

### 3. Evaluate on the held-out test set

```bash
python scripts/evaluate.py --checkpoint checkpoints/clover_unet/best.pt --save-vis 12
```

Reports IoU / Dice / Precision / Recall / F1 and writes prediction overlays to
`experiments/evaluation/`.

## Architecture

Lightweight **U-Net** with a pretrained **MobileNetV3-Small** encoder
(torchvision, 2.5M params). Decoder is a standard 4-level up-block chain with
skip connections. ~2.2M params total (~8.7 MB fp32) with the default
`decoder_channels=96` — mobile-friendly.

- `output_stride=32` (default): full encoder, bottleneck 576ch @1/32.
- `output_stride=16`: encoder truncated to @1/16, better for small targets.

> Memory note: measured on an 8 GB RTX 2070 SUPER, the default config
> (`640x480`, batch 8, AMP) peaks at ~4.1 GB, leaving headroom.

## Loss

**BCE + Dice** (default, equal weights). Rationale: foreground is ~0.3% of the
image, so plain BCE is background-dominated; Dice directly optimizes overlap and
handles imbalance; BCE keeps calibration. Alternatives: `--loss dice`, `bce`,
`focal`.

## Augmentation

Albumentations (image+mask applied identically): horizontal flip (0.5), slight
shift/scale/rotate (±15°, ±10%), brightness/contrast (0.2), gamma, light blur.
Masks resized with nearest-neighbor to stay binary.

## Metrics

Pixel-level IoU, Dice, Precision, Recall, F1 at threshold 0.5 (configurable),
macro-averaged over the set.

## Reproducibility

Fixed seed (42) for both numpy/torch and the DataLoader workers; deterministic
cuDNN; the train/val split is derived from the same seed. Every run records its
full config to `experiments/<name>/config.json` / `config.yaml`.

## Performance results (baseline)

Trained on the full train split (900/100 val, seed 42), 60 epochs, 640×480,
batch 8, AMP, BCE+Dice, cosine LR 1e-3, MobileNetV3-Small U-Net (2.2M params),
~32 s/epoch on an RTX 2070 SUPER.

**Validation** (best epoch 52, `checkpoints/clover_unet/best.pt`):

| IoU | Dice | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.250 | 0.399 | 0.414 | 0.384 | 0.398 |

**Test set** (held-out, 500 images, threshold 0.5):

| IoU | Dice | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.169 | 0.289 | 0.418 | 0.221 | 0.289 |

Test performance is lower than validation — the test set has larger/more
clovers (median foreground 0.71% vs 0.33%) plus JPEG mask noise. Recall is the
main weakness (0.22), suggesting small/hard clovers are missed. Obvious next
steps: train at 768×576, add random-scale augmentation, increase epochs, or
tune the post-processing threshold.

## Known limitations

1. **No negative images** — only 0.1–0.4% of masks are empty. False-positive
   control must be handled via threshold/post-processing; ideally collect
   clover-field photos without any four-leaf clovers.
2. Mobile/ONNX deployment is a later phase and not implemented yet.

## Project layout

```
scripts/          analyze_dataset.py, train.py, evaluate.py
src/config.py     centralized config (dataclass + YAML)
src/dataset/      CloverDataset, mask binarization, loader builders
src/models/       LightweightUNet (MobileNetV3-Small encoder)
src/training/     losses, trainer (AMP, checkpointing)
src/evaluation/   metrics (IoU/Dice/P/R/F1)
src/augmentations.py
experiments/      dataset analysis artifacts
checkpoints/      saved models
```