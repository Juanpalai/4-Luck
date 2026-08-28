"""Phase 1 dataset analysis for the four-leaf clover segmentation project.

Analyzes the clover-field dataset programmatically:
- counts, correspondence, formats, dimensions
- mask binarization quality (JPEG artifacts)
- foreground statistics (empty masks, foreground %)
- four-leaf clover pixel-size statistics via connected components
- relative target size + candidate-resolution analysis
- visual samples and target-size distribution plots

Writes a JSON report and PNG figures to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Components at least this many pixels are counted as a clover (filters JPEG noise).
MIN_COMPONENT_AREA = 5

# Candidate input resolutions (longest side) to evaluate for training.
CANDIDATE_RESOLUTIONS = [256, 320, 384, 512, 640, 768]

# Quantiles of interest for target-size statistics.
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def list_pairs(root: Path, img_dir: str, mask_dir: str) -> tuple[list[Path], list[Path], list[str]]:
    imgs = sorted(p for p in (root / img_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    masks = sorted(p for p in (root / mask_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    img_names = {p.stem for p in imgs}
    mask_names = {p.stem for p in masks}
    mismatched = sorted((img_names - mask_names) | (mask_names - img_names))
    return imgs, masks, mismatched


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    """Load a JPEG-encoded mask and binarize it (0=bg, 255=fg)."""
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8) * 255


def component_stats(mask: np.ndarray) -> list[dict]:
    """Per-component stats for a binary mask (clover blobs)."""
    stats = []
    if mask.max() == 0:
        return stats
    _, labels, stat, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, labels.max() + 1):
        x, y, w, h, area = stat[i]
        if area < MIN_COMPONENT_AREA:
            continue
        stats.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "max_dim": int(max(w, h)),
                "min_dim": int(min(w, h)),
                "area": int(area),
            }
        )
    return stats


def summarize(values, precision: int = 2) -> dict:
    if len(values) == 0:
        return {"count": 0}
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [q * 100 for q in QUANTILES])
    return {
        "count": int(len(v)),
        "min": round(float(v.min()), precision),
        "max": round(float(v.max()), precision),
        "mean": round(float(v.mean()), precision),
        "median": round(float(np.median(v)), precision),
        "std": round(float(v.std()), precision),
        "q05": round(float(q[0]), precision),
        "q25": round(float(q[1]), precision),
        "q75": round(float(q[3]), precision),
        "q95": round(float(q[4]), precision),
    }


def analyze_split(root: Path, img_dir: str, mask_dir: str, split: str) -> tuple[dict, dict, list]:
    """Return (json_report, raw_arrays, sample_names)."""
    imgs, masks, mismatched = list_pairs(root, img_dir, mask_dir)
    name_to_mask = {p.stem: p for p in masks}

    fg_pcts, img_sizes, mask_sizes, modes = [], {}, {}, {}
    empty_count, ambiguous = 0, []
    comps_all = []
    sizes_seen = {}

    for img_p in imgs:
        mask_p = name_to_mask.get(img_p.stem)
        if mask_p is None:
            continue
        with Image.open(img_p) as im:
            sizes_seen[(im.size[0], im.size[1])] = sizes_seen.get((im.size[0], im.size[1]), 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1

        mask = load_mask(mask_p)
        mask_sizes[(mask.shape[1], mask.shape[0])] = mask_sizes.get((mask.shape[1], mask.shape[0]), 0) + 1

        fg = int(mask.sum() // 255)
        fg_pcts.append(100.0 * fg / mask.size)
        if fg == 0:
            empty_count += 1

        gray = np.asarray(Image.open(mask_p).convert("L"))
        ambiguous.append(int(((gray > 10) & (gray < 245)).sum()))

        comps_all.extend(component_stats(mask))

    fg_pcts = np.asarray(fg_pcts, dtype=np.float64)
    areas = np.asarray([c["area"] for c in comps_all], dtype=np.float64)
    dims = np.asarray([c["max_dim"] for c in comps_all], dtype=np.float64)
    rel_areas = areas / (800 * 600) if areas.size else areas

    # pick representative sample names (3 mid-sized + smallest + largest)
    samples = []
    if comps_all:
        target = float(np.median(areas))
        order = sorted(range(len(comps_all)), key=lambda i: abs(areas[i] - target))
        mid_idx = order[:3]
        mid_names = [imgs[0].parent.name]  # placeholder
        largest = int(np.argmax(areas))
        smallest = int(np.argmin(areas))
        cand = [comps_all[i] for i in mid_idx] + [comps_all[largest], comps_all[smallest]]
        # We don't retain image mapping here; visualize() picks images with masks by FG size.
    # visualize will re-derive representative images; here just keep counts.

    report = {
        "split": split,
        "image_dir": img_dir,
        "mask_dir": mask_dir,
        "n_images": len(imgs),
        "n_masks": len(masks),
        "mismatched_names": mismatched,
        "image_sizes": {f"{w}x{h}": c for (w, h), c in sorted(sizes_seen.items())},
        "mask_sizes": {f"{w}x{h}": c for (w, h), c in sorted(mask_sizes.items())},
        "image_modes": modes,
        "empty_masks": empty_count,
        "empty_mask_pct": round(100.0 * empty_count / len(imgs), 3),
        "fg_pct": summarize(fg_pcts),
        "ambiguous_pixels_per_image": summarize(ambiguous),
        "clover_components": len(comps_all),
        "comp_area": summarize(areas),
        "comp_w": summarize([c["w"] for c in comps_all]),
        "comp_h": summarize([c["h"] for c in comps_all]),
        "comp_max_dim": summarize(dims),
        "comp_min_dim": summarize([c["min_dim"] for c in comps_all]),
        "comp_rel_area": summarize(rel_areas, precision=6),
    }
    raw = {
        "comp_area": areas,
        "comp_max_dim": dims,
        "comp_rel_area": rel_areas,
        "fg_pct": fg_pcts,
        "comps": comps_all,
    }
    return report, raw, imgs


def resolution_analysis(train_report: dict) -> dict:
    native_w, native_h = next(tuple(map(int, k.split("x"))) for k in train_report["image_sizes"])
    stats = train_report["comp_max_dim"]
    area = train_report["comp_area"]
    out = {}
    for res in CANDIDATE_RESOLUTIONS:
        scale = res / native_w  # longest side resized to res
        out[str(res)] = {
            "scale": round(scale, 4),
            "typical_max_dim_px": round(stats["median"] * scale, 1),
            "small_q05_max_dim_px": round(stats["q05"] * scale, 1),
            "typical_area_px": round(area["median"] * scale ** 2, 1),
            "small_q05_area_px": round(area["q05"] * scale ** 2, 1),
        }
    return {"native_size": f"{native_w}x{native_h}", "candidates": out}


def visualize(root: Path, report: dict, out_dir: Path, n: int = 5) -> None:
    """Render image | mask | overlay rows for representative samples."""
    img_dir = root / report["image_dir"]
    mask_dir = root / report["mask_dir"]

    # rank images by total foreground size to pick small/typical/large
    scored = []
    for img_p in img_dir.iterdir():
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_p = mask_dir / img_p.name
        if not mask_p.exists():
            continue
        fg = int(load_mask(mask_p).sum() // 255)
        scored.append((img_p.stem, fg))
    scored.sort(key=lambda x: x[1])
    if not scored:
        return
    # typical = median foreground
    med_fg = scored[len(scored) // 2][1]
    typical = min(scored, key=lambda x: abs(x[1] - med_fg))[0]
    names = [scored[0][0], typical, scored[-1][0]]
    # add two more spread samples if available
    if len(scored) > 3:
        names += [scored[len(scored) // 4][0], scored[3 * len(scored) // 4][0]]
    names = list(dict.fromkeys(names))[:n]

    fig, axes = plt.subplots(len(names), 3, figsize=(12, 4.2 * len(names)))
    if len(names) == 1:
        axes = axes[None, :]
    for r, name in enumerate(names):
        img = np.asarray(Image.open(img_dir / f"{name}.jpg").convert("RGB"))
        mask = load_mask(mask_dir / f"{name}.jpg")
        overlay = img.copy()
        overlay[mask > 0] = (255, 0, 0)
        overlay = (0.6 * img + 0.4 * overlay).astype(np.uint8)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"image {name}")
        axes[r, 1].imshow(mask, cmap="gray")
        axes[r, 1].set_title("ground-truth mask")
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("overlay")
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle(f"{report['split']} — representative samples (small / typical / large)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{report['split']}_samples.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / f'{report['split']}_samples.png'}")


def distribution_plots(report: dict, raw: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.suptitle(f"{report['split']} — four-leaf clover size distribution")

    axes[0].hist(raw["comp_area"], bins=60, color="#2a7fbc")
    axes[0].set_xlabel("component area (px @800x600)")
    axes[0].set_ylabel("count")
    axes[0].axvline(report["comp_area"]["median"], color="r", ls="--", label=f"median {report['comp_area']['median']:.0f}")

    axes[1].hist(raw["comp_max_dim"], bins=40, color="#4cae4c")
    axes[1].set_xlabel("component max dimension (px)")
    axes[1].axvline(report["comp_max_dim"]["median"], color="r", ls="--", label=f"median {report['comp_max_dim']['median']:.0f}")

    axes[2].hist(raw["comp_rel_area"], bins=60, color="#c29b4c")
    axes[2].set_xlabel("component area / image area")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{report['split']}_size_distribution.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / f'{report['split']}_size_distribution.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="training", help="root containing TrainImages etc.")
    ap.add_argument("--output-dir", default="experiments/dataset_analysis", help="where to write report + figures")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_cfg = [
        ("TrainImages", "TrainLabel", "train"),
        ("TestImages", "TestLabels", "test"),
    ]

    report = {"root": str(root.resolve())}
    for img_dir, mask_dir, split in splits_cfg:
        print(f"\nAnalyzing {split} ({img_dir}/{mask_dir})...")
        rep, raw, _ = analyze_split(root, img_dir, mask_dir, split)
        report[split] = rep
        visualize(root, rep, out_dir)
        distribution_plots(rep, raw, out_dir)
        print_report_summary(rep)

    report["resolution_analysis"] = resolution_analysis(report["train"])

    out = out_dir / "dataset_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report -> {out}")

    print("\nRESOLUTION ANALYSIS")
    print(f"  native: {report['resolution_analysis']['native_size']}")
    for res, v in report["resolution_analysis"]["candidates"].items():
        print(f"  {res:>4}px scale={v['scale']:.3f} | typical max-dim {v['typical_max_dim_px']:>6.1f}px "
              f"| small(q05) max-dim {v['small_q05_max_dim_px']:>5.1f}px | typical area {v['typical_area_px']:>7.1f}px")


def print_report_summary(rep: dict) -> None:
    print(f"  images={rep['n_images']} masks={rep['n_masks']} mismatch={rep['mismatched_names'] or 'none'}")
    print(f"  image sizes={rep['image_sizes']} modes={rep['image_modes']}")
    print(f"  empty masks={rep['empty_masks']} ({rep['empty_mask_pct']}%)")
    print(f"  foreground%: median={rep['fg_pct']['median']} q95={rep['fg_pct']['q95']} max={rep['fg_pct']['max']}")
    ca = rep["comp_area"]
    if ca["count"]:
        print(f"  clover components={ca['count']}")
        print(f"    area px:  min={ca['min']} q05={ca['q05']} med={ca['median']} q95={ca['q95']} max={ca['max']}")
        cm = rep["comp_max_dim"]
        print(f"    max-dim:  min={cm['min']} q05={cm['q05']} med={cm['median']} q95={cm['q95']} max={cm['max']}")
        cr = rep["comp_rel_area"]
        print(f"    rel-area: med={cr['median']} q05={cr['q05']} max={cr['max']} (fraction of 800x600)")


if __name__ == "__main__":
    main()