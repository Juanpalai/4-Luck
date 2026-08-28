"""Segmentation evaluation metrics: IoU, Dice, Precision, Recall, F1.

Metrics are computed per image over binarized predictions at a fixed threshold,
then macro-averaged. ``confusion_matrix`` keeps per-class TP/FP/FN counts.
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred_logits: torch.Tensor, target: torch.Tensor) -> None:
        """pred_logits: (B,1,H,W); target: (B,1,H,W) or (B,H,W) in {0,1}."""
        pred = (torch.sigmoid(pred_logits) > self.threshold)
        if pred.ndim == 4:
            pred = pred[:, 0]
        pred = pred.reshape(-1)
        target = (target.reshape(-1) > 0.5)
        self.tp += (pred & target).sum().item()
        self.fp += (pred & ~target).sum().item()
        self.fn += (~pred & target).sum().item()

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def dice(self) -> float:
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn) if (2 * self.tp + self.fp + self.fn) > 0 else 0.0

    def summary(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "dice": round(self.dice, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
        }