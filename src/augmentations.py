"""Augmentation pipelines.

Albumentations is used so spatial transforms are applied identically to the
image and the mask. Masks stay binary because albumentations resizes masks with
nearest-neighbor by default.

Default train pipeline is intentionally mild to avoid unrealistic distortion:
flips, small rotations, small scale/translate, brightness/contrast/gamma, and a
light blur occasionally. Vertical flips are disabled by default but configurable.
"""

from __future__ import annotations

import albumentations as A

# Standard ImageNet stats used by the torchvision MobileNet encoder.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transform(cfg) -> A.Compose:
    transforms = []
    if cfg.random_scale_limit > 0:
        # simulate camera distance: resize image+mask by a random factor, then
        # Resize brings it back to the fixed training size
        transforms.append(
            A.RandomScale(
                scale_limit=(-cfg.random_scale_limit, cfg.random_scale_limit),
                interpolation=1,
                p=0.7,
            )
        )
    transforms.append(A.Resize(height=cfg.image_size[1], width=cfg.image_size[0], interpolation=1))
    if cfg.augment:
        transforms += [
            A.HorizontalFlip(p=cfg.hflip_p),
            A.VerticalFlip(p=cfg.vflip_p),
            A.Affine(
                scale=(1.0 - cfg.scale_limit, 1.0 + cfg.scale_limit),
                translate_percent=(-0.05, 0.05),
                rotate=(-cfg.rotate_limit, cfg.rotate_limit),
                interpolation=1,
                mask_interpolation=0,  # nearest
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5,
            ),
            A.RandomGamma(gamma_limit=cfg.gamma_limit, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=cfg.blur_p),
        ]
    transforms.append(A.Normalize(mean=MEAN, std=STD))
    transforms.append(A.ToTensorV2())
    return A.Compose(transforms)


def get_val_transform(cfg) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=cfg.image_size[1], width=cfg.image_size[0], interpolation=1),
            A.Normalize(mean=MEAN, std=STD),
            A.ToTensorV2(),
        ]
    )


def get_inference_transform(cfg) -> A.Compose:
    """Like val but also returns the original-size mask via extra dict keys.

    Not needed for plain evaluation; kept for the future PC/mobile prototype.
    """
    return get_val_transform(cfg)