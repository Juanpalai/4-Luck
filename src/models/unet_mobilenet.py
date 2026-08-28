"""Lightweight U-Net for four-leaf clover segmentation.

Encoder: torchvision MobileNetV3-Small (pretrained) — 2.5M params, ideal for
mobile deployment. Decoder: standard U-Net up-blocks fused to the encoder's
multi-scale feature maps via skip connections.

Encoder feature layout (verified against torchvision source):
    block 0  -> 16ch @1/2      (skip)
    block 1  -> 16ch @1/4      (skip)
    block 2  -> 24ch @1/8      (skip)
    block 8  -> 48ch @1/16     (skip)
    block 9  -> 96ch @1/32     (stride-2 block)
    block 12 -> 576ch @1/32    (bottleneck)

output_stride=32: full encoder, bottleneck 576ch @1/32.
output_stride=16: encoder truncated after block 8, bottleneck 48ch @1/16.
Halving the downsampling keeps small targets larger in feature space, at a
small cost for large objects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# (feature-block index, channel count) at each skip level, shallow-to-deep.
SKIP_STRIDE32 = [(0, 16), (1, 16), (2, 24), (8, 48)]
SKIP_STRIDE16 = [(0, 16), (1, 16), (2, 24)]


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, norm: bool = True):
        pad = (kernel - 1) // 2
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class UpBlock(nn.Module):
    """2x upsample + concat skip + 2 convs."""

    def __init__(self, in_ch: int, skip_ch: int | None, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + (skip_ch or 0), out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.conv2(x)


class LightweightUNet(nn.Module):
    def __init__(self, encoder: str = "mobilenet_v3_small", encoder_pretrained: bool = True,
                 decoder_channels: int = 128, output_stride: int = 32, num_classes: int = 1):
        super().__init__()
        assert encoder == "mobilenet_v3_small", "only mobilenet_v3_small encoder is supported"
        assert output_stride in (16, 32), "output_stride must be 16 or 32"
        self.output_stride = output_stride

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if encoder_pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        if output_stride == 16:
            # blocks 0..8 -> 48ch @1/16 bottleneck
            self.encoder = nn.Sequential(*list(backbone.features[:9]))
            skip_spec = SKIP_STRIDE16
            bottleneck_ch = 48
            skips = [c for _, c in skip_spec]  # [16, 24, 40]
        else:
            # full encoder -> 576ch @1/32 bottleneck
            self.encoder = backbone.features
            skip_spec = SKIP_STRIDE32
            bottleneck_ch = 576
            skips = [c for _, c in skip_spec]  # [16, 24, 40, 48]

        self.num_classes = num_classes

        # Decoder, deepest-first: each UpBlock upsamples 2x then fuses the skip
        # feature that lives at the resulting resolution.
        #   stride32: 1/32->1/16(+48) ->1/8(+24) ->1/4(+16) ->1/2(+16) ->1/1
        #   stride16: 1/16->1/8(+24) ->1/4(+16) ->1/2(+16) ->1/1
        skip_chs = [c for _, c in skip_spec]  # shallow-to-deep
        fuse_skips = list(reversed(skip_chs))  # deepest first
        self.decoder = nn.ModuleList(
            [
                UpBlock(bottleneck_ch if i == 0 else decoder_channels, fuse_skips[i], decoder_channels)
                for i in range(len(fuse_skips))
            ]
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_blocks = {0, 1, 2} if self.output_stride == 16 else {0, 1, 2, 8}
        feats: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in skip_blocks:
                feats.append(x)
        feats = list(reversed(feats))  # deepest first
        for i, up in enumerate(self.decoder):
            skip = feats[i] if i < len(feats) else None
            x = up(x, skip)
        return self.final(x)  # last upsample -> full resolution

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg) -> LightweightUNet:
    return LightweightUNet(
        encoder=cfg.encoder,
        encoder_pretrained=cfg.encoder_pretrained,
        decoder_channels=cfg.decoder_channels,
        output_stride=cfg.output_stride,
    )