#!/usr/bin/env python3
"""
Shared MobileNetV2-style U-Net built with plain PyTorch (no torchvision).

Input : (B, 3, 128, 128) BGR float32 normalized to [-1, 1]
Output: (B, 1, 128, 128) logits (apply sigmoid for probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """MobileNetV2 bottleneck block: expand -> depthwise -> project."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2-style encoder for 128x128 input.

    Returns (skips, bottleneck) where skips is a list of 4 feature maps
    at 1/2, 1/4, 1/8, 1/16 resolution (channels 16, 24, 32, 64) and
    bottleneck is 4x4x320.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # (in, out, stride, expand_ratio)
        spec = [
            (32, 16, 1, 1),    # 64x64  -> skip 0
            (16, 24, 2, 6),    # 32x32
            (24, 24, 1, 6),    # 32x32  -> skip 1
            (24, 32, 2, 6),    # 16x16
            (32, 32, 1, 6),    # 16x16
            (32, 32, 1, 6),    # 16x16  -> skip 2
            (32, 64, 2, 6),    # 8x8
            (64, 64, 1, 6),    # 8x8
            (64, 64, 1, 6),    # 8x8
            (64, 64, 1, 6),    # 8x8    -> skip 3
            (64, 96, 1, 6),    # 8x8
            (96, 96, 1, 6),    # 8x8
            (96, 96, 1, 6),    # 8x8
            (96, 160, 2, 6),   # 4x4
            (160, 160, 1, 6),  # 4x4
            (160, 160, 1, 6),  # 4x4
            (160, 320, 1, 6),  # 4x4
        ]
        self.blocks = nn.ModuleList([InvertedResidual(*s) for s in spec])
        self.head = nn.Sequential(
            nn.Conv2d(320, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
        )
        self.skip_indices = {0, 2, 5, 9}

    def forward(self, x):
        skips = []
        x = self.stem(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.skip_indices:
                skips.append(x)
        x = self.head(x)
        return skips, x


class DecoderBlock(nn.Module):
    """Upsample -> concat skip -> 1x1 project -> two 3x3 convs."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.conv(x)


class BloodUNet(nn.Module):
    """MobileNetV2-U-Net. Output is 1-channel logits (no sigmoid)."""

    def __init__(self):
        super().__init__()
        self.encoder = MobileNetV2Encoder()
        # Skip channels (from 8x8, 16x16, 32x32, 64x64): 64, 32, 24, 16
        self.dec0 = DecoderBlock(320 + 64, 64)
        self.dec1 = DecoderBlock(64 + 32, 32)
        self.dec2 = DecoderBlock(32 + 24, 16)
        self.dec3 = DecoderBlock(16 + 16, 8)
        self.final = nn.Sequential(
            nn.Conv2d(8 + 3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        skips, bottleneck = self.encoder(x)
        s0, s1, s2, s3 = skips  # 64x64x16, 32x32x24, 16x16x32, 8x8x64
        d = self.dec0(bottleneck, s3)  # 8x8x64
        d = self.dec1(d, s2)           # 16x16x32
        d = self.dec2(d, s1)           # 32x32x24
        d = self.dec3(d, s0)           # 64x64x16
        d = F.interpolate(d, size=x.shape[2:], mode="bilinear",
                          align_corners=False)
        d = torch.cat([d, x], dim=1)
        return self.final(d)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)