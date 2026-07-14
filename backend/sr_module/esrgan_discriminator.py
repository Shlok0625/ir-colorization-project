"""
ESRGAN Relativistic Discriminator — Stage 3
============================================
WHY RELATIVISTIC, NOT STANDARD GAN:
    Standard discriminator: outputs P(x is real).
    Relativistic discriminator (Ra): outputs P(real patch is MORE realistic
    than fake patch). This forces the generator to produce outputs that are
    not just "looks plausible" but "looks more realistic than the actual HR
    thermal patch" — which is a harder, more useful training signal.
    This is the key discriminator improvement ESRGAN makes over SRGAN.

INPUT:
    - HR thermal tiles: (B, 1, H, W) float32, H/W = 256 (or your HR tile size)
    - SR thermal tiles: (B, 1, H, W) float32, same size

OUTPUT:
    - Scalar logit per patch (before sigmoid — sigmoid is applied in the loss)
"""

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    use_bn: bool = True,
) -> nn.Sequential:
    layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ESRGANDiscriminator(nn.Module):
    """
    VGG-style PatchGAN discriminator.
    Progressively downsamples with stride-2 convolutions.
    Final output is a spatial map of logits (not a single scalar) —
    the relativistic loss averages across the spatial map.

    Input: (B, 1, 256, 256) single-channel thermal tile
    Feature progression: 1 -> 64 -> 64 -> 128 -> 128 -> 256 -> 256 -> 512 -> 512
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            # No BN on first layer (standard practice)
            _conv_block(in_channels, 64,  stride=1, use_bn=False),
            _conv_block(64,          64,  stride=2),
            _conv_block(64,          128, stride=1),
            _conv_block(128,         128, stride=2),
            _conv_block(128,         256, stride=1),
            _conv_block(256,         256, stride=2),
            _conv_block(256,         512, stride=1),
            _conv_block(512,         512, stride=2),
        )

        # After 4 stride-2 ops on 256x256 input: spatial size = 16x16
        # Dense layers convert to final logit map
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),   # -> (B, 512, 4, 4)
            nn.Flatten(),                    # -> (B, 8192)
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),              # single logit, no sigmoid (loss handles it)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)
