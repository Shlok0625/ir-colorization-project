"""
ESRGAN Generator — Stage 3: Super-Resolution Module
=====================================================
INPUT CONTRACT (from Stage 2 synthetic_degradation.py):
    - Single-channel thermal tiles at 200m effective resolution
    - Stored as .npy or .tif files, float32, normalized to [0, 1]
    - Tile size: 64x64 (the 200m version of a 256x256 real-world patch,
      accounting for the ~4x pixel ratio when going 200m -> 50m display)
    - Actually: if Stage 2 produces 256x256 tiles at 100m and degrades to
      128x128 at 200m, the SR target is 256x256. Coordinate with Stage 2
      teammate on exact tile dimensions — this file uses:
          LR (input):  128x128 single-channel float32
          HR (target): 256x256 single-channel float32
      If your teammate uses different sizes, change SCALE_FACTOR and
      tile dims in esrgan_train.py only — this file is scale-agnostic.

OUTPUT CONTRACT (to esrgan_infer.py -> Stage 5):
    - Single-channel thermal tiles, float32, [0, 1]
    - Spatial size: 2x the input (128->256, or whatever your scale factor is)
    - Saved as checkpoint: sr_module/checkpoints/esrgan_generator.pth

ARCHITECTURE DECISIONS (documented, not arbitrary):
    - 16 RRDB blocks, not 23 (original paper): 23 blocks OOMs on 6 GB VRAM
      at 256x256 patches. 16 is what Real-ESRGAN uses for practical
      deployment. This is the honest choice for the hardware available.
    - PixelShuffle upsampling, not transposed convolution: avoids
      checkerboard artifacts in the upsampled thermal output.
    - Single-channel I/O throughout: thermal is monochrome. Do not add
      channels here; that is Stage 5's job.

PyTorch 2.x, CUDA 12.x, 6 GB VRAM.
"""

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    5-layer dense block used inside each RRDB.
    Each layer takes concatenation of all previous feature maps as input
    (densely connected), which is the key difference from plain residual blocks.
    growth_channels: how many new channels each conv layer adds.
    """

    def __init__(self, num_features: int = 64, growth_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features + 0 * growth_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + 1 * growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_channels, num_features,   3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Residual scaling: prevents training instability in deep dense networks.
        # 0.2 is the value used in the original ESRGAN paper.
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        # Last layer: no activation (outputs residual to add back to input)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.res_scale + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.
    Three DenseBlocks with a residual skip around all three.
    This is the core building unit of ESRGAN.
    """

    def __init__(self, num_features: int = 64, growth_channels: int = 32):
        super().__init__()
        self.db1 = DenseBlock(num_features, growth_channels)
        self.db2 = DenseBlock(num_features, growth_channels)
        self.db3 = DenseBlock(num_features, growth_channels)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * self.res_scale + x


class ESRGANGenerator(nn.Module):
    """
    Full ESRGAN generator.

    Args:
        in_channels:     1 for single-channel thermal input
        out_channels:    1 for single-channel thermal output
        num_features:    internal feature map width (64 for 6 GB VRAM)
        num_rrdb:        number of RRDB blocks (16, not 23 — see file header)
        scale_factor:    upscaling factor. Must be a power of 2 (2 or 4).
                         Use 2 if your LR tiles are 128x128 -> 256x256.
                         Use 4 if your LR tiles are  64x64  -> 256x256.
                         Coordinate with Stage 2.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_rrdb: int = 16,
        scale_factor: int = 2,
    ):
        super().__init__()

        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")

        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # RRDB trunk
        trunk = [RRDB(num_features) for _ in range(num_rrdb)]
        self.trunk = nn.Sequential(*trunk)
        self.conv_trunk_end = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Upsampling via PixelShuffle (avoids checkerboard artifacts)
        upsample_layers = []
        for _ in range(scale_factor // 2):   # one block per 2x step
            upsample_layers += [
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),           # 4*C channels -> C channels, 2x spatial
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        # Output refinement
        self.conv_hr   = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat       = self.conv_first(x)
        trunk_feat = self.conv_trunk_end(self.trunk(feat))
        feat       = feat + trunk_feat                      # long residual skip
        feat       = self.upsample(feat)
        feat       = self.lrelu(self.conv_hr(feat))
        out        = self.conv_last(feat)
        # Clamp to [0,1]: thermal values are normalized, output must stay valid.
        # Do NOT use sigmoid here — it flattens gradients. Clamp at inference,
        # let the loss drive it during training.
        return out

    def clamp_output(self, x: torch.Tensor) -> torch.Tensor:
        """Call this at inference time, not during training."""
        return x.clamp(0.0, 1.0)
