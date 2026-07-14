"""
ESRGAN Loss Functions — Stage 3
================================
Three named, distinct loss terms (not "some losses"):

1. L1 PIXEL LOSS
   Anchors SR output to the real 100m thermal ground truth.
   Prevents the generator from drifting into "looks sharp but wrong values."
   Weight: highest — reconstruction accuracy is the primary objective.

2. EDGE LOSS (Sobel + Laplacian)
   Directly optimizes for the brief's stated goal: "improve visibility of
   faint edges and textures." L1/MSE losses are spatially uniform and will
   under-penalize edge errors relative to flat-region errors (flat regions
   have more pixels). Edge loss corrects this by applying a gradient operator
   before computing the residual — so edge pixel errors are weighted higher.
   Uses both Sobel (directional edges) and Laplacian (second-order, catches
   texture), because they catch complementary structures.

3. VGG PERCEPTUAL LOSS
   Penalizes feature-level mismatch in a pretrained VGG-19 feature space.
   This is what makes SR outputs look sharp to a human rather than just
   having low MSE — it's the reason ESRGAN produces visually better results
   than SRCNN/EDSR on perceptual metrics.
   NOTE: VGG is pretrained on RGB images. We replicate the single thermal
   channel to 3 channels before passing through VGG. This is a known
   approximation used in satellite SR literature — document it.

4. RELATIVISTIC ADVERSARIAL LOSS
   Low weight. Sharpens local texture. Does NOT drive content decisions.
   The weighting enforces this — adversarial loss cannot override L1+Edge.

WHY NO MSE LOSS:
   MSE and L1 serve the same anchoring role. MSE produces over-smooth
   outputs (it minimizes the average, which blurs uncertainty). L1 produces
   sharper results empirically on SR tasks. Use L1, not MSE.

VGG loaded once at construction, frozen immediately.
Never re-instantiate per forward pass — on 6 GB VRAM this matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1Loss(nn.Module):
    """Pixel-level L1 reconstruction loss."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(sr, hr)


class EdgeLoss(nn.Module):
    """
    Combined Sobel + Laplacian edge loss.
    Extracts edges from both SR and HR, computes L1 between them.
    Kernels registered as buffers so they move to GPU with .to(device).
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels (horizontal + vertical)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x",   sobel_x)
        self.register_buffer("sobel_y",   sobel_y)
        self.register_buffer("laplacian", laplacian)

    def _extract_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sobel + Laplacian to a (B, 1, H, W) tensor."""
        ex = F.conv2d(x, self.sobel_x,   padding=1)
        ey = F.conv2d(x, self.sobel_y,   padding=1)
        el = F.conv2d(x, self.laplacian, padding=1)
        # Combine: gradient magnitude + laplacian response
        sobel_mag = torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)
        return sobel_mag + el.abs()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_edges = self._extract_edges(sr)
        hr_edges = self._extract_edges(hr)
        return F.l1_loss(sr_edges, hr_edges)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-19 feature maps.
    Uses relu3_4 features (layer index 18 in torchvision VGG-19).
    Loaded once, frozen completely. Never called on GPU unless .to(device)
    has been called on this module first.

    Input tensors: single-channel (B, 1, H, W).
    Internally replicated to 3-channel for VGG compatibility.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Features up to relu3_4 (index 18 inclusive)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:18])
        # Freeze: VGG is a fixed feature space, not trained
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Replicate single thermal channel to 3 channels for VGG input
        sr_rgb = sr.repeat(1, 3, 1, 1)
        hr_rgb = hr.repeat(1, 3, 1, 1)
        sr_feat = self.feature_extractor(sr_rgb)
        hr_feat = self.feature_extractor(hr_rgb)
        return F.l1_loss(sr_feat, hr_feat)


class RelativisticAdversarialLoss(nn.Module):
    """
    Relativistic average GAN loss (RaGAN).
    Generator loss: make SR look more realistic than HR on average.
    Discriminator loss: make HR look more realistic than SR on average.

    Both use BCE with logits. Sigmoid not applied in discriminator output —
    it's applied here via binary_cross_entropy_with_logits.

    Usage:
        loss_module = RelativisticAdversarialLoss()
        g_loss = loss_module.generator_loss(D, real_hr, fake_sr)
        d_loss = loss_module.discriminator_loss(D, real_hr, fake_sr)
    """

    def generator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        d_real = discriminator(real).detach()   # stop grad through D for G update
        d_fake = discriminator(fake)
        # Generator wants: D(fake) - mean(D(real)) > 0
        loss_rf = F.binary_cross_entropy_with_logits(
            d_real - d_fake.mean(), torch.zeros_like(d_real)
        )
        loss_fr = F.binary_cross_entropy_with_logits(
            d_fake - d_real.mean(), torch.ones_like(d_fake)
        )
        return (loss_rf + loss_fr) / 2

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        d_real = discriminator(real)
        d_fake = discriminator(fake).detach()   # stop grad through G for D update
        # Discriminator wants: D(real) - mean(D(fake)) > 0
        loss_rf = F.binary_cross_entropy_with_logits(
            d_real - d_fake.mean(), torch.ones_like(d_real)
        )
        loss_fr = F.binary_cross_entropy_with_logits(
            d_fake - d_real.mean(), torch.zeros_like(d_fake)
        )
        return (loss_rf + loss_fr) / 2


class ESRGANCombinedLoss(nn.Module):
    """
    Weighted combination of all four loss terms for the generator.
    Weights are the actual design decision — not arbitrary.

    λ_l1=1.0, λ_edge=0.1, λ_perceptual=0.1, λ_adversarial=0.005

    Why these weights:
    - L1 dominates (1.0): reconstruction accuracy is the primary goal.
      If L1 were lower, the model could produce "realistic-looking" SR tiles
      that are wrong in absolute thermal value — unacceptable for satellite data.
    - Edge (0.1): meaningful but secondary. We want better edges, not
      edges at the cost of overall fidelity.
    - Perceptual (0.1): same rationale as edge — improves perceived quality
      without overriding ground-truth anchoring.
    - Adversarial (0.005): deliberately tiny. Enough to provide texture
      sharpening signal; not enough to cause hallucination or training
      instability. This is the anti-hallucination choice, not a mistake.

    State these λ values explicitly in any report or presentation.
    """

    def __init__(
        self,
        lambda_l1:          float = 1.0,
        lambda_edge:        float = 0.1,
        lambda_perceptual:  float = 0.1,
        lambda_adversarial: float = 0.005,
    ):
        super().__init__()
        self.lambda_l1          = lambda_l1
        self.lambda_edge        = lambda_edge
        self.lambda_perceptual  = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial

        self.l1_loss          = L1Loss()
        self.edge_loss        = EdgeLoss()
        self.perceptual_loss  = VGGPerceptualLoss()
        self.adversarial_loss = RelativisticAdversarialLoss()

    def forward(
        self,
        discriminator: nn.Module,
        sr:            torch.Tensor,
        hr:            torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        l1  = self.l1_loss(sr, hr)
        edg = self.edge_loss(sr, hr)
        per = self.perceptual_loss(sr, hr)
        adv = self.adversarial_loss.generator_loss(discriminator, hr, sr)

        total = (
            self.lambda_l1          * l1
            + self.lambda_edge        * edg
            + self.lambda_perceptual  * per
            + self.lambda_adversarial * adv
        )

        # Return individual losses for logging — you need these to diagnose
        # training problems. A single "loss=X" number tells you nothing when
        # GAN training goes wrong.
        breakdown = {
            "l1":          l1.item(),
            "edge":        edg.item(),
            "perceptual":  per.item(),
            "adversarial": adv.item(),
            "total":       total.item(),
        }
        return total, breakdown
