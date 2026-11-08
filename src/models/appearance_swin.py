
"""Video Swin appearance branch for isolated sign language recognition."""

import torch
from torch import nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


class AppearanceSwin(nn.Module):
    """Extract a 1024-dimensional feature vector from a 16-frame RGB clip."""

    def __init__(self, pretrained: bool = True, feature_dim: int = 1024):
        super().__init__()

        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin3d_t(weights=weights)

        backbone_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.projection = nn.Linear(backbone_dim, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: RGB tensor shaped (B, T, C, H, W).

        Returns:
            Appearance features shaped (B, feature_dim).
        """
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames with 5 dimensions (B,T,C,H,W); "
                f"received shape {tuple(frames.shape)}."
            )

        if frames.shape[2] != 3:
            raise ValueError(
                f"Expected 3 RGB channels; received {frames.shape[2]}."
            )

        # torchvision Video Swin expects (B, C, T, H, W).
        video = frames.permute(0, 2, 1, 3, 4)

        features = self.backbone(video)
        features = self.projection(features)

        return features
