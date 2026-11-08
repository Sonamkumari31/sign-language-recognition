
"""Image Swin pose branch for isolated sign language recognition."""

import torch
from torch import nn
import timm


class PoseSwin(nn.Module):
    """Extract a 1024-dimensional feature vector from an LTI image."""

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1024,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )

        backbone_dim = self.backbone.num_features
        self.projection = nn.Linear(backbone_dim, feature_dim)

    def forward(self, lti: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lti: LTI image tensor shaped (B, 3, 224, 224).

        Returns:
            Pose features shaped (B, feature_dim).
        """
        if lti.ndim != 4:
            raise ValueError(
                f"Expected LTI with 4 dimensions (B,C,H,W); "
                f"received shape {tuple(lti.shape)}."
            )

        if lti.shape[1] != 3:
            raise ValueError(
                f"Expected 3 LTI channels (x, y, z); "
                f"received {lti.shape[1]}."
            )

        features = self.backbone(lti)
        features = self.projection(features)

        return features
