
"""Two-stream Swin architecture for isolated ASL recognition."""

import torch
from torch import nn

from src.models.appearance_swin import AppearanceSwin
from src.models.pose_swin import PoseSwin
from src.models.fusion_transformer import FusionTransformer


class TwoStreamSwin(nn.Module):
    """Video Swin + image Swin, followed by Transformer fusion."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbones: bool = True,
    ):
        super().__init__()

        self.appearance_swin = AppearanceSwin(
            pretrained=pretrained,
            feature_dim=1024,
        )

        self.pose_swin = PoseSwin(
            pretrained=pretrained,
            feature_dim=1024,
        )

        self.fusion = FusionTransformer(
            num_classes=num_classes,
            feature_dim=1024,
        )

        if freeze_backbones:
            # Freeze pretrained backbones, not their projection layers.
            for parameter in self.appearance_swin.backbone.parameters():
                parameter.requires_grad = False

            for parameter in self.pose_swin.backbone.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        frames: torch.Tensor,
        lti: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, 16, 3, 224, 224)
            lti:    (B, 3, 224, 224)

        Returns:
            Class logits shaped (B, num_classes).
        """
        appearance_features = self.appearance_swin(frames)
        pose_features = self.pose_swin(lti)

        logits = self.fusion(
            appearance_features,
            pose_features,
        )

        return logits
