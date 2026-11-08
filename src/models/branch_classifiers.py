
"""Standalone classifiers for appearance and pose branch training."""

import torch
from torch import nn

from src.models.appearance_swin import AppearanceSwin
from src.models.pose_swin import PoseSwin


class AppearanceClassifier(nn.Module):
    """Classify signs using the video appearance branch."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.branch = AppearanceSwin(
            pretrained=pretrained,
            feature_dim=1024,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

        if freeze_backbone:
            for parameter in self.branch.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.branch(frames)
        return self.classifier(features)


class PoseClassifier(nn.Module):
    """Classify signs using the pose LTI image branch."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.branch = PoseSwin(
            pretrained=pretrained,
            feature_dim=1024,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

        if freeze_backbone:
            for parameter in self.branch.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, lti: torch.Tensor) -> torch.Tensor:
        features = self.branch(lti)
        return self.classifier(features)
