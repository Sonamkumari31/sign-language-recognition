
"""Transformer fusion head for two-stream sign language recognition."""

import torch
from torch import nn


class FusionTransformer(nn.Module):
    """Fuse appearance and pose features for sign classification."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        fused_dim = feature_dim * 2
        token_dim = 32

        if fused_dim % token_dim != 0:
            raise ValueError(
                f"Fused dimension {fused_dim} must be divisible "
                f"by token dimension {token_dim}."
            )

        self.fused_dim = fused_dim
        self.token_dim = token_dim

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, appearance_features: torch.Tensor,
                pose_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            appearance_features: (B, feature_dim)
            pose_features: (B, feature_dim)

        Returns:
            Class logits shaped (B, num_classes).
        """
        if appearance_features.ndim != 2 or pose_features.ndim != 2:
            raise ValueError(
                "Appearance and pose features must both have shape (B, D)."
            )

        if appearance_features.shape != pose_features.shape:
            raise ValueError(
                "Appearance and pose feature shapes must match; "
                f"received {tuple(appearance_features.shape)} and "
                f"{tuple(pose_features.shape)}."
            )

        fused = torch.cat(
            [appearance_features, pose_features],
            dim=1,
        )

        tokens = fused.reshape(
            fused.shape[0],
            self.fused_dim // self.token_dim,
            self.token_dim,
        )
        tokens = self.encoder(tokens)

        fused_features = tokens.reshape(tokens.shape[0], self.fused_dim)
        return self.classifier(fused_features)
