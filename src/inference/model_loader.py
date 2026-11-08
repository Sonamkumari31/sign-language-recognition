"""Load trained ASL100 inference models."""

from pathlib import Path

import torch

from src.models.branch_classifiers import (
    AppearanceClassifier,
    PoseClassifier,
)
from src.models.two_stream_swin import TwoStreamSwin


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = {
    "appearance": ROOT / "experiments/aslcitizen100/appearance_full_v1/best.pt",
    "pose": ROOT / "experiments/aslcitizen100/pose_full/best.pt",
    "fusion": ROOT / "experiments/aslcitizen100/fusion_full_v1/best.pt",
}


def get_state_dict(checkpoint):
    """Extract model weights from supported checkpoint formats."""

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def load_one_model(name, device):
    """Load one trained model and its checkpoint."""

    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown model: {name}")

    checkpoint_path = CHECKPOINTS[name]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if name == "appearance":
        model = AppearanceClassifier(
            num_classes=100,
            pretrained=False,
            freeze_backbone=False,
        )
    elif name == "pose":
        model = PoseClassifier(
            num_classes=100,
            pretrained=False,
            freeze_backbone=False,
        )
    else:
        model = TwoStreamSwin(
            num_classes=100,
            pretrained=False,
            freeze_backbones=True,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = get_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    del checkpoint

    return model


def load_all_models(device=None):
    """Load all three trained models."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    models = {}

    for name in ("appearance", "pose", "fusion"):
        models[name] = load_one_model(name, device)
        print(f"Loaded {name} model successfully.")

    return models