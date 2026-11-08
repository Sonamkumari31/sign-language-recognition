"""Run appearance, pose, and fusion inference."""

import torch
import torch.nn.functional as F

from src.inference.model_loader import load_one_model


def predict_one(name, frames, lti, device):
    """Run one model and return its class probabilities."""

    model = load_one_model(name, device)

    frames = frames.to(device)
    lti = lti.to(device)

    try:
        with torch.inference_mode():
            if name == "appearance":
                output = model(frames)
            elif name == "pose":
                output = model(lti)
            else:
                output = model(frames, lti)

            probabilities = F.softmax(output, dim=1)

        return probabilities.cpu().squeeze(0)

    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def predict_all(frames, lti, device=None):
    """Run all three models sequentially."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    results = {}

    for name in ("appearance", "pose", "fusion"):
        results[name] = predict_one(
            name,
            frames,
            lti,
            device,
        )

    return results