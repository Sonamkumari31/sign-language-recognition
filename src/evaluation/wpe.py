
"""Weighted Prediction Ensemble (WPE) for three SLR branches.

Inputs are classification logits from:
    1. Appearance model
    2. Pose model
    3. Fusion model

Weights must be selected using validation data only.
"""

from itertools import product
from pathlib import Path
import json

import torch
import torch.nn.functional as F


BRANCH_NAMES = ("appearance", "pose", "fusion")


def weighted_prediction_ensemble(
    appearance_logits,
    pose_logits,
    fusion_logits,
    weights=(0.30, 0.30, 0.40),
):
    """Combine branch probabilities using a weighted sum.

    Each logits tensor must have shape [batch_size, num_classes].
    All branches must use the same class ordering.
    """
    if len(weights) != 3:
        raise ValueError("WPE requires exactly three weights.")

    if any(weight < 0 for weight in weights):
        raise ValueError("WPE weights cannot be negative.")

    if sum(weights) <= 0:
        raise ValueError("At least one WPE weight must be positive.")

    logits = (appearance_logits, pose_logits, fusion_logits)

    if any(x.ndim != 2 for x in logits):
        raise ValueError("Every logits tensor must have shape [B, C].")

    if not (
        appearance_logits.shape
        == pose_logits.shape
        == fusion_logits.shape
    ):
        raise ValueError("All branches must have matching [B, C] shapes.")

    device = appearance_logits.device
    pose_logits = pose_logits.to(device)
    fusion_logits = fusion_logits.to(device)

    probabilities = [
        F.softmax(x, dim=1)
        for x in (appearance_logits, pose_logits, fusion_logits)
    ]

    normalized_weights = [
        float(weight) / sum(weights) for weight in weights
    ]

    combined_probabilities = sum(
        weight * probability
        for weight, probability in zip(
            normalized_weights, probabilities
        )
    )

    predictions = combined_probabilities.argmax(dim=1)

    return predictions, combined_probabilities


def tune_wpe_weights(
    appearance_logits,
    pose_logits,
    fusion_logits,
    labels,
    step=0.1,
):
    """Select weights that maximize validation top-1 accuracy.

    Do not tune weights on the test set.
    Searches weights on a grid whose values sum to 1.
    """
    if step <= 0 or step > 1:
        raise ValueError("step must be in the interval (0, 1].")

    probabilities = [
        F.softmax(x.detach().float(), dim=1)
        for x in (
            appearance_logits,
            pose_logits,
            fusion_logits,
        )
    ]

    labels = labels.detach().long().to(probabilities[0].device)

    if any(p.shape != probabilities[0].shape for p in probabilities):
        raise ValueError("All validation logits must have matching shapes.")

    if labels.numel() != probabilities[0].shape[0]:
        raise ValueError("Number of labels must match the batch size.")

    increments = round(1.0 / step)
    if abs(increments * step - 1.0) > 1e-7:
        raise ValueError("Choose a step that divides 1 evenly, e.g. 0.1.")

    best_weights = None
    best_accuracy = -1.0

    for a_units in range(increments + 1):
        for p_units in range(increments - a_units + 1):
            f_units = increments - a_units - p_units

            weights = (
                a_units / increments,
                p_units / increments,
                f_units / increments,
            )

            combined = sum(
                w * p for w, p in zip(weights, probabilities)
            )
            predictions = combined.argmax(dim=1)
            accuracy = (
                predictions == labels
            ).float().mean().item()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights

    return {
        "weights": dict(zip(BRANCH_NAMES, best_weights)),
        "validation_top1": best_accuracy,
        "selection": "grid search on validation set",
        "step": step,
    }


def save_wpe_config(result, output_path):
    """Save the selected validation weights as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return output_path
