"""Utilities for formatting ASL predictions."""


def get_top_predictions(probabilities, label_map, top_k=5):
    """Return the top-k gloss predictions with confidence scores."""

    if probabilities.ndim != 1:
        raise ValueError(
            f"Expected 1D probabilities, got {probabilities.shape}"
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    k = min(top_k, probabilities.numel())

    scores, indices = probabilities.topk(k)

    reverse_map = {
        int(class_id): gloss
        for gloss, class_id in label_map.items()
    }

    predictions = []

    for score, index in zip(scores.tolist(), indices.tolist()):
        predictions.append({
            "gloss": reverse_map.get(index, f"UNKNOWN_{index}"),
            "class_id": index,
            "confidence": float(score),
            "confidence_percent": float(score * 100),
        })

    return predictions


def format_predictions(results, label_map, top_k=5):
    """Format predictions from all model branches."""

    return {
        model_name: get_top_predictions(
            probabilities,
            label_map,
            top_k,
        )
        for model_name, probabilities in results.items()
    }