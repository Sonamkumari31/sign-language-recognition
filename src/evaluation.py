
"""Evaluation utilities for isolated-word sign language recognition."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(labels, predictions, num_classes):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)

    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Labels/predictions are empty or have different lengths.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Ground-truth label outside class range.")
    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError("Predicted label outside class range.")

    class_ids = np.arange(num_classes)
    mp, mr, mf, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_ids,
        average="macro", zero_division=0,
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_ids,
        average="weighted", zero_division=0,
    )

    top1 = float(accuracy_score(labels, predictions))
    return {
        "samples": int(len(labels)),
        "top1_accuracy": top1,
        "accuracy": top1,
        "wer_single_word_equivalent": 1.0 - top1,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf),
    }


def calculate_topk_accuracy(logits, labels, ks=(1, 5)):
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes].")
    labels = labels.reshape(-1).long()
    if logits.shape[0] != labels.numel():
        raise ValueError("Logits and labels have different batch sizes.")

    max_k = min(max(ks), logits.shape[1])
    indices = logits.topk(max_k, dim=1).indices
    matches = indices.eq(labels.unsqueeze(1))

    return {
        f"top{k}_accuracy": matches[:, :min(k, logits.shape[1])]
        .any(dim=1).float().mean().item()
        for k in ks
    }


def calculate_wer(reference_words, predicted_words):
    """Sequence WER using Levenshtein edit distance."""
    if len(reference_words) != len(predicted_words):
        raise ValueError("Reference and prediction collections must match.")

    errors = 0
    reference_count = 0

    for reference, prediction in zip(reference_words, predicted_words):
        ref, hyp = list(reference), list(prediction)
        reference_count += len(ref)
        previous = list(range(len(hyp) + 1))

        for i, word in enumerate(ref, start=1):
            current = [i]
            for j, predicted_word in enumerate(hyp, start=1):
                current.append(min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (word != predicted_word),
                ))
            previous = current
        errors += previous[-1]

    if reference_count == 0:
        raise ValueError("WER is undefined for empty references.")
    return errors / reference_count


def print_metrics_box(title, metrics):
    """Print a boxed, consistent evaluation summary."""
    fields = [
        ("Samples", "samples", ".0f"),
        ("Loss", "loss", ".4f"),
        ("Top-1 accuracy", "top1_accuracy", ".2%"),
        ("Top-5 accuracy", "top5_accuracy", ".2%"),
        ("WER (single-word equivalent)", "wer_single_word_equivalent", ".2%"),
        ("Balanced accuracy", "balanced_accuracy", ".2%"),
        ("Macro precision", "macro_precision", ".4f"),
        ("Macro recall", "macro_recall", ".4f"),
        ("Macro F1", "macro_f1", ".4f"),
        ("Weighted precision", "weighted_precision", ".4f"),
        ("Weighted recall", "weighted_recall", ".4f"),
        ("Weighted F1", "weighted_f1", ".4f"),
    ]

    rows = [
        (label, format(float(metrics[key]), fmt))
        for label, key, fmt in fields
        if key in metrics and metrics[key] is not None
    ]
    width = max([len(title)] + [len(a) + len(b) + 3 for a, b in rows])
    border = "+" + "-" * (width + 2) + "+"

    print(border)
    print("| " + title.ljust(width) + " |")
    print(border)
    for label, value in rows:
        print("| " + f"{label}: {value}".ljust(width) + " |")
    print(border)


def save_metrics_json(metrics, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=_json_default)
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_classification_report(
    output_dir, split, epoch, labels, predictions, num_classes
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        labels, predictions,
        labels=list(range(num_classes)),
        output_dict=True, zero_division=0,
    )
    path = output_dir / f"{split}_per_class_epoch_{epoch:03d}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_id", "precision", "recall", "f1-score", "support"],
        )
        writer.writeheader()
        for class_id in range(num_classes):
            row = report[str(class_id)]
            writer.writerow({
                "class_id": class_id,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1-score": row["f1-score"],
                "support": row["support"],
            })
    return path


def save_confusion_matrix(
    output_dir, split, epoch, labels, predictions, num_classes
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(num_classes)
    )
    stem = output_dir / f"{split}_confusion_epoch_{epoch:03d}"
    np.save(str(stem) + ".npy", matrix)
    np.savetxt(str(stem) + ".csv", matrix, delimiter=",", fmt="%d")
    return matrix


def save_learning_curves(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    plots = [
        ("loss.png", "Loss", ["train_loss", "val_loss"]),
        ("accuracy.png", "Accuracy", [
            "train_top1", "val_top1", "train_top5", "val_top5"
        ]),
        ("macro_f1.png", "Macro F1", ["train_macro_f1", "val_macro_f1"]),
    ]

    for filename, ylabel, keys in plots:
        plt.figure(figsize=(9, 5))
        for key in keys:
            values = [row.get(key, float("nan")) for row in history]
            plt.plot(epochs, values, marker="o", label=key)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def measure_inference_latency(
    model, example_inputs, device, warmup=10, iterations=50
):
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    model.eval()
    example_inputs = example_inputs.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(example_inputs)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(iterations):
            model(example_inputs)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    batch_size = example_inputs.shape[0]
    return {
        "total_seconds": elapsed,
        "iterations": iterations,
        "batch_size": batch_size,
        "latency_ms_per_batch": elapsed * 1000 / iterations,
        "latency_ms_per_sample": elapsed * 1000 / (iterations * batch_size),
    }
