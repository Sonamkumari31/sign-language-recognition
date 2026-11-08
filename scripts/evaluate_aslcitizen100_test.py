
"""Held-out test evaluation for ASL Citizen100.

Evaluates appearance, pose, and fusion checkpoints on the same paired
test samples. Inference only: no training or checkpoint modification.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.data.aslcitizen_dataset import ASLCitizenDataset

from src.inference.model_loader import (
    CHECKPOINTS,
    load_one_model,
)

from src.evaluation import (
    calculate_metrics,
    print_metrics_box,
    save_classification_report,
    save_confusion_matrix,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_CLASSES = 100
BATCH_SIZE = 1
EXPECTED_TEST_SAMPLES = 1283

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "aslcitizen100"
    / "held_out_test_20261105"
)

CHECKPOINT_EPOCHS = {
    "appearance": 8,
    "pose": 23,
    "fusion": 3,
}


# ---------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------

def evaluate_model(name, dataset, device, output_dir):
    """Evaluate one model on the common paired test dataset."""

    print("\n" + "=" * 65)
    print(f"Evaluating: {name.upper()}")
    print("=" * 65)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = load_one_model(name, device)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    all_labels = []
    all_predictions = []
    prediction_rows = []

    total_loss = 0.0
    total_top5_correct = 0
    total_samples = 0

    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():

            for batch_index, batch in enumerate(loader, start=1):

                frames = batch["frames"].to(
                    device,
                    non_blocking=True,
                )

                lti = batch["lti"].to(
                    device,
                    non_blocking=True,
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True,
                ).long()

                # Forward pass
                if name == "appearance":
                    logits = model(frames)

                elif name == "pose":
                    logits = model(lti)

                elif name == "fusion":
                    logits = model(frames, lti)

                else:
                    raise ValueError(f"Unknown model: {name}")

                # Validate output
                if (
                    logits.ndim != 2
                    or logits.shape[1] != NUM_CLASSES
                ):
                    raise ValueError(
                        f"{name} returned unexpected logits shape: "
                        f"{tuple(logits.shape)}"
                    )

                # Loss
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Top-1 and Top-5 predictions
                probabilities = torch.softmax(logits, dim=1)

                top5 = probabilities.topk(5, dim=1)
                predictions = logits.argmax(dim=1)

                total_top5_correct += (
                    top5.indices.eq(labels.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                batch_labels = labels.cpu().numpy().tolist()

                batch_predictions = (
                    predictions.cpu().numpy().tolist()
                )

                video_names = batch["video_name"]

                all_labels.extend(batch_labels)
                all_predictions.extend(batch_predictions)

                # Save individual predictions
                for i, (true_id, pred_id) in enumerate(
                    zip(batch_labels, batch_predictions)
                ):

                    top5_ids = top5.indices[i].cpu().tolist()
                    top5_conf = top5.values[i].cpu().tolist()

                    prediction_rows.append({
                        "video_name": video_names[i],
                        "true_class_id": true_id,
                        "predicted_class_id": pred_id,
                        "correct": int(true_id == pred_id),
                        "top5_class_ids": ";".join(
                            map(str, top5_ids)
                        ),
                        "top5_confidences": ";".join(
                            f"{value:.6f}" for value in top5_conf
                        ),
                    })

                total_samples += labels.size(0)

                if batch_index % 100 == 0:
                    print(
                        f"Processed {total_samples}/"
                        f"{len(dataset)} samples"
                    )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time

        # Convert results to arrays
        labels_array = np.asarray(
            all_labels,
            dtype=np.int64,
        )

        predictions_array = np.asarray(
            all_predictions,
            dtype=np.int64,
        )

        if total_samples != len(dataset):
            raise RuntimeError(
                f"Expected {len(dataset)} predictions, "
                f"but obtained {total_samples}."
            )

        # Calculate classification metrics
        metrics = calculate_metrics(
            labels_array,
            predictions_array,
            NUM_CLASSES,
        )

        metrics["loss"] = total_loss / total_samples

        metrics["top5_accuracy"] = (
            total_top5_correct / total_samples
        )

        metrics["evaluation_elapsed_seconds"] = elapsed

        metrics["evaluation_samples_per_second"] = (
            total_samples / elapsed
            if elapsed > 0
            else None
        )

        # Create output directory
        model_output = output_dir / name

        model_output.mkdir(
            parents=True,
            exist_ok=False,
        )

        # Save predictions
        with (
            model_output / "predictions.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "video_name",
                    "true_class_id",
                    "predicted_class_id",
                    "correct",
                    "top5_class_ids",
                    "top5_confidences",
                ],
            )

            writer.writeheader()
            writer.writerows(prediction_rows)

        # Save metrics
        with (
            model_output / "metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=2,
            )

        epoch = CHECKPOINT_EPOCHS[name]

        # Save per-class report
        save_classification_report(
            model_output,
            "test",
            epoch,
            labels_array,
            predictions_array,
            NUM_CLASSES,
        )

        # Save confusion matrix
        save_confusion_matrix(
            model_output,
            "test",
            epoch,
            labels_array,
            predictions_array,
            NUM_CLASSES,
        )

        # Print results
        print_metrics_box(
            f"{name.upper()} — ASL CITIZEN100 TEST",
            metrics,
        )

        print(f"Results saved in: {model_output}")

        return metrics

    finally:
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    # Prevent accidental overwriting
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Output directory already exists. "
            "Nothing will be overwritten:\n"
            f"{OUTPUT_DIR}"
        )

    # Verify checkpoints
    for name, checkpoint_path in CHECKPOINTS.items():

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing {name} checkpoint:\n"
                f"{checkpoint_path}"
            )

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 65)
    print("ASL CITIZEN100 — HELD-OUT TEST EVALUATION")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("Training: DISABLED")
    print("Checkpoint modification: DISABLED")

    # Load common paired test dataset
    dataset = ASLCitizenDataset(
        dataset_size=100,
        split="test",
        modality="paired",
    )

    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SAMPLES} paired test samples, "
            f"but found {len(dataset)}. Evaluation stopped."
        )

    print(f"\nCommon paired test samples: {len(dataset)}")

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Save evaluation metadata
    metadata = {
        "dataset": "ASL Citizen100",
        "split": "test",
        "modality": "paired",
        "num_classes": NUM_CLASSES,
        "evaluated_samples": len(dataset),
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "checkpoints": {
            name: str(path.relative_to(ROOT))
            for name, path in CHECKPOINTS.items()
        },
        "comparison_protocol": (
            "All three models evaluated on the same paired "
            "held-out test samples."
        ),
        "training_performed": False,
        "checkpoint_modification": False,
    }

    with (
        OUTPUT_DIR / "evaluation_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # Evaluate models sequentially
    all_results = {}

    for name in (
        "appearance",
        "pose",
        "fusion",
    ):

        all_results[name] = evaluate_model(
            name,
            dataset,
            device,
            OUTPUT_DIR,
        )

    # Save comparison summary
    summary_path = OUTPUT_DIR / "comparison_summary.csv"

    summary_fields = [
        "model",
        "samples",
        "accuracy",
        "loss",
        "top1_accuracy",
        "top5_accuracy",
        "wer_single_word_equivalent",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "evaluation_elapsed_seconds",
        "evaluation_samples_per_second",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        for name, metrics in all_results.items():

            row = {
                "model": name,
                **metrics,
            }

            writer.writerow(row)

    # Final summary
    print("\n" + "=" * 65)
    print("HELD-OUT TEST EVALUATION COMPLETED")
    print("=" * 65)

    for name, metrics in all_results.items():

        print(
            f"{name:12s} "
            f"Top-1: {metrics['top1_accuracy']:.2%} | "
            f"Top-5: {metrics['top5_accuracy']:.2%} | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )

    print(f"\nAll results: {OUTPUT_DIR}")
    print(f"Comparison: {summary_path}")


if __name__ == "__main__":
    main()
