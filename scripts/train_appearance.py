
"""Train the ASL Citizen appearance-only Swin classifier."""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aslcitizen_dataset import ASLCitizenDataset
from src.models.branch_classifiers import AppearanceClassifier
from src.evaluation import (
    calculate_metrics,
    print_metrics_box,
    save_classification_report,
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_json,
    measure_inference_latency,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/aslcitizen100.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--run-name", default="appearance_fresh")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(
    model, loader, criterion, device,
    optimizer=None, scaler=None, max_batches=0,
    backbone_frozen=True,
):
    training = optimizer is not None
    model.train(training)

    if training and backbone_frozen:
        model.branch.backbone.eval()

    loss_sum = 0.0
    sample_count = 0
    top1_correct = 0
    top5_correct = 0
    labels_all = []
    predictions_all = []

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(frames)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size

        top_indices = logits.detach().topk(
            min(5, logits.shape[1]), dim=1
        ).indices
        top1_correct += (top_indices[:, 0] == labels).sum().item()
        top5_correct += (
            top_indices == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(
            logits.detach().argmax(dim=1).cpu().tolist()
        )

    if sample_count == 0:
        raise RuntimeError("No samples processed; check batch limits.")

    metrics = calculate_metrics(
        labels_all, predictions_all, model.classifier[-1].out_features
    )
    metrics["loss"] = loss_sum / sample_count
    metrics["top1_accuracy"] = top1_correct / sample_count
    metrics["top5_accuracy"] = top5_correct / sample_count
    metrics["wer_single_word_equivalent"] = 1.0 - metrics["top1_accuracy"]
    return metrics, labels_all, predictions_all


def write_history(path, history):
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    seed_everything(args.seed)
    config_path = ROOT / args.config
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    num_classes = int(config["dataset"]["num_classes"])
    if num_classes not in (100, 300):
        raise ValueError("This script supports ASL Citizen100 or 300.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = f"aslcitizen{num_classes}"
    output_dir = ROOT / "experiments" / dataset_name / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: ASL Citizen{num_classes}")
    print("Branch: appearance only")
    print(f"Classes: {num_classes}")
    print(f"Backbone frozen: {not args.unfreeze_backbone}")
    print(f"Output directory: {output_dir}")

    train_dataset = ASLCitizenDataset(
        dataset_size=num_classes, split="train", modality="appearance"
    )
    val_dataset = ASLCitizenDataset(
        dataset_size=num_classes, split="val", modality="appearance"
    )

    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.workers, False
    )

    model = AppearanceClassifier(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized from checkpoint: {checkpoint_path}")

    backbone_frozen = not args.unfreeze_backbone
    backbone_params, head_params = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "branch.backbone." in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params, "lr": args.backbone_lr
        })
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    history = []
    best_val_top1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()

        train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_train_batches,
            backbone_frozen=backbone_frozen,
        )

        with torch.no_grad():
            val_metrics, val_labels, val_predictions = run_epoch(
                model, val_loader, criterion, device,
                max_batches=args.max_val_batches,
            )

        print_metrics_box(f"TRAINING — EPOCH {epoch}", train_metrics)
        print_metrics_box(f"VALIDATION — EPOCH {epoch}", val_metrics)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "val_top1": val_metrics["top1_accuracy"],
            "train_top5": train_metrics["top5_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        write_history(output_dir / "history.csv", history)
        save_learning_curves(history, output_dir)

        print(f"Epoch duration: {time.perf_counter() - started:.1f} seconds")

        # Select checkpoints using validation top-1 only.
        if val_metrics["top1_accuracy"] > best_val_top1:
            best_val_top1 = val_metrics["top1_accuracy"]
            best_epoch = epoch
            stale_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "dataset": dataset_name,
                "branch": "appearance",
                "epoch": epoch,
                "val_metrics": val_metrics,
                "seed": args.seed,
            }, output_dir / "best.pt")

            save_metrics_json(
                val_metrics, output_dir, "best_validation_metrics.json"
            )
            save_classification_report(
                output_dir, "validation", epoch,
                val_labels, val_predictions, num_classes,
            )
            save_confusion_matrix(
                output_dir, "validation", epoch,
                val_labels, val_predictions, num_classes,
            )
        else:
            stale_epochs += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }, output_dir / "last.pt")

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    print(f"Best validation top-1: {best_val_top1:.2%} at epoch {best_epoch}")

    if args.skip_test:
        print("Test evaluation skipped by --skip-test.")
        print("These results are not final test results.")
        return

    # Load the best validation-selected checkpoint before test evaluation.
    checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = ASLCitizenDataset(
        dataset_size=num_classes, split="test", modality="appearance"
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, args.workers, False
    )

    with torch.no_grad():
        test_metrics, test_labels, test_predictions = run_epoch(
            model, test_loader, criterion, device
        )

    print_metrics_box("FINAL TEST RESULTS", test_metrics)
    save_metrics_json(test_metrics, output_dir, "test_metrics.json")
    save_classification_report(
        output_dir, "test", best_epoch,
        test_labels, test_predictions, num_classes,
    )
    save_confusion_matrix(
        output_dir, "test", best_epoch,
        test_labels, test_predictions, num_classes,
    )

    example_batch = next(iter(test_loader))
    latency = measure_inference_latency(
        model, example_batch["frames"], device,
        warmup=2, iterations=10,
    )
    print_metrics_box("INFERENCE LATENCY", latency)
    save_metrics_json(latency, output_dir, "inference_latency.json")
    print(f"All results saved in: {output_dir}")


if __name__ == "__main__":
    main()
