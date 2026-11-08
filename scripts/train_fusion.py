
"""Independent fusion trainer for ASL Citizen100/300.

Architecture:
    Video Swin appearance branch -> 1024 features
    Swin pose branch             -> 1024 features
    Concatenation                 -> 2048 features
    Transformer fusion            -> class logits

The two pretrained/trained branches remain frozen in this baseline.
This script does not call scripts/train_stage.py.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aslcitizen_dataset import ASLCitizenDataset
from src.models.appearance_swin import AppearanceSwin
from src.models.pose_swin import PoseSwin
from src.models.fusion_transformer import FusionTransformer
from src.evaluation import (
    calculate_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_learning_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent two-stream Swin fusion trainer."
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/aslcitizen100.yaml",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument(
        "--appearance-checkpoint",
        default=None,
        help=(
            "Path to appearance best.pt; defaults to "
            "experiments/aslcitizenN/appearance/best.pt."
        ),
    )
    parser.add_argument(
        "--pose-checkpoint",
        default=None,
        help=(
            "Path to pose best.pt; defaults to "
            "experiments/aslcitizenN/pose/best.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Defaults to "
            "experiments/aslcitizenN/fusion_independent_v1."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from this independent trainer's last.pt checkpoint.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path):
    path = resolve_path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    if not isinstance(config.get("dataset"), dict):
        raise ValueError("Config must contain a dataset section.")

    if not isinstance(config.get("dataloader"), dict):
        raise ValueError("Config must contain a dataloader section.")

    dataset_config = config["dataset"]
    num_classes = int(dataset_config.get("num_classes", 0))

    if dataset_config.get("name") != "aslcitizen":
        raise ValueError("This script supports ASL Citizen only.")

    if num_classes not in (100, 300):
        raise ValueError("num_classes must be 100 or 300.")

    return config, num_classes


def make_loader(dataset, batch_size, shuffle, workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


class FusionStage(nn.Module):
    """Frozen Swin branches with a trainable Transformer fusion head."""

    def __init__(self, appearance, pose, fusion):
        super().__init__()

        self.appearance_swin = appearance
        self.pose_swin = pose
        self.fusion = fusion

        for parameter in self.appearance_swin.parameters():
            parameter.requires_grad = False

        for parameter in self.pose_swin.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        # Keep frozen branches in evaluation mode.
        super().train(mode)
        self.appearance_swin.eval()
        self.pose_swin.eval()
        return self

    def forward(self, frames, lti):
        # Frozen branches do not need gradients.
        with torch.no_grad():
            appearance_features = self.appearance_swin(frames)
            pose_features = self.pose_swin(lti)

        return self.fusion(appearance_features, pose_features)


def load_branch_checkpoint(path, num_classes, name):
    """Load branch weights from this repository's saved checkpoints.

    Supports:
      1. Checkpoints containing branch_state_dict.
      2. Checkpoints containing model_state_dict keys such as
         'branch.backbone.patch_embed.proj.weight'.

    In the second format, the 'branch.' prefix is removed before loading
    weights into AppearanceSwin or PoseSwin.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} checkpoint not found:\n{path}\n"
            "Provide the correct checkpoint path using the corresponding "
            "command-line argument."
        )

    # These are locally generated training checkpoints.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Invalid {name} checkpoint format: expected a dictionary."
        )

    saved_classes = checkpoint.get("num_classes")

    if saved_classes != num_classes:
        raise ValueError(
            f"{name} checkpoint class count does not match the config. "
            f"Config expects {num_classes}, checkpoint contains "
            f"{saved_classes}."
        )

    # Validate the saved branch metadata when it is available.
    expected_branch = name.lower()
    saved_branch = checkpoint.get("branch")

    if isinstance(saved_branch, str) and saved_branch.strip():
        normalized_branch = saved_branch.strip().lower()

        allowed_names = {
            expected_branch,
            f"{expected_branch}_swin",
            f"{expected_branch}_branch",
        }

        if normalized_branch not in allowed_names:
            raise ValueError(
                f"Expected a {expected_branch} checkpoint, but the "
                f"checkpoint metadata says branch={saved_branch!r}: {path}"
            )

    # Newer checkpoint format: branch parameters are stored separately.
    branch_state = checkpoint.get("branch_state_dict")

    if isinstance(branch_state, dict):
        state = branch_state

    # Existing repository format: full model state has "branch." prefixes.
    elif isinstance(checkpoint.get("model_state_dict"), dict):
        full_state = checkpoint["model_state_dict"]

        state = {
            key.removeprefix("branch."): value
            for key, value in full_state.items()
            if key.startswith("branch.")
        }

        if not state:
            raise KeyError(
                f"No parameters prefixed with 'branch.' were found in "
                f"{path}.\n"
                f"Example available keys: {list(full_state.keys())[:10]}"
            )

    else:
        raise KeyError(
            f"Unsupported {name} checkpoint format: {path}\n"
            f"Available checkpoint keys: {list(checkpoint.keys())}\n"
            "Expected 'branch_state_dict' or 'model_state_dict'."
        )

    if not state:
        raise ValueError(
            f"The extracted {name} branch state dictionary is empty: {path}"
        )

    print(
        f"Loaded {name.lower()} branch weights: "
        f"{len(state)} tensors from {path.name}"
    )

    return state


def make_model(num_classes, appearance_path, pose_path, device):
    appearance_state = load_branch_checkpoint(
        appearance_path,
        num_classes,
        "Appearance",
    )

    pose_state = load_branch_checkpoint(
        pose_path,
        num_classes,
        "Pose",
    )

    # Construct the branches without downloading or initializing
    # additional pretrained weights.
    appearance = AppearanceSwin(
        pretrained=False,
        feature_dim=1024,
    )

    pose = PoseSwin(
        pretrained=False,
        feature_dim=1024,
    )

    # Strict loading catches architecture/checkpoint mismatches.
    appearance.load_state_dict(appearance_state, strict=True)
    pose.load_state_dict(pose_state, strict=True)

    fusion = FusionTransformer(
        num_classes=num_classes,
        feature_dim=1024,
    )

    model = FusionStage(
        appearance=appearance,
        pose=pose,
        fusion=fusion,
    )

    return model.to(device)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    optimizer=None,
    max_batches=None,
    description="Train",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    top5_correct = 0
    batches = 0

    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

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
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(frames, lti)

            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise RuntimeError(
                    f"Expected logits [batch, {num_classes}], "
                    f"received {tuple(logits.shape)}."
                )

            if (
                labels.numel() == 0
                or labels.min().item() < 0
                or labels.max().item() >= num_classes
            ):
                raise ValueError("Invalid class labels.")

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite {description} loss: {loss.item()}"
                )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            topk = logits.detach().topk(
                min(5, num_classes),
                dim=1,
            ).indices

            top5_correct += (
                (topk == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )

            all_labels.extend(labels.detach().cpu().tolist())

            all_predictions.extend(
                logits.detach().argmax(dim=1).cpu().tolist()
            )

            batches += 1

            if batches % 10 == 0:
                print(
                    f"{description}: {batches} batches, "
                    f"loss={total_loss / total_samples:.4f}"
                )

    if total_samples == 0:
        raise RuntimeError(
            f"No samples processed during {description}."
        )

    labels_np = np.asarray(all_labels, dtype=np.int64)

    predictions_np = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    metrics = calculate_metrics(
        labels_np,
        predictions_np,
        num_classes,
    )

    accuracy = float(metrics["accuracy"])

    metrics.update(
        {
            "loss": total_loss / total_samples,
            "top1": accuracy,
            "top5": top5_correct / total_samples,
            "samples": total_samples,
            "batches": batches,
            # For isolated single-word recognition, this is a
            # proxy error rate, not sequence-level continuous-sign WER.
            "wer": 1.0 - accuracy,
        }
    )

    return metrics, labels_np, predictions_np


METRIC_NAMES = [
    "loss",
    "top1",
    "top5",
    "wer",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


def scalar_metrics(metrics):
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    }


def main():
    args = parse_args()

    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("--epochs and --lr must be positive.")

    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    for name, value in (
        ("--max-train-batches", args.max_train_batches),
        ("--max-val-batches", args.max_val_batches),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1.")

    partial_run = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    if partial_run and args.resume:
        raise ValueError(
            "Do not combine a partial smoke test with --resume."
        )

    set_seed(args.seed)

    config, num_classes = load_config(args.config)
    loader_config = config["dataloader"]

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(loader_config.get("batch_size", 2))
    )

    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(loader_config.get("num_workers", 0))
    )

    pin_memory = bool(
        loader_config.get("pin_memory", True)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    experiment_root = (
        PROJECT_ROOT / "experiments" / f"aslcitizen{num_classes}"
    )

    appearance_path = resolve_path(
        args.appearance_checkpoint
        or str(experiment_root / "appearance" / "best.pt")
    )

    pose_path = resolve_path(
        args.pose_checkpoint
        or str(experiment_root / "pose" / "best.pt")
    )

    output_dir = resolve_path(
        args.output_dir
        or str(experiment_root / "fusion_independent_v1")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"

    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print("\nIndependent fusion trainer")
    print(f"Config: {resolve_path(args.config)}")
    print(f"Classes: {num_classes}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Appearance checkpoint: {appearance_path}")
    print(f"Pose checkpoint: {pose_path}")
    print(f"Output directory: {output_dir}")
    print(f"Partial smoke test: {partial_run}")

    train_dataset = ASLCitizenDataset(
        dataset_size=num_classes,
        split="train",
        modality="paired",
    )

    val_dataset = ASLCitizenDataset(
        dataset_size=num_classes,
        split="val",
        modality="paired",
    )

    train_loader = make_loader(
        train_dataset,
        batch_size,
        bool(loader_config.get("shuffle_train", True)),
        workers,
        pin_memory,
    )

    val_loader = make_loader(
        val_dataset,
        batch_size,
        False,
        workers,
        pin_memory,
    )

    model = make_model(
        num_classes,
        appearance_path,
        pose_path,
        device,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable fusion parameters found.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
    )

    criterion = nn.CrossEntropyLoss()

    # Smoke tests use separate filenames so they cannot overwrite
    # full-run checkpoints or history.
    if partial_run:
        last_path = output_dir / "smoke_test.pt"
        history_path = output_dir / "smoke_test_history.csv"

    start_epoch = 1
    best_val_top1 = -1.0
    history = []

    if args.resume:
        resume_path = resolve_path(args.resume)

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        if checkpoint.get("stage") != "fusion_independent":
            raise ValueError(
                "This checkpoint was not created by this independent "
                "fusion trainer."
            )

        if checkpoint.get("num_classes") != num_classes:
            raise ValueError(
                "Resume checkpoint class count does not match config."
            )

        if checkpoint.get("is_partial_run", False):
            raise ValueError(
                "Cannot resume from a partial smoke-test checkpoint."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        start_epoch = int(checkpoint["epoch"]) + 1

        best_val_top1 = float(
            checkpoint.get("best_val_top1", -1.0)
        )

        print(f"Resuming from: {resume_path}")
        print(f"Next epoch: {start_epoch}")

        if history_path.is_file():
            with history_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                history = list(csv.DictReader(file))

    elif not partial_run and (
        last_path.exists() or best_path.exists()
    ):
        raise FileExistsError(
            f"Checkpoints already exist in {output_dir}. "
            "Choose a new --output-dir or use --resume."
        )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch - 1}. "
            f"Set --epochs to a larger total, such as {start_epoch + 9}."
        )

    fields = ["epoch"]

    for prefix in ("train", "val"):
        fields.extend(
            f"{prefix}_{metric}" for metric in METRIC_NAMES
        )

    fields.extend(["train_samples", "val_samples"])

    file_mode = (
        "a" if args.resume and not partial_run else "w"
    )

    write_header = (
        file_mode == "w" or not history_path.exists()
    )

    with history_path.open(
        file_mode,
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_metrics, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                num_classes,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
                description="Train",
            )

            val_metrics, val_labels, val_predictions = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                optimizer=None,
                max_batches=args.max_val_batches,
                description="Validation",
            )

            for name, metrics in (
                ("Train", train_metrics),
                ("Validation", val_metrics),
            ):
                print(
                    f"{name}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"top-1={metrics['top1']:.4f}, "
                    f"top-5={metrics['top5']:.4f}, "
                    f"macro-F1={metrics['macro_f1']:.4f}, "
                    f"balanced accuracy="
                    f"{metrics['balanced_accuracy']:.4f}"
                )

            row = {
                "epoch": epoch,
                "train_samples": train_metrics["samples"],
                "val_samples": val_metrics["samples"],
            }

            for prefix, metrics in (
                ("train", train_metrics),
                ("val", val_metrics),
            ):
                for metric in METRIC_NAMES:
                    row[f"{prefix}_{metric}"] = metrics[metric]

            writer.writerow(row)
            csv_file.flush()
            history.append(row)

            save_classification_report(
                evaluation_dir,
                "val",
                epoch,
                val_labels,
                val_predictions,
                num_classes,
            )

            save_confusion_matrix(
                evaluation_dir,
                "val",
                epoch,
                val_labels,
                val_predictions,
                num_classes,
            )

            improved = (
                not partial_run
                and val_metrics["top1"] > best_val_top1
            )

            if improved:
                best_val_top1 = val_metrics["top1"]

            checkpoint = {
                "stage": "fusion_independent",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "learning_rate": args.lr,
                "seed": args.seed,
                "config": config,
                "appearance_checkpoint": str(appearance_path),
                "pose_checkpoint": str(pose_path),
                "train_metrics": scalar_metrics(train_metrics),
                "val_metrics": scalar_metrics(val_metrics),
                "best_val_top1": best_val_top1,
                "is_partial_run": partial_run,
            }

            torch.save(checkpoint, last_path)
            print(f"Saved checkpoint: {last_path}")

            if improved:
                torch.save(checkpoint, best_path)
                print(f"Saved best checkpoint: {best_path}")

            if not partial_run:
                try:
                    save_learning_curves(history, output_dir)
                except (KeyError, TypeError, ValueError) as exc:
                    print(
                        f"Warning: could not update learning curves: {exc}"
                    )

    print("\nFusion training finished.")
    print(f"History: {history_path}")
    print(f"Last checkpoint: {last_path}")

    if not partial_run:
        print(f"Best checkpoint: {best_path}")
        print(f"Validation reports: {evaluation_dir}")
    else:
        print("Smoke-test results are not final model results.")


if __name__ == "__main__":
    main()
