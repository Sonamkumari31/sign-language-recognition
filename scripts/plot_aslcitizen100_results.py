
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "aslcitizen100"
TEST = EXPERIMENT / "held_out_test_20261105"
OUTPUT = EXPERIMENT / "research_figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Appearance": {
        "history": EXPERIMENT / "appearance_full_v1" / "history.csv",
        "test": TEST / "appearance",
    },
    "Pose": {
        "history": EXPERIMENT / "pose_full" / "history.csv",
        "test": TEST / "pose",
    },
    "Fusion": {
        "history": EXPERIMENT / "fusion_full_v1" / "history.csv",
        "test": TEST / "fusion",
    },
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
})


def load_histories():
    histories = {}

    for name, paths in MODELS.items():
        if not paths["history"].exists():
            raise FileNotFoundError(paths["history"])

        df = pd.read_csv(paths["history"])
        histories[name] = df

    return histories


def load_test_results():
    results = {}

    for name, paths in MODELS.items():
        metrics_file = paths["test"] / "metrics.json"
        per_class_file = next(
            paths["test"].glob("test_per_class_*.csv")
        )
        confusion_file = next(
            paths["test"].glob("test_confusion_*.npy")
        )

        results[name] = {
            "metrics": pd.read_json(metrics_file, typ="series"),
            "per_class": pd.read_csv(per_class_file),
            "confusion": np.load(confusion_file),
        }

    return results


def save_training_curve(histories, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, df in histories.items():
        ax.plot(
            df["epoch"], df[f"train_{metric}"],
            marker="o", label=f"{name} - Train"
        )
        ax.plot(
            df["epoch"], df[f"val_{metric}"],
            marker="s", linestyle="--",
            label=f"{name} - Validation"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_test_comparison(results):
    metrics = {
        "Top-1 Accuracy": "top1_accuracy",
        "Top-5 Accuracy": "top5_accuracy",
        "Macro F1": "macro_f1",
        "Weighted F1": "weighted_f1",
        "Balanced Accuracy": "balanced_accuracy",
    }

    rows = []

    for model, data in results.items():
        for label, key in metrics.items():
            rows.append({
                "Model": model,
                "Metric": label,
                "Score (%)": float(data["metrics"][key]) * 100,
            })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=df,
        x="Metric",
        y="Score (%)",
        hue="Model",
        ax=ax,
    )

    ax.set_title("ASL Citizen100 Held-Out Test Performance")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "test_metrics_comparison.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_per_class_f1(results):
    frames = []

    for model, data in results.items():
        df = data["per_class"].copy()
        df["Model"] = model
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(17, 7))

    sns.lineplot(
        data=combined,
        x="class_id",
        y="f1-score",
        hue="Model",
        marker="",
        ax=ax,
    )

    ax.set_title("Per-Class F1 Comparison - ASL Citizen100")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "per_class_f1_comparison.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrices(results):
    for model, data in results.items():
        matrix = data["confusion"]

        if matrix.shape != (100, 100):
            raise ValueError(
                f"{model}: expected 100x100 matrix, "
                f"got {matrix.shape}"
            )

        # Normalize each true-class row to show recall distribution.
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )

        fig, ax = plt.subplots(figsize=(13, 11))

        sns.heatmap(
            normalized,
            cmap="viridis",
            vmin=0,
            vmax=1,
            xticklabels=10,
            yticklabels=10,
            ax=ax,
            cbar_kws={"label": "Proportion of true-class samples"},
        )

        ax.set_title(
            f"{model} - Normalized Confusion Matrix"
        )
        ax.set_xlabel("Predicted Class ID")
        ax.set_ylabel("True Class ID")
        fig.tight_layout()

        filename = (
            model.lower() + "_confusion_matrix.png"
        )
        fig.savefig(
            OUTPUT / filename,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    print("Loading ASL100 training histories...")
    histories = load_histories()

    print("Loading held-out test results...")
    results = load_test_results()

    print("Generating training curves...")

    save_training_curve(
        histories,
        "loss",
        "Loss",
        "Training and Validation Loss",
        "training_validation_loss.png",
    )

    save_training_curve(
        histories,
        "top1",
        "Top-1 Accuracy",
        "Training and Validation Top-1 Accuracy",
        "training_validation_top1.png",
    )

    save_training_curve(
        histories,
        "macro_f1",
        "Macro F1",
        "Training and Validation Macro F1",
        "training_validation_macro_f1.png",
    )

    print("Generating test comparison...")
    plot_test_comparison(results)

    print("Generating per-class F1 comparison...")
    plot_per_class_f1(results)

    print("Generating confusion matrices...")
    plot_confusion_matrices(results)

    print("\nAll figures generated successfully.")
    print("Output directory:", OUTPUT)

    for path in sorted(OUTPUT.glob("*.png")):
        print(" -", path.name)


if __name__ == "__main__":
    main()
