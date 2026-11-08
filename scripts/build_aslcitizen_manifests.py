
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")


def read_split(dataset_root, split):
    path = dataset_root / "splits" / f"{split}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Split file is empty: {path}")

    required = {"Gloss", "Participant ID", "Video file"}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build ASL Citizen100 and ASL Citizen300 manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="ASL Citizen dataset directory containing videos/ and splits/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/manifests"),
        help="Directory where subset manifests will be written.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_root}")

    data = {
        split: read_split(dataset_root, split)
        for split in SPLITS
    }

    # Rank classes using training data only.
    train_counts = Counter(row["Gloss"] for row in data["train"])
    ranked = sorted(train_counts, key=lambda gloss: (-train_counts[gloss], gloss))

    if len(ranked) < 300:
        raise ValueError(
            f"At least 300 training glosses are required; found {len(ranked)}."
        )

    # Verify that original splits have no signer overlap.
    participants = {
        split: {row["Participant ID"] for row in rows}
        for split, rows in data.items()
    }

    for first, second in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = participants[first] & participants[second]
        if overlap:
            raise ValueError(
                f"Participant overlap between {first} and {second}: "
                f"{len(overlap)} participants."
            )

    # Verify every referenced video exists before generating manifests.
    video_root = dataset_root / "videos"
    missing = [
        row["Video file"]
        for rows in data.values()
        for row in rows
        if not (video_root / row["Video file"]).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} videos are missing; first examples: {missing[:5]}"
        )

    for num_classes in (100, 300):
        classes = ranked[:num_classes]
        class_set = set(classes)

        output_dir = output_root / f"aslcitizen{num_classes}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "class_list.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            file.write("\n".join(classes) + "\n")

        summary = {
            "num_classes": num_classes,
            "class_selection": (
                "training frequency descending; alphabetical tie-break"
            ),
        }

        for split in SPLITS:
            rows = [
                row for row in data[split]
                if row["Gloss"] in class_set
            ]

            output_path = output_dir / f"{split}.csv"

            with output_path.open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=data[split][0].keys(),
                )
                writer.writeheader()
                writer.writerows(rows)

            summary[f"{split}_videos"] = len(rows)
            summary[f"{split}_classes"] = len(
                {row["Gloss"] for row in rows}
            )

        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)

        print(f"\nASL Citizen{num_classes}")
        print("Classes:", num_classes)

        for split in SPLITS:
            print(
                f"{split}: {summary[f'{split}_videos']} videos, "
                f"{summary[f'{split}_classes']} classes"
            )

        print("Output:", output_dir)

    print("\nManifest generation and validation completed.")


if __name__ == "__main__":
    main()
