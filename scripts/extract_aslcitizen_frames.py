
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
NUM_FRAMES = 16
IMAGE_SIZE = 224


def extract_frames(video_path):
    """Return 16 evenly spaced RGB frames resized to 224x224."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        indices = np.linspace(
            0, total_frames - 1, NUM_FRAMES
        ).round().astype(int)

        frames = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if not success or frame is None:
                raise RuntimeError(
                    f"Failed reading frame {index}: {video_path}"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            frames.append(frame)

        result = np.stack(frames).astype(np.uint8)

        expected_shape = (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
        if result.shape != expected_shape:
            raise RuntimeError(f"Unexpected output shape: {result.shape}")

        return result

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB frames from ASL Citizen videos."
    )
    parser.add_argument(
        "--dataset",
        choices=("100", "300"),
        required=True,
        help="ASL Citizen class subset to process.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum videos per split; 0 processes all videos.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="ASL Citizen dataset directory containing videos/.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifests"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/frames"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_root}")

    dataset_name = f"aslcitizen{args.dataset}"
    manifest_dir = manifest_root / dataset_name
    dataset_output = output_root / dataset_name
    video_root = dataset_root / "videos"

    selected_splits = (
        SPLITS if args.split == "all" else (args.split,)
    )

    for split in selected_splits:
        manifest_path = manifest_dir / f"{split}.csv"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        split_output = dataset_output / split
        split_output.mkdir(parents=True, exist_ok=True)

        with manifest_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        if args.limit > 0:
            rows = rows[:args.limit]

        succeeded = 0
        skipped = 0
        failed = []

        print(f"\nDataset: {dataset_name} | Split: {split}")
        print(f"Videos selected: {len(rows)}")

        for index, row in enumerate(rows, start=1):
            video_name = row["Video file"]
            video_path = video_root / video_name
            output_path = split_output / f"{Path(video_name).stem}.npy"

            if output_path.exists():
                try:
                    existing = np.load(output_path, mmap_mode="r")
                    if (
                        existing.shape == (
                            NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3
                        )
                        and existing.dtype == np.uint8
                    ):
                        skipped += 1
                        continue
                except (OSError, ValueError):
                    pass

            try:
                frames = extract_frames(video_path)
                np.save(output_path, frames)
                succeeded += 1
            except Exception as exc:
                failed.append({
                    "video": video_name,
                    "error": str(exc),
                })

            if index % 25 == 0 or index == len(rows):
                print(
                    f"Progress: {index}/{len(rows)} | "
                    f"new={succeeded} | skipped={skipped} | "
                    f"failed={len(failed)}"
                )

        report = {
            "dataset": dataset_name,
            "split": split,
            "selected": len(rows),
            "newly_extracted": succeeded,
            "skipped_existing": skipped,
            "failed": len(failed),
            "failures": failed,
            "shape": [NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3],
            "dtype": "uint8",
        }

        report_path = split_output / "extraction_report.json"
        report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        print(f"Finished {split}: {report_path}")

    print("\nFrame extraction run completed.")


if __name__ == "__main__":
    main()
