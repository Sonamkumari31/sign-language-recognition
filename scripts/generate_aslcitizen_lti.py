
"""Generate LTI images from extracted ASL Citizen pose landmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.lti import landmarks_to_lti


def load_manifest(manifest_root, dataset, split):
    manifest_path = (
        manifest_root / f"aslcitizen{dataset}" / f"{split}.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "Video file" not in reader.fieldnames:
            raise ValueError(
                f"Manifest has no Video file column: {manifest_path}"
            )

        return list(reader)


def valid_existing_file(path):
    if not path.is_file():
        return False

    try:
        array = np.load(path, mmap_mode="r")
        return (
            array.shape == (224, 224, 3)
            and array.dtype == np.float32
            and np.isfinite(array).all()
        )
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate LTI images from ASL Citizen pose landmarks."
    )
    parser.add_argument(
        "--dataset",
        type=int,
        choices=[100, 300],
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum videos to process; 0 processes all videos.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("data/pose_landmarks"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifests"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lti_images"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("data/lti_reports"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    pose_root = args.pose_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()

    rows = load_manifest(
        manifest_root, args.dataset, args.split
    )

    if args.limit > 0:
        rows = rows[:args.limit]

    output_dir = (
        output_root / f"aslcitizen{args.dataset}" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    newly_generated = 0
    skipped_existing = 0
    failed = 0
    failures = []

    for row in tqdm(rows, desc=f"ASL Citizen{args.dataset} {args.split}"):
        video_name = row["Video file"]
        stem = Path(video_name).stem

        pose_path = pose_root / f"{stem}.npz"
        output_path = output_dir / f"{stem}.npy"
        temp_path = output_path.with_suffix(".tmp.npy")

        if valid_existing_file(output_path):
            skipped_existing += 1
            continue

        try:
            if not pose_path.is_file():
                raise FileNotFoundError(pose_path)

            with np.load(pose_path) as data:
                if not {"landmarks", "valid_mask"}.issubset(data.files):
                    raise ValueError(
                        f"Pose file missing required arrays: {pose_path}"
                    )

                landmarks = data["landmarks"]
                valid_mask = data["valid_mask"]

            if (
                landmarks.ndim != 3
                or landmarks.shape[1:] != (67, 3)
                or landmarks.dtype != np.float32
                or valid_mask.shape != landmarks.shape[:2]
                or valid_mask.dtype != np.bool_
                or not np.isfinite(landmarks).all()
            ):
                raise ValueError(
                    f"Invalid pose array format: {pose_path}"
                )

            lti = landmarks_to_lti(landmarks, valid_mask)

            if (
                lti.shape != (224, 224, 3)
                or lti.dtype != np.float32
                or not np.isfinite(lti).all()
            ):
                raise ValueError(
                    f"Invalid generated LTI: {video_name}"
                )

            np.save(temp_path, lti)

            if not valid_existing_file(temp_path):
                raise RuntimeError("Saved LTI file failed validation")

            temp_path.replace(output_path)
            newly_generated += 1

        except Exception as exc:
            failed += 1
            failures.append({
                "video": video_name,
                "error": str(exc),
            })

            if temp_path.exists():
                temp_path.unlink()

    report_dir = report_root / f"aslcitizen{args.dataset}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.split}_report.json"

    report = {
        "dataset": f"aslcitizen{args.dataset}",
        "split": args.split,
        "selected": len(rows),
        "newly_generated": newly_generated,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "failures": failures,
        "output_shape": [224, 224, 3],
        "dtype": "float32",
    }

    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nLTI generation finished.")
    print("New:", newly_generated)
    print("Skipped:", skipped_existing)
    print("Failed:", failed)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
