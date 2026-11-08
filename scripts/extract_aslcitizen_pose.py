
import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS = 67


def get_video_names(manifest_root, dataset, split):
    manifest = (
        manifest_root / f"aslcitizen{dataset}" / f"{split}.csv"
    )

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    with manifest.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "Video file" not in reader.fieldnames:
            raise ValueError("Manifest has no Video file column")

        names = [row["Video file"] for row in reader]

    return list(dict.fromkeys(names))


def valid_existing_file(path):
    if not path.exists():
        return False

    try:
        with np.load(path) as data:
            if not {"landmarks", "valid_mask"}.issubset(data.files):
                return False

            landmarks = data["landmarks"]
            mask = data["valid_mask"]

            return (
                landmarks.ndim == 3
                and landmarks.shape[1:] == (67, 3)
                and landmarks.shape[0] > 0
                and landmarks.dtype == np.float32
                and mask.shape == landmarks.shape[:2]
                and mask.dtype == np.bool_
                and np.isfinite(landmarks).all()
                and np.all(landmarks[~mask] == 0)
            )
    except Exception:
        return False


def extract_landmarks(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open video")

    all_landmarks = []
    all_masks = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)

            points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            mask = np.zeros(NUM_LANDMARKS, dtype=bool)

            # Body: landmarks 0-24
            if result.pose_landmarks is not None:
                for index in range(25):
                    lm = result.pose_landmarks.landmark[index]
                    points[index] = [lm.x, lm.y, lm.z]
                    mask[index] = True

            # Left hand: landmarks 25-45
            if result.left_hand_landmarks is not None:
                for index, lm in enumerate(
                    result.left_hand_landmarks.landmark
                ):
                    points[25 + index] = [lm.x, lm.y, lm.z]
                    mask[25 + index] = True

            # Right hand: landmarks 46-66
            if result.right_hand_landmarks is not None:
                for index, lm in enumerate(
                    result.right_hand_landmarks.landmark
                ):
                    points[46 + index] = [lm.x, lm.y, lm.z]
                    mask[46 + index] = True

            all_landmarks.append(points)
            all_masks.append(mask)

    finally:
        cap.release()

    if not all_landmarks:
        raise RuntimeError("No frames decoded")

    return np.stack(all_landmarks), np.stack(all_masks)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks from ASL Citizen."
    )
    parser.add_argument(
        "--dataset",
        type=int,
        choices=[100, 300],
        default=100,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--limit", type=int, default=None)
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
        default=Path("data/pose_landmarks"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("data/pose_reports"),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()

    if not dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_root}")

    video_root = dataset_root / "videos"

    video_names = get_video_names(
        manifest_root, args.dataset, args.split
    )

    if args.limit is not None:
        video_names = video_names[:args.limit]

    print(f"\nDataset: ASL Citizen{args.dataset}")
    print(f"Split: {args.split}")
    print(f"Videos selected: {len(video_names)}")

    newly_extracted = 0
    skipped_existing = 0
    failed = 0
    failures = []

    for index, video_name in enumerate(video_names, start=1):
        video_path = video_root / video_name
        output_path = output_root / (Path(video_name).stem + ".npz")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if valid_existing_file(output_path):
            skipped_existing += 1
            continue

        temp_path = output_path.with_suffix(".tmp.npz")

        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)

            with mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                landmarks, valid_mask = extract_landmarks(
                    video_path, holistic
                )

            np.savez_compressed(
                temp_path,
                landmarks=landmarks,
                valid_mask=valid_mask,
            )

            if not valid_existing_file(temp_path):
                raise RuntimeError("Saved file failed validation")

            temp_path.replace(output_path)
            newly_extracted += 1

        except Exception as exc:
            failed += 1
            failures.append({
                "video": video_name,
                "error": str(exc),
            })

            if temp_path.exists():
                temp_path.unlink()

        if index % 25 == 0 or index == len(video_names):
            print(
                f"Progress: {index}/{len(video_names)} | "
                f"new={newly_extracted} | "
                f"skipped={skipped_existing} | failed={failed}",
                flush=True,
            )

    report_dir = report_root / f"aslcitizen{args.dataset}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.split}_report.json"

    report = {
        "dataset": f"aslcitizen{args.dataset}",
        "split": args.split,
        "selected": len(video_names),
        "newly_extracted": newly_extracted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "failures": failures,
        "landmarks_per_frame": NUM_LANDMARKS,
        "coordinate_dimensions": 3,
    }

    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nExtraction finished.")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
