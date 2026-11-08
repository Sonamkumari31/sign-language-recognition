
"""PyTorch dataset for ASL Citizen appearance, pose, and fusion inputs."""

from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"
FRAME_ROOT = PROJECT_ROOT / "data" / "frames"
LTI_ROOT = PROJECT_ROOT / "data" / "lti_images"

IMAGE_SIZE = 224
NUM_FRAMES = 16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path):
    """Read a CSV manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_label_map(dataset_size):
    """Build a stable label map using the training split only."""
    if dataset_size not in (100, 300):
        raise ValueError("dataset_size must be 100 or 300")

    train_path = MANIFEST_ROOT / f"aslcitizen{dataset_size}" / "train.csv"
    rows = read_manifest(train_path)
    glosses = sorted({row["Gloss"] for row in rows})

    return {gloss: index for index, gloss in enumerate(glosses)}


class ASLCitizenDataset(Dataset):
    """Load appearance frames, pose LTI images, or both modalities.

    modality="appearance": RGB frame sequences only.
    modality="pose":       LTI images only.
    modality="paired":     RGB frame sequences and LTI images.
    """

    def __init__(self, dataset_size=300, split="train", modality="paired"):
        if dataset_size not in (100, 300):
            raise ValueError("dataset_size must be 100 or 300")

        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")

        if modality not in ("appearance", "pose", "paired"):
            raise ValueError(
                "modality must be 'appearance', 'pose', or 'paired'"
            )

        self.dataset_size = dataset_size
        self.split = split
        self.modality = modality
        self.label_map = load_label_map(dataset_size)

        manifest_path = (
            MANIFEST_ROOT / f"aslcitizen{dataset_size}" / f"{split}.csv"
        )
        rows = read_manifest(manifest_path)

        self.samples = []
        self.missing_frames = []
        self.missing_lti = []

        frame_dir = FRAME_ROOT / f"aslcitizen{dataset_size}" / split
        lti_dir = LTI_ROOT / f"aslcitizen{dataset_size}" / split

        for row in rows:
            video_name = row["Video file"]
            stem = Path(video_name).stem
            gloss = row["Gloss"]

            frame_path = frame_dir / f"{stem}.npy"
            lti_path = lti_dir / f"{stem}.npy"

            # Check only the modalities required by this dataset instance.
            if (
                modality in ("appearance", "paired")
                and not frame_path.is_file()
            ):
                self.missing_frames.append(video_name)
                continue

            if (
                modality in ("pose", "paired")
                and not lti_path.is_file()
            ):
                self.missing_lti.append(video_name)
                continue

            if gloss not in self.label_map:
                raise ValueError(
                    f"Gloss {gloss!r} in {split}.csv is absent "
                    "from the training label map"
                )

            self.samples.append(
                {
                    "video_name": video_name,
                    "gloss": gloss,
                    "label": self.label_map[gloss],
                    "frame_path": frame_path,
                    "lti_path": lti_path,
                }
            )

        print(
            f"ASL Citizen{dataset_size} {split} [{modality}]: "
            f"{len(self.samples)}/{len(rows)} samples available"
        )

        if modality in ("appearance", "paired"):
            print(f"Missing frames: {len(self.missing_frames)}")

        if modality in ("pose", "paired"):
            print(f"Missing LTI images: {len(self.missing_lti)}")

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found for ASL Citizen{dataset_size} "
                f"{split} with modality={modality!r}. "
                "Check manifests and preprocessed files."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        result = {
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "video_name": sample["video_name"],
        }

        # Appearance branch: 16 RGB frames, ImageNet-normalized.
        if self.modality in ("appearance", "paired"):
            frames = np.load(sample["frame_path"], allow_pickle=False)

            expected_shape = (
                NUM_FRAMES,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            )

            if frames.shape != expected_shape:
                raise ValueError(
                    f"Unexpected frame shape for {sample['video_name']}: "
                    f"{frames.shape}; expected {expected_shape}"
                )

            if frames.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 frames for {sample['video_name']}, "
                    f"got {frames.dtype}"
                )

            frames = torch.from_numpy(frames.copy())
            frames = frames.permute(0, 3, 1, 2).float() / 255.0

            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

            result["frames"] = (frames - mean) / std

        # Pose branch: preprocessed LTI image, retaining coordinate values.
        if self.modality in ("pose", "paired"):
            lti = np.load(sample["lti_path"], allow_pickle=False)

            expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

            if lti.shape != expected_shape:
                raise ValueError(
                    f"Unexpected LTI shape for {sample['video_name']}: "
                    f"{lti.shape}; expected {expected_shape}"
                )

            if lti.dtype != np.float32:
                raise ValueError(
                    f"Expected float32 LTI for {sample['video_name']}, "
                    f"got {lti.dtype}"
                )

            if not np.isfinite(lti).all():
                raise ValueError(
                    f"Non-finite values in LTI for {sample['video_name']}"
                )

            result["lti"] = torch.from_numpy(
                lti.copy()
            ).permute(2, 0, 1)

        return result
