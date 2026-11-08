"""Raw-video preprocessing for ASL inference."""

import cv2
import mediapipe as mp
import numpy as np
import torch

from scripts.extract_aslcitizen_frames import extract_frames
from scripts.extract_aslcitizen_pose import extract_landmarks
from src.preprocessing.lti import landmarks_to_lti
from src.data.aslcitizen_dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess_video(video_path):
    """Convert an input video into appearance and pose model inputs."""

    frames = extract_frames(video_path)

    if frames is None or len(frames) == 0:
        raise ValueError("Could not extract frames from the video.")

    if frames.shape != (16, 224, 224, 3):
        raise ValueError(
            f"Expected frames of shape (16, 224, 224, 3), "
            f"received {frames.shape}"
        )

    frames = frames.astype(np.float32) / 255.0

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)

    frames = (frames - mean) / std
    frames = np.transpose(frames, (0, 3, 1, 2))

    frames_tensor = torch.from_numpy(
        np.ascontiguousarray(frames)
    ).float().unsqueeze(0)

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        landmarks, mask = extract_landmarks(video_path, holistic)

    if landmarks is None or mask is None:
        raise ValueError("Could not extract pose landmarks.")

    if np.asarray(mask).sum() == 0:
        raise ValueError("No valid pose landmarks detected.")

    lti = landmarks_to_lti(landmarks, mask)

    if lti is None:
        raise ValueError("Could not generate the LTI representation.")

    lti = np.asarray(lti, dtype=np.float32)

    if lti.shape != (224, 224, 3):
        raise ValueError(f"Unexpected LTI shape: {lti.shape}")

    lti_tensor = torch.from_numpy(
        np.ascontiguousarray(
            np.transpose(lti, (2, 0, 1))
        )
    ).float().unsqueeze(0)

    return frames_tensor, lti_tensor