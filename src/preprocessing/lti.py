
"""Landmark-to-Image (LTI) transformation for ASL sign recognition.

Input:
    landmarks: float array shaped (L, 67, 3), in temporal order.
    valid_mask: optional boolean array shaped (L, 67).

Output:
    float32 array shaped (224, 224, 3), with x/y/z mapped to RGB channels.

Paper-derived behavior:
    Repeat each retained frame's landmark block consecutively, then
    zero-pad to 224*224 rows before reshaping.

Implementation decisions:
    - Frames without either detected hand are removed when a validity mask
      is provided.
    - Coordinate scaling is deliberately not applied: the paper does not
      specify its normalization procedure.
    - Videos exceeding the raster capacity are uniformly subsampled.
      This is an explicit implementation policy, not a paper-specified rule.
"""

from __future__ import annotations

import numpy as np

IMAGE_SIZE = 224
NUM_LANDMARKS = 67
NUM_COORDINATES = 3
NUM_ROWS = IMAGE_SIZE * IMAGE_SIZE


def landmarks_to_lti(
    landmarks: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a video's 3D landmarks into a 224x224x3 LTI image.

    Landmark order per frame:
        body: indices 0-24
        left hand: indices 25-45
        right hand: indices 46-66

    If valid_mask is supplied, retain only frames where at least one hand
    has at least one valid landmark. Missing landmarks remain zero-valued.

    Raises:
        ValueError: if shapes are invalid or no frames remain.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.ndim != 3 or landmarks.shape[1:] != (
        NUM_LANDMARKS,
        NUM_COORDINATES,
    ):
        raise ValueError(
            f"Expected landmarks shaped (L, 67, 3); got {landmarks.shape}."
        )

    if landmarks.shape[0] == 0:
        raise ValueError("Cannot create an LTI image from zero frames.")

    if not np.isfinite(landmarks).all():
        raise ValueError("Landmarks contain NaN or infinite values.")

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != landmarks.shape[:2]:
            raise ValueError(
                f"Expected valid_mask shaped {landmarks.shape[:2]}; "
                f"got {valid_mask.shape}."
            )

        left_hand_detected = valid_mask[:, 25:46].any(axis=1)
        right_hand_detected = valid_mask[:, 46:67].any(axis=1)
        keep_frame = left_hand_detected | right_hand_detected

        landmarks = landmarks[keep_frame]

        if landmarks.shape[0] == 0:
            raise ValueError(
                "No frames contain detected hand landmarks."
            )

    frame_count = landmarks.shape[0]

    # The raster holds exactly NUM_ROWS rows of 3D coordinates.
    max_frames = NUM_ROWS // NUM_LANDMARKS

    if frame_count > max_frames:
        # Explicit policy for the paper's underspecified long-video branch.
        indices = np.linspace(
            0, frame_count - 1, num=max_frames, dtype=np.int64
        )
        landmarks = landmarks[indices]
        frame_count = landmarks.shape[0]

    repetitions = NUM_ROWS // (frame_count * NUM_LANDMARKS)

    # Repeat each frame block consecutively, preserving landmark order.
    repeated = np.repeat(landmarks, repetitions, axis=0)
    coordinate_rows = repeated.reshape(-1, NUM_COORDINATES)

    # Zero-pad the remaining rows.
    cfm = np.zeros((NUM_ROWS, NUM_COORDINATES), dtype=np.float32)
    cfm[: coordinate_rows.shape[0]] = coordinate_rows

    # Row-major reshape; channels correspond to x, y, z respectively.
    return cfm.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_COORDINATES)
