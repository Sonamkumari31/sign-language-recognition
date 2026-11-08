
import unittest

import numpy as np

from src.preprocessing.lti import landmarks_to_lti


class TestLandmarksToLTI(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        landmarks = np.ones((60, 67, 3), dtype=np.float32)

        result = landmarks_to_lti(landmarks)

        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_per_frame_repetition_and_zero_padding(self):
        landmarks = np.zeros((60, 67, 3), dtype=np.float32)

        for frame in range(60):
            landmarks[frame, :, :] = frame + 1

        result = landmarks_to_lti(landmarks)
        rows = result.reshape(224 * 224, 3)

        # r = floor(50176 / (60 * 67)) = 12.
        self.assertTrue(np.all(rows[:804].reshape(12, 67, 3) == 1))
        self.assertTrue(np.all(rows[12 * 67 : 24 * 67] == 2))

        filled_rows = 60 * 12 * 67
        self.assertTrue(np.all(rows[filled_rows:] == 0))

    def test_xyz_channel_mapping(self):
        landmarks = np.zeros((100, 67, 3), dtype=np.float32)
        landmarks[..., 0] = 0.1
        landmarks[..., 1] = 0.2
        landmarks[..., 2] = -0.3

        result = landmarks_to_lti(landmarks)

        np.testing.assert_allclose(result[0, 0], [0.1, 0.2, -0.3])

    def test_handless_frames_are_removed(self):
        landmarks = np.zeros((2, 67, 3), dtype=np.float32)
        landmarks[0] = 1.0
        landmarks[1] = 2.0

        mask = np.zeros((2, 67), dtype=bool)
        mask[1, 25] = True  # A detected left-hand landmark.

        result = landmarks_to_lti(landmarks, mask)
        rows = result.reshape(224 * 224, 3)

        self.assertTrue(np.all(rows[:67] == 2.0))

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            landmarks_to_lti(np.empty((0, 67, 3), dtype=np.float32))

    def test_rejects_video_without_detected_hands(self):
        landmarks = np.zeros((2, 67, 3), dtype=np.float32)
        mask = np.zeros((2, 67), dtype=bool)

        with self.assertRaises(ValueError):
            landmarks_to_lti(landmarks, mask)


if __name__ == "__main__":
    unittest.main()
