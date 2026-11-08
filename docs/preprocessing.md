# ASL Citizen Dataset Preprocessing

## 1. Overview

The preprocessing pipeline prepares ASL Citizen videos for two modalities:

1. Appearance: 16 RGB video frames resized to 224 x 224.
2. Pose: MediaPipe Holistic landmarks converted into Landmark Temporal Images (LTI).

The pipeline supports the ASL Citizen100 and ASL Citizen300 subsets.

## 2. Manifest Generation

The manifest builder reads the official train, validation and test CSV files.

Class selection:
- Counts gloss frequency using the training split only.
- Selects the 100 or 300 most frequent glosses.
- Resolves frequency ties alphabetically.
- Applies the selected classes to all three original splits.
- Checks participant separation between train, validation and test.
- Verifies that all referenced videos exist.

Each subset contains:
- `class_list.txt`
- `train.csv`
- `val.csv`
- `test.csv`
- `summary.json`

## 3. Appearance Preprocessing

The frame extraction script:
1. Reads each video.
2. Selects 16 evenly spaced frame indices.
3. Converts frames from BGR to RGB.
4. Resizes frames to 224 x 224 using bilinear interpolation.
5. Saves the resulting array as `uint8`.

Output shape:

`(16, 224, 224, 3)`

Output directory:

`data/frames/aslcitizen100/` or `data/frames/aslcitizen300/`

Each subset has separate train, validation and test directories. Extraction reports are saved within the corresponding split directory.

## 4. Pose Preprocessing

MediaPipe Holistic extracts landmarks from every decoded video frame.

The 67-landmark representation is organized as follows:

| Landmark group | Indices | Count |
|---|---:|---:|
| Body pose | 0-24 | 25 |
| Left hand | 25-45 | 21 |
| Right hand | 46-66 | 21 |
| Face | Excluded | 0 |

Each frame contains three coordinates per landmark: x, y and z.

Two arrays are saved in each compressed NPZ file:

- `landmarks`: float32, shape `(K, 67, 3)`
- `valid_mask`: Boolean, shape `(K, 67)`

Missing landmarks are represented by zero coordinates and marked false in the validity mask.

Pose files are stored under `data/pose_landmarks/`. Extraction reports are stored under `data/pose_reports/`.

## 5. Landmark Temporal Image Generation

The LTI generator converts pose landmark sequences into 224 x 224 x 3 float32 arrays.

Processing steps:

1. Loads the landmark sequence and validity mask.
2. Retains frames containing at least one detected left- or right-hand landmark.
3. Uniformly subsamples sequences exceeding the raster capacity.
4. Repeats each retained frame's landmark block consecutively.
5. Zero-pads the remaining rows.
6. Reshapes the result in row-major order into a 224 x 224 x 3 image.

The three channels represent x, y and z coordinates.

The raster contains 50,176 rows of three coordinates. With 67 landmarks per frame, its maximum frame capacity is 748.

### Implementation policies

The paper specifies the LTI representation but does not fully specify every edge-case handling decision. This implementation explicitly uses:

- Removal of frames without detected hand landmarks.
- Uniform temporal subsampling when more than 748 frames remain.
- Original MediaPipe coordinate values without additional coordinate scaling.
- Consecutive repetition of each frame's landmark block.
- Zero-padding of unused raster rows.

These are implementation choices and should not be interpreted as additional procedures explicitly specified by the paper.

Output shape: `(224, 224, 3)`

Output dtype: `float32`

Output directory: `data/lti_images/aslcitizen100/` or `data/lti_images/aslcitizen300/`

## 6. Running the Pipeline

Run commands from the repository root. Replace the example dataset path with the location of your authorized ASL Citizen dataset.

### Step 1: Build manifests

```cmd
python scripts\build_aslcitizen_manifests.py --dataset-root "D:\datasets\ASL_Citizen"
### Missing-modality handling

Some videos may not produce valid LTI images if no frame contains detected hand landmarks. Such samples are recorded as failures in the LTI generation report and do not receive an LTI output.

For ASL Citizen100 training data, one of the 1,800 manifest entries currently has this issue, leaving 1,799 valid LTI files. Training and evaluation using paired appearance and pose modalities must account for missing samples. Do not silently treat a missing LTI file as a valid sample.
