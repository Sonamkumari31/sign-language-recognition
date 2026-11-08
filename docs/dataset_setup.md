
# ASL Citizen Dataset Setup

## 1. Purpose

This guide explains how to configure a local ASL Citizen dataset for this project.

The dataset is not distributed with this repository. Researchers must obtain authorized access and comply with the applicable license.

## 2. Obtain the dataset

Visit the official Microsoft ASL Citizen project page:

https://www.microsoft.com/en-us/research/project/asl-citizen/

Follow the access instructions and read the included license terms before using the dataset.

**Do not upload dataset videos, manifests, class lists, or derived dataset artifacts to this public repository.**

## 3. Expected dataset structure

```text
ASL_Citizen/
|-- splits/
|   |-- train.csv
|   |-- val.csv
|   `-- test.csv
|-- videos/
`-- use.txt
```

The exact directory structure may vary depending on how the archive is extracted. The project needs access to the official split CSV files and video directory.

## 4. Dataset location

Set this to the directory containing your extracted ASL_Citizen dataset:

`D:\datasets\ASL_Citizen`

Pass your own dataset directory to preprocessing commands using the --dataset-root argument. Do not commit local dataset paths or data artifacts.

## 5. Official data splits

The project uses the official training, validation, and test splits:

- `train.csv`: training examples
- `val.csv`: validation examples
- `test.csv`: held-out test examples

Do not randomly re-split these examples. Preserve the official split assignments throughout preprocessing and evaluation.

## 6. ASL Citizen100 and ASL Citizen300

This implementation selects the top 100 or top 300 gloss classes by training-split frequency. Alphabetical order breaks ties.

The ASL Citizen100 class set is nested within ASL Citizen300.

This is the selection procedure used by this implementation. The resulting subsets and sample counts differ from those reported in the research paper, so exact reproduction of the paper's subsets is not claimed.

## 7. Generate local manifests

From the repository root, activate the project environment and run:

```bat
python scripts\build_aslcitizen_manifests.py --dataset-root "D:\datasets\ASL_Citizen"
```

The generated manifests and class lists are local artifacts. Keep them out of public version control.

## 8. Preprocessing checks

Before preprocessing, verify that:

- The dataset directory exists.
- All three official split CSV files are present.
- Referenced video files exist.
- The dataset license permits the intended research use.
- Dataset files and derived artifacts are excluded from Git.

## 9. License and redistribution

Do not redistribute the ASL Citizen dataset or restricted derived artifacts through this repository. Users must obtain their own authorized copy.

This guide is for technical setup and does not replace the dataset's license terms.
