# MIMIC-IV-ECG Dataset

Documentation for using the MIMIC-IV-ECG dataset.

## Overview

MIMIC-IV-ECG is a large-scale ECG dataset with machine measurements.

## Dataset Structure

```
mimic_path/
├── files/
│   └── p{XXXX}/
│       └── p{subject_id}/
│           └── s{study_id}/
│               ├── {study_id}.hea  # Header file
│               └── {study_id}.dat  # Signal data
└── machine_measurements.csv
```

## Setup

### 1. Download Data

Follow MIMIC-IV-ECG access instructions at PhysioNet.

### 2. Organize Data

Place data in `data/raw/mimic/`:

```bash
data/raw/mimic/
├── files/
└── machine_measurements.csv
```

### 3. Prepare Data (Optional)

```bash
python scripts/data/prepare_mimic.py \
    --raw_dir data/raw/mimic \
    --output_dir data/processed/mimic
```

## Usage

### Basic Usage

```python
from data.datasets.mimic import MIMICIVECGDataset

dataset = MIMICIVECGDataset(
    mimic_path="data/raw/mimic",
    split="train",
    max_samples=1000,  # For testing
)

# Get a sample
sample = dataset[0]
ecg = sample["ecg_signals"]  # Shape: (12, 5000)
features = sample["features"]  # Shape: (9,)
```

### With DataModule

```python
from data.datasets.mimic.datamodule import MIMICDataModule

datamodule = MIMICDataModule(
    data_dir="data/raw/mimic",
    batch_size=32,
    num_workers=4,
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
```

## Features

The dataset provides 9 machine measurement features:
1. rr_interval
2. p_onset
3. p_end
4. qrs_onset
5. qrs_end
6. t_end
7. p_axis
8. qrs_axis
9. t_axis

## Data Format

- **ECG signals**: (num_leads, seq_length) = (12, 5000)
- **Sampling rate**: 500 Hz
- **Duration**: 10 seconds
- **Leads**: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

## Preprocessing

The dataset automatically:
- Normalizes ECG signals
- Handles missing values
- Splits into train/val/test

## Citation

If you use MIMIC-IV-ECG, please cite:

```
@article{mimic-iv-ecg,
  title={MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset},
  author={...},
  journal={PhysioNet},
  year={2023}
}
```
