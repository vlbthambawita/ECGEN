# MIMIC-IV-ECG Dataset

Dataset implementation for MIMIC-IV-ECG.

## Files

- `dataset.py` - MIMICIVECGDataset class
- `datamodule.py` - PyTorch Lightning DataModule
- `pulse2pulse_datamodule.py` - DataModule for Pulse2Pulse GAN

## Usage

```python
from data.datasets.mimic import MIMICIVECGDataset

dataset = MIMICIVECGDataset(
    mimic_path="/path/to/mimic",
    split="train",
)
```

## Dataset Structure

The MIMIC-IV-ECG dataset should be organized as:
```
mimic_path/
├── files/
│   └── p{XXXX}/
│       └── p{subject_id}/
│           └── s{study_id}/
│               ├── {study_id}.hea
│               └── {study_id}.dat
└── machine_measurements.csv
```
