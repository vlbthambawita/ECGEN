# Data

Data handling for ECGEN project.

## Structure

- `raw/` - Raw datasets (gitignored)
- `processed/` - Preprocessed data (gitignored)
- `datasets/` - Dataset implementations
  - `base.py` - Base dataset class
  - `mimic/` - MIMIC-IV-ECG dataset
  - [future datasets] - PTB-XL, CinC, etc.

## Adding New Datasets

1. Create a new folder under `datasets/`
2. Implement your dataset class inheriting from `BaseECGDataset`
3. Create a DataModule for PyTorch Lightning
4. Add a README.md with usage instructions

## Dataset Organization

Raw data should be stored in `data/raw/{dataset_name}/`
Preprocessed data should be stored in `data/processed/{dataset_name}/`
