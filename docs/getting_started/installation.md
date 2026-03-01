# Installation

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- PyTorch Lightning
- WFDB for ECG data reading

## Installation Steps

### 1. Clone the repository

```bash
git clone <repository-url>
cd ECGEN
```

### 2. Install dependencies

```bash
pip install -e .
```

Or install from pyproject.toml:

```bash
pip install torch pytorch-lightning pyyaml pandas scikit-learn wfdb matplotlib
```

### 3. Verify installation

```bash
python -c "import models; import data; import training; print('Installation successful!')"
```

## Development Installation

For development, install additional dependencies:

```bash
pip install pytest black flake8 mypy
```

## Dataset Setup

See [MIMIC-IV-ECG Dataset](../datasets/mimic.md) for dataset setup instructions.
