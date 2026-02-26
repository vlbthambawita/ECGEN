## ECGGEN

Project scaffolding for ECG generation and modeling experiments.

### Repository structure

- **`pyproject.toml`**: Project dependencies and build configuration.
- **`LICENSE`**: Project license.
- **`README.md`**: High-level overview and repo structure (this file).
- **`scripts/`**: Helper shell scripts for running experiments.
  - **`run_train.sh`**: Entry point for training runs.
  - **`run_sweep.sh`**: Example script for running parameter sweeps.
  - **`export_predictions.sh`**: Script for exporting model predictions.
- **`src/my_project/`**: Main Python source code.
  - **`models/`**: Model architectures for ECG generation and related tasks (e.g. `cnn1d.py`, `imn.py`).
  - **`data/`**: Data loading and `LightningDataModule`-style abstractions (e.g. `datamodule.py`).
  - **`training/`**: Training, validation and testing entry points (`train.py`, `validate.py`, `test.py`).
  - **`utils/`**: Utility functions for I/O, logging and seeding (`io.py`, `logging.py`, `seed.py`).
- **`model_repos/Pulse2Pulse/`**: Vendorized Pulse2Pulse baseline implementation and sample ECG data.
  - **`pulse2pulse_train.py` / `pulse2pulse_train.sh`**: Original training entry points.
  - **`models/`**: Pulse2Pulse model definitions.
  - **`data/`**: Data loaders specific to the Pulse2Pulse code.
  - **`sample_ecg_data/`**: Small sample ECG traces used for quick tests.

### Usage

- **Install** (from the `ECGEN` directory):

```bash
pip install -e .
```

- **Run training**:

```bash
bash scripts/run_train.sh
```

Adjust script arguments or call the Python entry points in `src/my_project/training` directly to customize experiments.

