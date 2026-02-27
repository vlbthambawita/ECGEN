# ECGEN Project Structure

## Overview

This document describes the logical organization of the ECGEN project.

---

## Directory Structure

```
DL2026/ECGEN/
│
├── src/ecgen/                          # Main package (installable)
│   ├── __init__.py
│   │
│   ├── data/                           # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── mimic_dataset.py           # MIMIC-IV-ECG dataset
│   │   └── transforms.py              # Data transformations
│   │
│   ├── models/                         # Model architectures
│   │   ├── __init__.py
│   │   ├── vae.py                     # VAE model
│   │   ├── pulse2pulse.py             # Pulse2Pulse GAN
│   │   └── components.py              # Shared components
│   │
│   ├── training/                       # Training utilities ⭐
│   │   ├── __init__.py                # Exports all callbacks
│   │   ├── callbacks.py               # ALL CALLBACKS (422 lines)
│   │   │   ├── ECGVisualizationCallback
│   │   │   ├── GeneratedSamplesCallback
│   │   │   └── VAEVisualizationCallback
│   │   ├── losses.py                  # Loss functions
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── train.py                   # Training loops
│   │   ├── test.py                    # Testing utilities
│   │   └── validate.py                # Validation utilities
│   │
│   ├── evaluation/                     # Evaluation tools
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       ├── seed.py                    # Random seed utilities
│       └── visualization.py           # General visualization
│
├── scripts/                            # Executable scripts
│   ├── train_vae_mimic.py             # VAE training script
│   ├── train_pulse2pulse.py           # Pulse2Pulse training
│   ├── generate_pulse2pulse.py        # Sample generation
│   ├── run_train_vae_mimic_config.sh  # Shell wrapper for VAE
│   ├── run_train_pulse2pulse_mimic.sh # Shell wrapper for P2P
│   └── README_VISUALIZATION.md        # Quick reference
│
├── configs/                            # Configuration files
│   ├── dataset/                       # Dataset configs
│   ├── experiments/                   # Experiment configs
│   │   └── vae_mimic.yaml            # VAE training config
│   ├── model/                         # Model configs
│   └── trainer/                       # Trainer configs
│
├── docs/                               # Documentation
│   ├── VAE_VISUALIZATION.md           # Visualization guide
│   └── ...
│
├── tests/                              # Unit tests
│   └── ...
│
├── runs/                               # Training outputs (gitignored)
│   └── {experiment_name}/
│       └── seed_{seed}/
│           ├── checkpoints/           # Model checkpoints
│           ├── samples/               # Visualization samples
│           └── tb/                    # TensorBoard logs
│
├── REORGANIZATION_SUMMARY.md          # This reorganization
├── VISUALIZATION_SUMMARY.md           # Visualization feature
├── CHANGES.md                         # Detailed changes
├── README_VAE_VISUALIZATION.md        # Main README
├── PROJECT_STRUCTURE.md               # This file
│
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
└── README.md                          # Project README
```

---

## Module Responsibilities

### 📦 src/ecgen/ (Package)

The main Python package containing all reusable code.

#### data/
- Dataset classes (MIMIC-IV-ECG, etc.)
- Data loading and preprocessing
- Transforms and augmentations

#### models/
- Model architectures (VAE, Pulse2Pulse, etc.)
- Model components (encoders, decoders, etc.)
- Model configurations

#### training/ ⭐
- **callbacks.py** - All training callbacks
  - `ECGVisualizationCallback` - Visualize GAN outputs
  - `GeneratedSamplesCallback` - Save generated samples
  - `VAEVisualizationCallback` - Visualize VAE reconstructions
- **losses.py** - Loss functions
- **metrics.py** - Evaluation metrics
- **train.py** - Training loops
- **test.py** - Testing utilities
- **validate.py** - Validation utilities

#### evaluation/
- Evaluation metrics
- Performance analysis tools

#### utils/
- Random seed management
- General utilities
- Helper functions

### 📜 scripts/

Executable scripts that use the package.

- **Training scripts** - `train_*.py`
- **Generation scripts** - `generate_*.py`
- **Shell wrappers** - `run_*.sh`
- **Documentation** - `README_*.md`

### ⚙️ configs/

YAML configuration files for experiments.

- **dataset/** - Dataset configurations
- **experiments/** - Full experiment configs
- **model/** - Model hyperparameters
- **trainer/** - Training parameters

### 📚 docs/

Project documentation.

### 🧪 tests/

Unit tests for the package.

### 📊 runs/

Training outputs (gitignored).

---

## Import Patterns

### ✅ Correct Imports

```python
# Import callbacks from the package
from ecgen.training.callbacks import VAEVisualizationCallback
from ecgen.training.callbacks import ECGVisualizationCallback
from ecgen.training.callbacks import GeneratedSamplesCallback

# Or import all at once
from ecgen.training.callbacks import (
    VAEVisualizationCallback,
    ECGVisualizationCallback,
    GeneratedSamplesCallback,
)

# Or use the training module __init__
from ecgen.training import VAEVisualizationCallback

# Import models
from ecgen.models.vae import VAE1D, VAEConfig

# Import data
from ecgen.data.mimic_dataset import MIMICIVECGDataset
```

### ❌ Incorrect Imports (Old)

```python
# DON'T DO THIS - scripts/callbacks/ no longer exists
from callbacks.vae_visualization import VAEVisualizationCallback
```

---

## Callback Locations

### All Callbacks in One Place

**File**: `src/ecgen/training/callbacks.py` (422 lines)

**Contains**:
1. **ECGVisualizationCallback** (lines 20-97)
   - For GAN models (Pulse2Pulse)
   - Visualizes real vs generated ECG
   - Used by: `train_pulse2pulse.py`

2. **GeneratedSamplesCallback** (lines 99-168)
   - Saves generated samples periodically
   - Creates plots and saves tensors
   - Used by: `train_pulse2pulse.py`

3. **VAEVisualizationCallback** (lines 171-422)
   - For VAE models
   - Visualizes real vs reconstructed ECG
   - Two modes: separate leads or overlay
   - Used by: `train_vae_mimic.py`

---

## Training Workflows

### VAE Training

```bash
# 1. Configure
nano configs/experiments/vae_mimic.yaml

# 2. Train
./scripts/run_train_vae_mimic_config.sh

# 3. Monitor
tensorboard --logdir runs/vae_mimic/seed_42/tb

# 4. Check visualizations
ls runs/vae_mimic/seed_42/samples/
```

**Uses**: `VAEVisualizationCallback`

### Pulse2Pulse Training

```bash
# 1. Train
./scripts/run_train_pulse2pulse_mimic.sh

# 2. Monitor
tensorboard --logdir runs/pulse2pulse/seed_42/tb
```

**Uses**: `ECGVisualizationCallback`, `GeneratedSamplesCallback`

---

## Key Design Principles

### 1. Separation of Concerns
- **Package** (`src/ecgen/`) - Reusable code
- **Scripts** (`scripts/`) - Executable entry points
- **Configs** (`configs/`) - Experiment parameters

### 2. Single Source of Truth
- All callbacks in one file: `src/ecgen/training/callbacks.py`
- No duplicate code
- Easy to maintain

### 3. Clean Imports
- Standard Python package imports
- No `sys.path` manipulation needed
- IDE autocomplete works

### 4. Logical Grouping
- Training utilities together
- Models together
- Data loading together

---

## Adding New Components

### Adding a New Callback

1. Open `src/ecgen/training/callbacks.py`
2. Add your callback class
3. Update `src/ecgen/training/__init__.py` to export it
4. Use it in your training script:
   ```python
   from ecgen.training.callbacks import YourNewCallback
   ```

### Adding a New Model

1. Create file in `src/ecgen/models/`
2. Implement your model
3. Import in training script:
   ```python
   from ecgen.models.your_model import YourModel
   ```

### Adding a New Dataset

1. Create file in `src/ecgen/data/`
2. Implement dataset class
3. Import in training script:
   ```python
   from ecgen.data.your_dataset import YourDataset
   ```

---

## Benefits of This Structure

### ✅ Maintainability
- Clear organization
- Easy to find code
- Logical grouping

### ✅ Reusability
- Package can be installed
- Code can be imported anywhere
- Callbacks work in notebooks

### ✅ Scalability
- Easy to add new components
- No naming conflicts
- Clear dependencies

### ✅ Best Practices
- Follows Python conventions
- Standard package structure
- Professional organization

---

## Quick Reference

### Where is...?

| Component | Location |
|-----------|----------|
| Callbacks | `src/ecgen/training/callbacks.py` |
| VAE model | `src/ecgen/models/vae.py` |
| Pulse2Pulse model | `src/ecgen/models/pulse2pulse.py` |
| MIMIC dataset | `src/ecgen/data/mimic_dataset.py` |
| Training scripts | `scripts/train_*.py` |
| Configs | `configs/experiments/*.yaml` |
| Documentation | `docs/*.md` |
| Outputs | `runs/{experiment}/seed_{seed}/` |

### How do I...?

| Task | Command |
|------|---------|
| Train VAE | `./scripts/run_train_vae_mimic_config.sh` |
| Train Pulse2Pulse | `./scripts/run_train_pulse2pulse_mimic.sh` |
| View TensorBoard | `tensorboard --logdir runs/{exp}/seed_{seed}/tb` |
| Check visualizations | `ls runs/{exp}/seed_{seed}/samples/` |
| Import callback | `from ecgen.training.callbacks import VAEVisualizationCallback` |
| Edit config | `nano configs/experiments/vae_mimic.yaml` |

---

**Last Updated**: 2026-02-27  
**Status**: ✅ Organized and documented
