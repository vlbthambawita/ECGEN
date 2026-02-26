# Installation

This guide will help you install ECGEN and all its dependencies.

## Prerequisites

Before installing ECGEN, ensure you have:

- **Python**: Version 3.9 or higher
- **pip**: Latest version recommended
- **CUDA** (optional): For GPU acceleration
- **Git**: For cloning the repository

## Installation Methods

### Method 1: Development Installation (Recommended)

For development and experimentation:

```bash
# Clone the repository
git clone https://github.com/yourusername/ECGEN.git
cd ECGEN

# Install in editable mode
pip install -e .
```

This installs ECGEN in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

### Method 2: Standard Installation

For production use:

```bash
# Clone the repository
git clone https://github.com/yourusername/ECGEN.git
cd ECGEN

# Install normally
pip install .
```

### Method 3: From PyPI (Coming Soon)

Once published to PyPI:

```bash
pip install ecggen
```

## Optional Dependencies

### Documentation Tools

To build and serve documentation locally:

```bash
pip install -e ".[docs]"
```

This installs:
- MkDocs with Material theme
- API documentation generators
- All necessary plugins

### Development Tools

For contributing to ECGEN:

```bash
pip install -e ".[dev]"
```

This includes:
- Testing frameworks (pytest)
- Code formatters (black, isort)
- Linters (flake8, mypy)

## Verifying Installation

After installation, verify everything works:

```bash
# Test imports
python -c "import ecgen; print(ecgen.__version__)"

# Run model tests
python scripts/test_models_only.py
```

Expected output:
```
✓ ALL TESTS PASSED!
```

## GPU Setup

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

### MIMIC-IV-ECG Dataset

ECGEN works with the MIMIC-IV-ECG dataset from PhysioNet:

1. **Request Access**: Visit [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/)
2. **Complete Training**: Complete required CITI training
3. **Download Data**: Download the dataset after approval
4. **Set Path**: Update data paths in configs or scripts

Example directory structure:
```
/path/to/MIMIC-IV-ECG/
├── files/
│   ├── p100/
│   ├── p101/
│   └── ...
└── record_list.csv
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ecgen'`

**Solution**: Ensure you installed in editable mode from the ECGEN directory:
```bash
cd /path/to/ECGEN
pip install -e .
```

#### CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Check CUDA toolkit installation

#### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:
- Reduce batch size in config
- Use gradient accumulation
- Close other GPU processes

#### Jinja2 Errors

**Problem**: `ImportError: cannot import name 'pass_context' from 'jinja2'`

**Solution**: Upgrade jinja2:
```bash
pip install --upgrade jinja2
```

## Next Steps

Now that ECGEN is installed:

- [Quick Start Guide](quickstart.md) - Train your first model
- [Pulse2Pulse Overview](../user-guide/pulse2pulse/overview.md) - Learn about the models
- [Configuration Guide](../user-guide/pulse2pulse/configuration.md) - Customize your experiments

## System Requirements

### Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 16 GB
- **Storage**: 50 GB (for dataset and outputs)
- **GPU**: Not required but highly recommended

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 32 GB
- **Storage**: 200 GB SSD
- **GPU**: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, V100)

## Dependencies

Core dependencies installed automatically:

- `torch` - PyTorch deep learning framework
- `pytorch-lightning` - High-level training framework
- `pyyaml` - Configuration file parsing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `wfdb` - ECG data format handling

For a complete list, see `pyproject.toml`.
