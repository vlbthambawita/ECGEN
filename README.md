# ECGEN - ECG Generation Models

A comprehensive framework for ECG generation using various deep learning approaches including VAE, GAN, Diffusion, and State Space Models.

## Features

- **Multiple Model Categories**
  - VAE (Variational Autoencoders)
  - GAN (Generative Adversarial Networks)
  - Diffusion Models (coming soon)
  - State Space Models (coming soon)

- **Scalable Dataset Support**
  - MIMIC-IV-ECG
  - PTB-XL (coming soon)
  - Easy to add custom datasets

- **Comprehensive Tools**
  - Training utilities and callbacks
  - Evaluation metrics (signal quality, diversity, fidelity)
  - Visualization tools
  - Benchmarking utilities

## Project Structure

```
ECGEN/
├── data/                   # Data handling
│   ├── datasets/           # Dataset implementations
│   │   ├── mimic/          # MIMIC-IV-ECG
│   │   └── base.py         # Base dataset class
│   ├── raw/                # Raw data (gitignored)
│   └── processed/          # Processed data (gitignored)
├── models/                 # Model architectures
│   ├── vae/                # VAE models
│   ├── gan/                # GAN models
│   ├── diffusion/          # Diffusion models (future)
│   ├── ssm/                # State Space Models (future)
│   └── components/         # Shared components
├── training/               # Training utilities
│   ├── callbacks/          # Training callbacks
│   ├── losses/             # Loss functions
│   ├── metrics/            # Training metrics
│   └── trainers/           # Training loops
├── evaluation/             # Evaluation tools
│   ├── metrics/            # Evaluation metrics
│   ├── visualize.py        # Evaluation visualization
│   └── benchmark.py        # Benchmarking
├── visualization/          # Visualization tools
│   ├── ecg_plots.py        # ECG plotting
│   ├── latent_space.py     # Latent space visualization
│   └── training_curves.py  # Training progress
├── scripts/                # Executable scripts
│   ├── train/              # Training scripts
│   ├── generate/           # Generation scripts
│   ├── evaluate/           # Evaluation scripts
│   ├── data/               # Data processing
│   └── shell/              # Shell wrappers
├── configs/                # Configuration files
│   ├── dataset/            # Dataset configs
│   ├── model/              # Model configs
│   ├── training/           # Training configs
│   └── experiments/        # Experiment configs
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── outputs/                # Training outputs (gitignored)
└── notebooks/              # Jupyter notebooks
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ECGEN

# Install dependencies
pip install -e .
```

### Train VAE on MIMIC-IV-ECG

```bash
# Configure experiment
nano configs/experiments/vae_mimic.yaml

# Train model
./scripts/shell/run_train_vae_mimic_config.sh
```

### Generate Samples

```python
from models.vae import VAELightning

# Load trained model
model = VAELightning.load_from_checkpoint("outputs/vae_mimic/checkpoints/best.ckpt")

# Generate samples
samples = model.sample(n_samples=16, seq_length=5000)
```

## Model Categories

### VAE (Variational Autoencoders)
- Learn compressed latent representations
- Reconstruct ECG signals
- Generate new samples from latent space

### GAN (Generative Adversarial Networks)
- Pulse2Pulse model for ECG generation
- High-quality synthetic ECG generation

### Diffusion Models (Coming Soon)
- State-of-the-art generation quality
- Iterative denoising process

### State Space Models (Coming Soon)
- Efficient long-sequence modeling
- Mamba, S4 architectures

## Datasets

### MIMIC-IV-ECG
Large-scale ECG dataset with 12-lead recordings.

See [docs/datasets/mimic.md](docs/datasets/mimic.md) for setup instructions.

### Adding New Datasets
1. Create folder in `data/datasets/{dataset_name}/`
2. Implement dataset class inheriting from `BaseECGDataset`
3. Create DataModule for PyTorch Lightning
4. Add configuration in `configs/dataset/`

## Documentation

- [Installation Guide](docs/getting_started/installation.md)
- [Quick Start](docs/getting_started/quickstart.md)
- [Training Guide](docs/training/guide.md)
- [Model Documentation](docs/models/)
- [API Documentation](docs/api/)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black .
flake8 .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ecgen2026,
  title={ECGEN: ECG Generation Models},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ECGEN}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- MIMIC-IV-ECG dataset from PhysioNet
- PyTorch Lightning framework
- Research community contributions
