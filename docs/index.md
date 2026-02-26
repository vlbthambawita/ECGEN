# ECGEN Documentation

Welcome to the ECGEN documentation! ECGEN is a comprehensive framework for ECG (Electrocardiogram) generation and modeling experiments.

## Overview

ECGEN provides a modular and extensible platform for:

- **ECG Generation**: State-of-the-art generative models for synthesizing realistic ECG signals
- **Pulse2Pulse Integration**: WaveGAN-based architecture for high-quality 8-lead ECG generation
- **PyTorch Lightning**: Modern training framework with automatic GPU handling and checkpointing
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **MIMIC-IV-ECG Support**: Built-in data loaders for the MIMIC-IV-ECG dataset

## Key Features

- 🚀 **Easy to Use**: Get started with a single command
- 🔧 **Highly Configurable**: YAML-based configuration system
- 📊 **Comprehensive Monitoring**: TensorBoard and W&B integration
- 🎯 **Production Ready**: Modular design with type hints and documentation
- 🔬 **Research Focused**: Built for experimentation and reproducibility

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get ECGEN up and running
- [Quick Start](getting-started/quickstart.md) - Train your first model in minutes
- [Pulse2Pulse Training](user-guide/pulse2pulse/training.md) - Detailed training guide
- [API Reference](reference/index.md) - Complete API documentation

## Project Structure

```
ECGEN/
├── src/ecgen/              # Main source code
│   ├── models/             # Model architectures
│   ├── data/               # Data loading modules
│   ├── training/           # Training utilities
│   └── utils/              # Helper functions
├── scripts/                # Training and generation scripts
├── configs/                # Configuration files
├── docs/                   # Documentation (you are here!)
└── tests/                  # Unit tests
```

## Getting Started

The fastest way to get started:

```bash
# Install ECGEN
pip install -e .

# Install documentation dependencies (optional)
pip install -e ".[docs]"

# Run a quick test
python scripts/test_models_only.py

# Start training
./scripts/run_train_pulse2pulse_mimic.sh
```

## What's Next?

- **New Users**: Start with the [Installation Guide](getting-started/installation.md)
- **Quick Training**: Jump to the [Quick Start](getting-started/quickstart.md)
- **Advanced Usage**: Explore the [User Guide](user-guide/pulse2pulse/overview.md)
- **API Details**: Check the [API Reference](reference/)

## Support

- **Documentation**: You're reading it!
- **Issues**: Report bugs and request features on GitHub
- **Changelog**: See [Changelog](development/changelog.md) for version history

## License

ECGEN is released under the MIT License. See the LICENSE file for details.

---

**Ready to generate some ECGs?** Let's get started! 🚀
