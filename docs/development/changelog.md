# Changelog

All notable changes to ECGEN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MkDocs documentation with Material theme
- Comprehensive API reference documentation
- GitHub Actions for automatic documentation deployment

## [0.1.0] - 2024-02-26

### Added
- Initial Pulse2Pulse WaveGAN integration
- PyTorch Lightning training framework
- MIMIC-IV-ECG data module
- WGAN-GP training algorithm
- ECG visualization callbacks
- TensorBoard logging support
- Weights & Biases integration
- Config-based training system
- Automatic checkpointing
- Sample generation during training

### Model Components
- `WaveGANGenerator`: U-Net style generator (~10.5M parameters)
- `WaveGANDiscriminator`: Convolutional discriminator (~334M parameters)
- `Pulse2PulseGAN`: PyTorch Lightning module wrapper

### Data Components
- `Pulse2PulseMIMICDataModule`: LightningDataModule for MIMIC-IV-ECG
- `ECGDatasetAdapter`: Adapter for ECG-only data loading
- Automatic train/val/test splits
- 8-lead ECG extraction from 12-lead data

### Training Components
- `train.py`: Main training script with config support
- `train_pulse2pulse.py`: Standalone training script
- `callbacks.py`: Visualization and sample generation callbacks
- `losses.py`: WGAN-GP loss functions
- `metrics.py`: Evaluation metrics

### Scripts
- `run_train_pulse2pulse_mimic.sh`: Config-based training
- `run_train_pulse2pulse_standalone.sh`: Standalone training
- `train_pulse2pulse.py`: Direct Python training
- `generate_pulse2pulse.py`: Sample generation
- `test_models_only.py`: Model architecture tests

### Documentation
- `README.md`: Project overview
- `README_PULSE2PULSE.md`: Pulse2Pulse integration guide
- `QUICKSTART_PULSE2PULSE.md`: Quick start guide
- `WANDB_QUICKSTART.md`: W&B setup guide
- `WANDB_INTEGRATION_SUMMARY.md`: W&B integration details
- `CHANGELOG_PULSE2PULSE.md`: Integration changelog
- `docs/pulse2pulse_training.md`: Detailed training guide
- `docs/wandb_setup.md`: W&B configuration guide

### Configuration
- `configs/experiments/pulse2pulse_mimic.yaml`: Default experiment config
- YAML-based configuration system
- Command-line config overrides
- Environment variable support

### Features
- Automatic GPU detection and usage
- Multi-GPU training support
- Gradient accumulation
- Mixed precision training (optional)
- Automatic checkpoint saving (top-K + last)
- Real vs. fake ECG comparison plots
- Generated sample visualization
- TensorBoard metric logging
- W&B experiment tracking
- Resume training from checkpoint

## Pulse2Pulse Integration Details

### Changes from Original Implementation

#### Improvements
1. **PyTorch Lightning Integration**
   - Automatic GPU handling
   - Built-in checkpointing
   - TensorBoard logging
   - Cleaner training loop

2. **Modular Design**
   - Separate model, data, training, and callback modules
   - Config-based training
   - Reusable components

3. **Better Visualization**
   - Automatic sample generation during training
   - Real vs. fake comparison plots
   - Saved sample tensors for analysis

4. **Improved Data Handling**
   - LightningDataModule for data loading
   - Automatic train/val/test splits
   - Better error handling

5. **Enhanced Checkpointing**
   - Save top-K checkpoints based on validation metric
   - Automatic last checkpoint saving
   - Easy resume from checkpoint

#### Preserved Features
- Same model architecture (WaveGANGenerator, WaveGANDiscriminator)
- Same training algorithm (WGAN-GP)
- Same hyperparameters (by default)
- Compatible with MIMIC-IV-ECG dataset

### Known Issues
- None reported

### Deprecations
- None

## [0.0.1] - Initial Scaffolding

### Added
- Project structure
- Basic dependencies
- License
- Initial README

---

## Release Notes

### Version 0.1.0

This is the first major release of ECGEN, featuring a complete Pulse2Pulse WaveGAN implementation with PyTorch Lightning support.

**Highlights:**
- 🎉 Complete Pulse2Pulse integration
- ⚡ PyTorch Lightning for easy training
- 📊 TensorBoard and W&B support
- 🎨 Automatic visualization
- 📝 Comprehensive documentation

**Breaking Changes:**
- None (initial release)

**Migration Guide:**
- Not applicable (initial release)

**Contributors:**
- ECGEN Team

---

## Future Plans

### Planned Features
- [ ] Support for 12-lead ECGs
- [ ] Additional generative models (VAE, Diffusion)
- [ ] Pre-trained model weights
- [ ] Evaluation metrics (FID, IS, etc.)
- [ ] Data augmentation techniques
- [ ] Multi-dataset support
- [ ] Model compression techniques
- [ ] Real-time generation API

### Under Consideration
- [ ] Conditional generation (by diagnosis)
- [ ] Style transfer between ECGs
- [ ] Anomaly detection
- [ ] ECG segmentation
- [ ] Beat-level generation

---

For more details on specific changes, see the individual documentation files:
- [Pulse2Pulse Changelog](../CHANGELOG_PULSE2PULSE.md)
- [W&B Integration Summary](../WANDB_INTEGRATION_SUMMARY.md)
