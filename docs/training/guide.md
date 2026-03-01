# Training Guide

General guide for training models in ECGEN.

## Quick Start

### 1. Prepare Data

```bash
python scripts/data/prepare_mimic.py --raw_dir /path/to/raw --output_dir data/processed/mimic
```

### 2. Configure Experiment

Edit configuration file in `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
model:
  type: vae
  in_channels: 12
  latent_channels: 8

dataset:
  name: mimic
  data_dir: data/processed/mimic

training:
  batch_size: 32
  epochs: 100
  lr: 0.0001
```

### 3. Train Model

```bash
./scripts/shell/run_train_vae_mimic_config.sh
```

## Training Components

### Callbacks

Located in `training/callbacks/`:
- Visualization callbacks
- Checkpointing callbacks
- Logging callbacks

### Losses

Located in `training/losses/`:
- VAE losses
- GAN losses
- Common losses

### Metrics

Located in `training/metrics/`:
- Reconstruction metrics
- Generation metrics
- Clinical metrics

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/{experiment_name}/
```

### Weights & Biases

Configure in your training script:

```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="ecgen", name="my_experiment")
```

## Best Practices

1. **Start small**: Test with small dataset first
2. **Monitor metrics**: Watch for overfitting
3. **Save checkpoints**: Regular checkpoint saving
4. **Use callbacks**: Leverage visualization callbacks
5. **Document experiments**: Keep notes on configurations

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce model size
- Use gradient accumulation

### Slow Training
- Increase num_workers for data loading
- Use mixed precision training
- Profile bottlenecks

### Poor Results
- Check data preprocessing
- Adjust learning rate
- Try different architectures
- Increase training time
