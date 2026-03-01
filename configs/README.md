# Configuration Files

YAML configuration files for experiments, models, datasets, and training.

## Structure

- `dataset/` - Dataset configurations
  - `mimic.yaml` - MIMIC-IV-ECG config
  - `ptbxl.yaml` - PTB-XL config
- `model/` - Model configurations by category
  - `vae/` - VAE model configs
  - `gan/` - GAN model configs
  - `diffusion/` - Diffusion model configs (future)
  - `ssm/` - SSM configs (future)
- `training/` - Training configurations
  - `default.yaml` - Default training settings
  - `fast_dev.yaml` - Quick development runs
  - `production.yaml` - Full production training
- `experiments/` - Complete experiment configurations
  - `vae_mimic.yaml` - VAE on MIMIC
  - `pulse2pulse_mimic.yaml` - Pulse2Pulse on MIMIC

## Usage

### Using Experiment Configs

```bash
python scripts/train/train_vae_mimic.py --config configs/experiments/vae_mimic.yaml
```

### Composing Configs

Combine multiple config files:

```bash
python train.py \
    --config configs/model/vae/vae_1d_base.yaml \
    --config configs/dataset/mimic.yaml \
    --config configs/training/default.yaml
```

### Overriding Parameters

Override specific parameters:

```bash
python train.py --config configs/experiments/vae_mimic.yaml \
    --batch_size 64 \
    --lr 0.001
```

## Config Format

### Dataset Config

```yaml
# configs/dataset/mimic.yaml
dataset:
  name: mimic
  data_dir: data/raw/mimic
  num_leads: 12
  seq_length: 5000
  batch_size: 32
  num_workers: 4
```

### Model Config

```yaml
# configs/model/vae/vae_1d_base.yaml
model:
  type: vae
  in_channels: 12
  base_channels: 64
  latent_channels: 8
  channel_multipliers: [1, 2, 4, 4]
  num_res_blocks: 2
```

### Training Config

```yaml
# configs/training/default.yaml
training:
  max_epochs: 100
  lr: 0.0001
  optimizer: adam
  scheduler: cosine
  early_stopping: true
  patience: 10
```

## Best Practices

1. **Modular configs**: Keep dataset, model, and training configs separate
2. **Experiment configs**: Create complete experiment configs for reproducibility
3. **Version control**: Track config changes in git
4. **Documentation**: Add comments to explain non-obvious parameters
5. **Validation**: Validate configs before training
