# Weights & Biases (wandb) Integration

This guide explains how to use Weights & Biases for experiment tracking with Pulse2Pulse training.

## Setup

### 1. Install wandb

```bash
pip install wandb
```

### 2. Login to wandb

```bash
wandb login
```

This will prompt you for your API key. Get it from: https://wandb.ai/authorize

Alternatively, set the API key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

## Usage

### Method 1: Config-Based Training

Edit `configs/experiments/pulse2pulse_mimic.yaml`:

```yaml
wandb:
  enabled: true                          # Enable wandb logging
  project: ecg-pulse2pulse               # Your wandb project name
  entity: your-username                  # Your wandb username/team (optional)
  run_name: null                         # Auto-generated if null
  tags:
    - pulse2pulse
    - wavegan
    - ecg-generation
  notes: "Pulse2Pulse WaveGAN training on MIMIC-IV-ECG"
  log_model: true                        # Log model checkpoints to wandb
  log_freq: 50                           # Log frequency (steps)
```

Then run:

```bash
./scripts/run_train_pulse2pulse_mimic.sh
```

### Method 2: Standalone Script

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse \
    --wandb-entity your-username \
    --wandb-run-name my_experiment \
    --wandb-tags pulse2pulse wavegan experiment1
```

### Method 3: Environment Variables

```bash
# Enable wandb in config, then:
WANDB_PROJECT=ecg-pulse2pulse \
WANDB_ENTITY=your-username \
./scripts/run_train_pulse2pulse_mimic.sh
```

## What Gets Logged

### Metrics

- **Training metrics** (every step):
  - `train/d_loss`: Discriminator loss
  - `train/d_wasserstein`: Wasserstein distance
  - `train/g_loss`: Generator loss

- **Validation metrics** (every validation epoch):
  - `val/d_wasserstein`: Validation Wasserstein distance
  - `val_loss`: Validation loss

### Media

- **ECG Samples** (configurable frequency):
  - Real vs. fake comparison plots
  - Generated ECG samples
  - Logged as images with captions

### Model

- **Checkpoints** (if `log_model: true`):
  - Best model checkpoint
  - Last model checkpoint
  - Linked to wandb artifacts

### Configuration

- **Hyperparameters**:
  - Model config (model_size, num_channels, etc.)
  - Training config (batch_size, lr, etc.)
  - Data config (data_dir, max_samples, etc.)

### System

- **Hardware info**:
  - GPU type and count
  - CPU info
  - Memory usage

- **Code**:
  - Git commit hash (if in git repo)
  - Code diff
  - Requirements

## Viewing Results

### Web Interface

After starting training, you'll see:

```
Weights & Biases logging enabled: ecg-pulse2pulse/pulse2pulse_mimic_seed42
View run at: https://wandb.ai/your-username/ecg-pulse2pulse/runs/xxxxx
```

Click the link to view your run in real-time.

### Dashboard Features

1. **Charts**: Real-time metric plots
2. **System**: GPU/CPU utilization
3. **Logs**: Training logs
4. **Files**: Saved files and checkpoints
5. **Artifacts**: Model checkpoints
6. **Media**: Generated ECG samples

## Advanced Usage

### Custom Tags

Add tags to organize experiments:

```yaml
wandb:
  tags:
    - pulse2pulse
    - wavegan
    - experiment-v2
    - high-lr
```

### Run Notes

Add notes to describe the experiment:

```yaml
wandb:
  notes: |
    Testing higher learning rate (2e-4) with larger model size (100).
    Hypothesis: Better convergence with more capacity.
```

### Group Runs

Group related runs together:

```yaml
wandb:
  group: "lr-sweep"
  job_type: "train"
```

### Sweep (Hyperparameter Tuning)

Create a sweep config `configs/sweeps/pulse2pulse_sweep.yaml`:

```yaml
program: scripts/train_pulse2pulse.py
method: bayes
metric:
  name: val/d_wasserstein
  goal: maximize
parameters:
  lr:
    min: 0.00001
    max: 0.001
  model_size:
    values: [50, 75, 100]
  n_critic:
    values: [3, 5, 10]
```

Run sweep:

```bash
wandb sweep configs/sweeps/pulse2pulse_sweep.yaml
wandb agent your-username/ecg-pulse2pulse/sweep-id
```

## Disable wandb

### Temporarily Disable

Set in config:

```yaml
wandb:
  enabled: false
```

Or use environment variable:

```bash
WANDB_MODE=disabled ./scripts/run_train_pulse2pulse_mimic.sh
```

### Permanently Disable

Remove or comment out the `wandb` section in your config.

## Offline Mode

Train without internet connection:

```bash
WANDB_MODE=offline ./scripts/run_train_pulse2pulse_mimic.sh
```

Sync later:

```bash
wandb sync runs/pulse2pulse_mimic/seed_42/wandb/
```

## Tips and Best Practices

1. **Use descriptive run names**: Include key hyperparameters
   ```yaml
   run_name: "pulse2pulse_lr1e-4_bs128_ms50"
   ```

2. **Tag experiments**: Use tags to organize related runs
   ```yaml
   tags: ["baseline", "lr-sweep", "v1"]
   ```

3. **Add notes**: Document experiment goals and hypotheses
   ```yaml
   notes: "Testing impact of gradient penalty coefficient"
   ```

4. **Compare runs**: Use wandb's comparison tools to analyze multiple runs

5. **Share results**: Generate public links to share with collaborators

6. **Use artifacts**: Track model lineage and versioning

7. **Monitor in real-time**: Keep wandb dashboard open during training

## Troubleshooting

### Issue: wandb not installed

**Error**: `ImportError: No module named 'wandb'`

**Solution**:
```bash
pip install wandb
```

### Issue: Not logged in

**Error**: `wandb: ERROR Not logged in`

**Solution**:
```bash
wandb login
```

### Issue: Permission denied

**Error**: `wandb: ERROR Permission denied`

**Solution**: Check your wandb entity/username is correct

### Issue: Network error

**Error**: `wandb: ERROR Network error`

**Solution**: Use offline mode:
```bash
WANDB_MODE=offline ./scripts/run_train_pulse2pulse_mimic.sh
```

### Issue: Too many logs

**Solution**: Reduce log frequency:
```yaml
wandb:
  log_freq: 100  # Log every 100 steps instead of 50
```

## Examples

### Example 1: Basic Training with wandb

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse \
    --max-epochs 300
```

### Example 2: Experiment with Custom Tags

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse \
    --wandb-run-name "high_lr_experiment" \
    --wandb-tags high-lr experiment-1 baseline \
    --lr 2e-4
```

### Example 3: Quick Test Run

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse-debug \
    --wandb-run-name "quick_test" \
    --max-samples 1000 \
    --max-epochs 10
```

## Integration with TensorBoard

Both wandb and TensorBoard are enabled simultaneously:

- **TensorBoard**: Local visualization, always enabled
- **wandb**: Cloud-based tracking, optional

View both:

```bash
# TensorBoard (local)
tensorboard --logdir runs/pulse2pulse_mimic/seed_42/tb

# wandb (cloud)
# Click the link shown during training
```

## Resources

- **wandb Documentation**: https://docs.wandb.ai/
- **PyTorch Lightning Integration**: https://docs.wandb.ai/guides/integrations/lightning
- **Example Projects**: https://wandb.ai/gallery
- **Community Forum**: https://community.wandb.ai/

## Summary

Weights & Biases provides:
- ✅ Real-time metric tracking
- ✅ Automatic hyperparameter logging
- ✅ Model checkpoint versioning
- ✅ Experiment comparison
- ✅ Collaboration and sharing
- ✅ System monitoring
- ✅ Media logging (ECG samples)

Enable it with just `--wandb` flag or `enabled: true` in config!
