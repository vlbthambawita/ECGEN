# Weights & Biases Integration Summary

## ✅ What Was Added

### 1. Config Support (`configs/experiments/pulse2pulse_mimic.yaml`)

Added wandb configuration section:

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
  entity: null
  run_name: null
  tags:
    - pulse2pulse
    - wavegan
    - ecg-generation
  notes: "Pulse2Pulse WaveGAN training on MIMIC-IV-ECG"
  log_model: true
  log_freq: 50
```

### 2. Training Script Integration (`src/ecgen/training/train.py`)

- Added WandbLogger support alongside TensorBoard
- Automatic wandb initialization from config
- Logs all hyperparameters and configuration
- Graceful fallback if wandb not installed

### 3. Standalone Script Support (`scripts/train_pulse2pulse.py`)

Added command-line arguments:
- `--wandb`: Enable wandb logging
- `--wandb-project`: Project name
- `--wandb-entity`: Username/team
- `--wandb-run-name`: Custom run name
- `--wandb-tags`: Tags for organization

### 4. Callback Integration (`src/ecgen/training/callbacks.py`)

Both callbacks now log to wandb:
- **ECGVisualizationCallback**: Logs real vs. fake comparison images
- **GeneratedSamplesCallback**: Logs generated ECG samples

### 5. Documentation

Created comprehensive guides:
- `docs/wandb_setup.md`: Full setup and usage guide
- `WANDB_QUICKSTART.md`: Quick reference card
- Updated existing docs with wandb info

## 🚀 How to Use

### Quick Start

```bash
# Install
pip install wandb
wandb login

# Enable in config
# Edit configs/experiments/pulse2pulse_mimic.yaml
# Set wandb.enabled: true

# Train
./scripts/run_train_pulse2pulse_mimic.sh
```

### Or Use Command Line

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse
```

## 📊 What Gets Logged

### Metrics (Automatic)
- `train/d_loss`: Discriminator loss
- `train/d_wasserstein`: Wasserstein distance
- `train/g_loss`: Generator loss
- `val/d_wasserstein`: Validation Wasserstein distance
- `val_loss`: Validation loss

### Media (Via Callbacks)
- Real vs. fake ECG comparisons (every 10 epochs)
- Generated ECG samples (every 25 epochs)

### Configuration (Automatic)
- All model hyperparameters
- Training configuration
- Data configuration
- System information

### Model Checkpoints (Optional)
- Best model checkpoint
- Last model checkpoint
- Linked as wandb artifacts

## 🎯 Key Features

1. **Dual Logging**: TensorBoard (local) + wandb (cloud)
2. **Optional**: Can be disabled without breaking anything
3. **Flexible**: Config-based or command-line
4. **Automatic**: No manual logging code needed
5. **Graceful**: Falls back if wandb not installed

## 📁 Modified Files

```
Modified:
├── configs/experiments/pulse2pulse_mimic.yaml    [Added wandb section]
├── src/ecgen/training/train.py                  [Added WandbLogger]
├── src/ecgen/training/callbacks.py              [Added wandb image logging]
├── scripts/train_pulse2pulse.py                 [Added wandb args]
├── QUICKSTART_PULSE2PULSE.md                    [Added wandb section]

New:
├── docs/wandb_setup.md                          [Full guide]
├── WANDB_QUICKSTART.md                          [Quick reference]
└── WANDB_INTEGRATION_SUMMARY.md                 [This file]
```

## 🔧 Configuration Examples

### Minimal (Just Enable)

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
```

### Full Configuration

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
  entity: your-username
  run_name: pulse2pulse_baseline_v1
  tags:
    - pulse2pulse
    - wavegan
    - baseline
    - v1
  notes: |
    Baseline experiment with default hyperparameters.
    Model size: 50, LR: 1e-4, Batch size: 128
  log_model: true
  log_freq: 50
```

### Disable wandb

```yaml
wandb:
  enabled: false
```

## 🎓 Usage Examples

### Example 1: Basic Training with wandb

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse
```

### Example 2: Custom Run Name and Tags

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse \
    --wandb-run-name "lr_sweep_high" \
    --wandb-tags lr-sweep high-lr experiment-1
```

### Example 3: Config-Based

```bash
# Edit config to enable wandb
./scripts/run_train_pulse2pulse_mimic.sh
```

### Example 4: Offline Mode

```bash
WANDB_MODE=offline python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb
```

## 🔍 Viewing Results

After starting training, you'll see:

```
Weights & Biases logging enabled: ecg-pulse2pulse/pulse2pulse_mimic_seed42
View run at: https://wandb.ai/your-username/ecg-pulse2pulse/runs/xxxxx
```

Click the link to view:
- Real-time metric plots
- ECG sample images
- System monitoring
- Hyperparameter tracking
- Model checkpoints

## 🆘 Troubleshooting

### wandb not installed
```bash
pip install wandb
```

### Not logged in
```bash
wandb login
```

### Disable temporarily
```bash
WANDB_MODE=disabled ./scripts/run_train_pulse2pulse_mimic.sh
```

### Network issues
```bash
WANDB_MODE=offline ./scripts/run_train_pulse2pulse_mimic.sh
```

## ✨ Benefits

1. **Cloud-based tracking**: Access from anywhere
2. **Experiment comparison**: Compare multiple runs easily
3. **Collaboration**: Share results with team
4. **Model versioning**: Track checkpoint lineage
5. **Media logging**: View ECG samples in dashboard
6. **System monitoring**: GPU/CPU usage tracking
7. **Hyperparameter tracking**: Automatic logging
8. **No code changes**: Just enable in config

## 🔄 Backward Compatibility

- **TensorBoard still works**: Both loggers run simultaneously
- **Optional feature**: Can be disabled without issues
- **Graceful degradation**: Falls back if wandb not available
- **No breaking changes**: Existing code continues to work

## 📚 Documentation

- **Quick Start**: `WANDB_QUICKSTART.md`
- **Full Guide**: `docs/wandb_setup.md`
- **Main README**: `README_PULSE2PULSE.md`
- **Training Guide**: `docs/pulse2pulse_training.md`

## 🎉 Summary

Weights & Biases is now fully integrated with Pulse2Pulse training:

✅ Config-based and command-line support  
✅ Automatic metric and hyperparameter logging  
✅ ECG sample visualization in dashboard  
✅ Model checkpoint tracking  
✅ Optional and non-breaking  
✅ Works alongside TensorBoard  
✅ Comprehensive documentation  

**Enable it with just one flag: `--wandb`** 🚀
