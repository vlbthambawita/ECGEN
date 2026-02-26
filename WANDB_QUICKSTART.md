# Weights & Biases Quick Start

## 🚀 3 Steps to Enable wandb

### 1. Install and Login

```bash
pip install wandb
wandb login
```

### 2. Enable in Config

Edit `configs/experiments/pulse2pulse_mimic.yaml`:

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
  entity: your-username  # Optional
```

### 3. Train

```bash
./scripts/run_train_pulse2pulse_mimic.sh
```

## 🎯 Or Use Command Line

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse
```

## 📊 What You Get

- ✅ Real-time metric tracking
- ✅ ECG sample visualization
- ✅ Model checkpoint versioning
- ✅ Hyperparameter logging
- ✅ System monitoring (GPU/CPU)
- ✅ Experiment comparison
- ✅ Shareable results

## 🔗 View Results

After starting training, click the link:

```
View run at: https://wandb.ai/your-username/ecg-pulse2pulse/runs/xxxxx
```

## ⚙️ Configuration Options

### Full Config Example

```yaml
wandb:
  enabled: true
  project: ecg-pulse2pulse
  entity: your-username
  run_name: null  # Auto-generated
  tags:
    - pulse2pulse
    - wavegan
    - ecg-generation
  notes: "Experiment description"
  log_model: true
  log_freq: 50
```

### Command Line Options

```bash
--wandb                          # Enable wandb
--wandb-project PROJECT          # Project name
--wandb-entity ENTITY            # Username/team
--wandb-run-name NAME            # Run name
--wandb-tags TAG1 TAG2 TAG3      # Tags
```

## 🔧 Common Use Cases

### Baseline Experiment

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse \
    --wandb-tags baseline experiment-1
```

### Hyperparameter Sweep

```bash
# Run 1: Low learning rate
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb --wandb-tags lr-sweep low-lr \
    --lr 5e-5

# Run 2: High learning rate
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb --wandb-tags lr-sweep high-lr \
    --lr 2e-4
```

### Quick Test

```bash
python scripts/train_pulse2pulse.py \
    --data-dir /path/to/MIMIC-IV-ECG \
    --wandb \
    --wandb-project ecg-pulse2pulse-debug \
    --max-samples 1000 \
    --max-epochs 10
```

## 🚫 Disable wandb

### In Config

```yaml
wandb:
  enabled: false
```

### Environment Variable

```bash
WANDB_MODE=disabled ./scripts/run_train_pulse2pulse_mimic.sh
```

## 📴 Offline Mode

Train without internet:

```bash
WANDB_MODE=offline ./scripts/run_train_pulse2pulse_mimic.sh
```

Sync later:

```bash
wandb sync runs/pulse2pulse_mimic/seed_42/wandb/
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Not installed | `pip install wandb` |
| Not logged in | `wandb login` |
| Network error | Use offline mode: `WANDB_MODE=offline` |
| Permission denied | Check entity/username |

## 📚 More Information

- **Full Guide**: `docs/wandb_setup.md`
- **wandb Docs**: https://docs.wandb.ai/
- **PyTorch Lightning**: https://docs.wandb.ai/guides/integrations/lightning

---

**Ready?** Just add `--wandb` to your training command! 🎉
