# Pulse2Pulse Training

This page covers training the [Pulse2Pulse GAN](../models/gan.md) on MIMIC-IV-ECG and
generating samples from a trained checkpoint. Training runs through the shared runner, `ecgen train`
(see [Training with `ecgen train`](runner.md)).

| File | Role |
|------|------|
| `configs/experiments/pulse2pulse_mimic.yaml` | Experiment config (`model.name: pulse2pulse`) |
| `scripts/shell/run_train_pulse2pulse_mimic.sh` | `python -m ecgen train <config> [overrides]` |
| `scripts/shell/run_train_pulse2pulse_standalone.sh` | Same, with settings from environment variables |
| `scripts/generate/generate_pulse2pulse.py` | Sample generation and plotting from a checkpoint |
| `configs/model/gan/pulse2pulse.yaml` | Model fragment (not read by anything) |

## Training

### Option A: `ecgen train` (or its wrapper)

```bash
ecgen train configs/experiments/pulse2pulse_mimic.yaml
ecgen train configs/experiments/pulse2pulse_mimic.yaml \
    data.dataset.params.root=/data/mimic-iv-ecg data.dataset.params.max_samples=1000 \
    trainer.max_epochs=5 wandb.enabled=false

./scripts/shell/run_train_pulse2pulse_mimic.sh [config] [overrides]   # default configs/experiments/pulse2pulse_mimic.yaml
```

### Option B: environment-variable wrapper

```bash
DATA_DIR=/data/mimic-iv-ecg BATCH_SIZE=64 GPU_ID=1 \
    ./scripts/shell/run_train_pulse2pulse_standalone.sh
```

The wrapper `cd`s to the repo root and runs `pulse2pulse_mimic.yaml` with these overrides:

| Variable | Default | Override |
|----------|---------|-----------|
| `DATA_DIR` | `/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0` | `data.dataset.params.root` |
| `EXP_NAME` | `pulse2pulse_mimic` | `experiment.name` |
| `BATCH_SIZE` | `128` | `data.batch_size` |
| `MAX_EPOCHS` | `300` | `trainer.max_epochs` |
| `GPU_ID` | `0` | `trainer.devices=[GPU_ID]` |
| `NUM_WORKERS` | `4` | `data.num_workers` |

It also always sets `data.dataset.params.max_samples=null` (full dataset),
`data.dataset.params.skip_missing_check=true` and `wandb.enabled=false`; further `key=value`
arguments are appended and win.

### Configuration

`configs/experiments/pulse2pulse_mimic.yaml`:

| Section | Values |
|---------|--------|
| `model` | `name: pulse2pulse`; `Pulse2PulseConfig` fields `model_size: 50`, `num_channels: 8`, `seq_length: 5000` (multiple of 2500), `lr: 1e-4`, `b1: 0.5`, `b2: 0.9`, `lmbda: 10.0` (gradient-penalty weight), `n_critic: 5` |
| `data.dataset` | `mimic_iv_ecg` with **`leads: independent8`** (I, II, V1–V6), `seq_length: 5000`, `scale: zscore`, `max_samples: 10000` |
| `data` | `batch_size: 128`, `num_workers: 4`, `split_params: {val: {max_samples: 1000}}` (caps validation at 1000 records) |
| `checkpoint` | `monitor: val/d_wasserstein`, `mode: max`, `save_top_k: 3` |
| `callbacks` | `ecg_visualization` (`every_n_epochs: 10`), `generated_samples` (`every_n_epochs: 25`, `n_samples: 16`) |
| `trainer` | `max_epochs: 300`, `devices: [0]`, `check_val_every_n_epoch: 5`, `log_every_n_steps: 50` |
| `wandb` | `enabled: true`, `project: ecg-pulse2pulse` |

`model.params.num_channels` must equal the number of selected leads; the runner's
[compatibility check](runner.md#compatibility-check) stops the run otherwise.

`configs/model/gan/pulse2pulse.yaml` (`latent_dim: 100`, `base_channels: 64`, `lr_g/lr_d:
0.0002`, `b2: 0.999`) does not match `Pulse2PulseConfig` and nothing reads it.

## Data

Batches come from the generic `ECGDataModule` over `MIMICIVECG` (see
[MIMIC-IV-ECG](../datasets/mimic.md)): `train` and `val` splits from the dataset's seeded,
subject-level split, each record `(8, 5000)`, z-scored per record over the selected leads.
`Pulse2PulseGAN` reads `batch.signal` and ignores the measurement conditions.

!!! warning "Changed lead set"
    The leads are now **I, II, V1–V6** (`leads: independent8`), which match the CSV columns that
    `ecgen.generate` writes. The removed `scripts/train/train_pulse2pulse.py` fed the
    first 8 leads in file order (I, II, III, aVR, aVF, aVL, V1, V2) and z-scored all 12 leads before
    dropping 4. Models trained that way have a different lead layout from new ones.

## Outputs

```
runs/<exp-name>/seed_<seed>/
├── checkpoints/
│   ├── epoch{EEE}-step{SSSSSS}.ckpt   # top-3 by val/d_wasserstein (mode="max")
│   └── last.ckpt
├── samples/
│   ├── sample_epoch_NNNN.png          # ECGVisualizationCallback: real vs fake, first sample
│   ├── generated_epoch_NNNN.pt        # GeneratedSamplesCallback: (16, C, L) tensor
│   └── generated_epoch_NNNN.png
├── metrics/train_val_metrics.json
├── tb/                                # TensorBoard logs
├── config_resolved.yaml
├── metadata.json
└── wandb/                             # only with wandb.enabled
```

!!! note "Sample callbacks rarely fire with the default schedule"
    Both callbacks run in `on_validation_epoch_end` and only act when
    `trainer.current_epoch % every_n_epochs == 0`. With `check_val_every_n_epoch: 5`,
    validation runs at `current_epoch` 4, 9, 14, … These values are never multiples of 10 or
    25, so in practice the callbacks only write files during the **sanity-check validation at
    epoch 0** (verified with Lightning 2.0). To get periodic plots, use values that line up,
    e.g. `trainer.check_val_every_n_epoch=1`, or pick `every_n_epochs` so that `current_epoch + 1`
    is a multiple of the validation interval.

For the meaning of the logged `train/*` and `val/*` metrics, see
[Pulse2Pulse GAN → Logged metrics](../models/gan.md#logged-metrics).

## Generating samples

```bash
python scripts/generate/generate_pulse2pulse.py \
    --checkpoint runs/pulse2pulse_mimic/seed_42/checkpoints/last.ckpt \
    --output-dir generated_samples \
    --n-samples 16 \
    --save-tensors
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | **required** | Lightning checkpoint |
| `--output-dir` | `generated_samples` | Created if missing |
| `--n-samples` | `16` | Number of ECGs to generate |
| `--device` | `cuda:0` | Falls back to CPU if CUDA is unavailable |
| `--seed` | `42` | `torch.manual_seed` |
| `--save-tensors` | off | Also save `generated_samples.pt` |

The script writes:

- `generated_grid.png`: a 4-column grid with all leads overlaid per sample
- `generated_sample_0_detailed.png`: one subplot per lead for sample 0
- `generated_sample_{0..3}.png`: per-lead plots for the first `min(n_samples, 4)` samples
- `generated_samples.pt`: a `(n_samples, C, L)` CPU tensor, only with `--save-tensors`

The script loads the model with
`Pulse2PulseGAN.load_from_checkpoint(args.checkpoint, map_location=device)` and needs no extra
arguments; see [Pulse2Pulse GAN → Loading a checkpoint](../models/gan.md#loading-a-checkpoint).

## Setup check

`scripts/utils/test_pulse2pulse_setup.py` instantiates the model, the legacy data config
(`ecgen.data.pulse2pulse_mimic`), callbacks, losses and metrics from the `ecgen` package as a quick
smoke test:

```bash
python scripts/utils/test_pulse2pulse_setup.py
```

For an end-to-end check without data, train for one epoch on synthetic ECGs:

```bash
ecgen train configs/experiments/pulse2pulse_mimic.yaml data.dataset.name=synthetic \
    'data.dataset.params={num_samples: 64, leads: independent8}' data.split_params={} \
    trainer.max_epochs=1 trainer.check_val_every_n_epoch=1 \
    trainer.accelerator=cpu trainer.devices=1 wandb.enabled=false
```

## See also

- [Pulse2Pulse GAN](../models/gan.md): architecture, loss, known issues
- [Callbacks](callbacks.md)
- [Training Guide](guide.md)
- [Evaluation Metrics](../evaluation/metrics.md)
