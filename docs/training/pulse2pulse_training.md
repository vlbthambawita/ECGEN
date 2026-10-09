# Pulse2Pulse Training

This page covers training the [Pulse2Pulse GAN](../models/gan.md) on MIMIC-IV-ECG and
generating samples from a trained checkpoint.

| File | Role |
|------|------|
| `scripts/train/train_pulse2pulse.py` | Training entry point (argparse only, **no** `--config` flag) |
| `scripts/shell/run_train_pulse2pulse_standalone.sh` | Wrapper that calls the train script with CLI flags (**use this one**) |
| `scripts/shell/run_train_pulse2pulse_mimic.sh` | Wrapper that passes `--config` (**currently broken**, see below) |
| `scripts/generate/generate_pulse2pulse.py` | Sample generation and plotting from a checkpoint |
| `configs/experiments/pulse2pulse_mimic.yaml` | Experiment YAML (not read by the train script) |
| `configs/model/gan/pulse2pulse.yaml` | Model fragment (not read by anything) |
| `data/datasets/mimic/pulse2pulse_datamodule.py` | New-layout copy of the Pulse2Pulse `LightningDataModule` |

## Requirement: the legacy `ecgen` package

!!! warning "This path imports the legacy `ecgen` package from `src/ecgen/`"
    `train_pulse2pulse.py` and `generate_pulse2pulse.py` import from the **legacy** layout:

    ```python
    from ecgen.models.pulse2pulse import Pulse2PulseGAN, Pulse2PulseConfig
    from ecgen.data.pulse2pulse_mimic import Pulse2PulseMIMICDataModule, Pulse2PulseMIMICConfig
    from ecgen.training.callbacks import ECGVisualizationCallback, GeneratedSamplesCallback
    from ecgen.utils.seed import set_global_seed
    ```

    `pip install -e .` does **not** make `ecgen` importable. `pyproject.toml` names the
    distribution `ecgen`, but its package list is
    `["models", "data", "training", "evaluation", "utils", "visualization"]`, so `src/ecgen/`
    is not installed.

    Both scripts try to fix this themselves:

    ```python
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    ```

    However, the scripts live two levels deep (`scripts/train/`, `scripts/generate/`), so
    `_REPO_ROOT` resolves to `scripts/` and the inserted path is `scripts/src`, which does not
    exist. **The `sys.path` hack has no effect.**

    The scripts work in an environment only if `src/` is already on `sys.path`. This can happen
    through a stale `.pth` file left by an older editable install that used a `src/` layout,
    which is how the maintainers' `pytorch2` conda env works. In a clean environment, you get
    `ModuleNotFoundError: No module named 'ecgen'`.

    **What you need to do:** put `src/` on `PYTHONPATH` yourself, in addition to
    `pip install -e .`:

    ```bash
    cd /path/to/ECGEN
    pip install -e .
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python -c "import ecgen.models.pulse2pulse"   # sanity check
    ```

    `data/datasets/mimic/pulse2pulse_datamodule.py`, the new-layout copy of the data module,
    has the same dependency. Inside `setup()` it does
    `from ecgen.data.mimic_dataset import MIMICIVECGDataset`, so it also needs `src/` on the path.

## Training

### Option A: standalone wrapper (recommended)

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
./scripts/shell/run_train_pulse2pulse_standalone.sh
```

The wrapper `cd`s to the repo root and reads these environment variables:

| Variable | Default | Passed as |
|----------|---------|-----------|
| `DATA_DIR` | `/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0` | `--data-dir` |
| `EXP_NAME` | `pulse2pulse_mimic` | `--exp-name` |
| `BATCH_SIZE` | `128` | `--batch-size` |
| `MAX_EPOCHS` | `300` | `--max-epochs` |
| `GPU_ID` | `0` | `--devices` |
| `NUM_WORKERS` | `4` | `--num-workers` |

It also always passes `--accelerator gpu --skip-missing-check --viz-every-n-epochs 10
--save-samples-every-n-epochs 25 --check-val-every-n-epoch 5 --log-every-n-steps 50`.
If `DATA_DIR` is missing, it prints a warning but still runs.

```bash
# example: custom data path, smaller batch, GPU 1
DATA_DIR=/data/mimic-iv-ecg BATCH_SIZE=64 GPU_ID=1 \
    ./scripts/shell/run_train_pulse2pulse_standalone.sh
```

The wrapper has no way to set `--max-samples` or `--wandb`. For those, call the Python script
directly.

### Option B: call the script directly

```bash
python scripts/train/train_pulse2pulse.py \
    --data-dir /path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
    --max-samples 1000 --max-epochs 5 --skip-missing-check
```

#### CLI flags

=== "Experiment / data"

    | Flag | Default | Description |
    |------|---------|-------------|
    | `--exp-name` | `pulse2pulse_mimic` | Run name |
    | `--seed` | `42` | Passed to `set_global_seed` |
    | `--runs-root` | `runs` | Root output directory |
    | `--data-dir` | **required** | MIMIC-IV-ECG root |
    | `--batch-size` | `128` | Train and val batch size |
    | `--num-workers` | `4` | DataLoader workers |
    | `--max-samples` | `None` | Limit the train set (debugging) |
    | `--skip-missing-check` | off | Skip checking that every WFDB record exists |

=== "Model"

    | Flag | Default | `Pulse2PulseConfig` field |
    |------|---------|---------------------------|
    | `--model-size` | `50` | `model_size` |
    | `--num-channels` | `8` | `num_channels` (also the number of leads kept by the data module) |
    | `--seq-length` | `5000` | `seq_length` (must be a multiple of 2500) |
    | `--lr` | `1e-4` | `lr` |
    | `--b1` | `0.5` | `b1` |
    | `--b2` | `0.9` | `b2` |
    | `--lmbda` | `10.0` | `lmbda` (gradient-penalty weight) |
    | `--n-critic` | `5` | `n_critic` |

=== "Trainer / callbacks"

    | Flag | Default | Description |
    |------|---------|-------------|
    | `--max-epochs` | `300` | |
    | `--accelerator` | `gpu` | `gpu` or `cpu` |
    | `--devices` | `[0]` | One or more device IDs (`nargs="+"`, ints) |
    | `--log-every-n-steps` | `50` | |
    | `--check-val-every-n-epoch` | `5` | |
    | `--viz-every-n-epochs` | `10` | `ECGVisualizationCallback.every_n_epochs` |
    | `--save-samples-every-n-epochs` | `25` | `GeneratedSamplesCallback.every_n_epochs` |
    | `--resume` | `None` | Checkpoint passed to `trainer.fit(ckpt_path=...)` |

=== "Weights & Biases"

    | Flag | Default | Description |
    |------|---------|-------------|
    | `--wandb` | off | Add a `WandbLogger` (with `log_model=True`) |
    | `--wandb-project` | `ecg-pulse2pulse` | |
    | `--wandb-entity` | `None` | |
    | `--wandb-run-name` | `None` | Defaults to `<exp-name>_seed<seed>` |
    | `--wandb-tags` | `pulse2pulse wavegan` | |

    If `wandb` is not installed or fails to initialise, the script prints a warning and
    continues with TensorBoard only.

### Option C: `run_train_pulse2pulse_mimic.sh` (broken)

!!! failure "Does not run"
    ```bash
    ./scripts/shell/run_train_pulse2pulse_mimic.sh [config]   # default configs/experiments/pulse2pulse_mimic.yaml
    ```
    This wrapper calls `python scripts/train/train_pulse2pulse.py --config "${CONFIG}"`, but
    `train_pulse2pulse.py` defines no `--config` option and requires `--data-dir`. argparse
    exits with
    `error: the following arguments are required: --data-dir`. Unlike the VAE/VQ-VAE train
    scripts, the Pulse2Pulse script does not read YAML configs at all.

`configs/experiments/pulse2pulse_mimic.yaml` still documents the intended setup. Its values
match the CLI defaults, with these differences:

| YAML key | YAML value | CLI equivalent / default |
|----------|------------|--------------------------|
| `wandb.enabled` | `true` | `--wandb` (off by default) |
| `wandb.log_model` | `False` | script hard-codes `log_model=True` |
| `data.params.config.max_samples` | `10000` | `--max-samples` (default `None` = all) |
| `data.params.config.skip_missing_check` | `False` | `--skip-missing-check` (off = `False`) |

Its `target:` strings point at `ecgen.*` classes.

`configs/model/gan/pulse2pulse.yaml` (`latent_dim: 100`, `base_channels: 64`, `lr_g/lr_d:
0.0002`, `b2: 0.999`) does not match `Pulse2PulseConfig` and nothing reads it.

## Data module

The train script uses `ecgen.data.pulse2pulse_mimic.Pulse2PulseMIMICDataModule`.
`data/datasets/mimic/pulse2pulse_datamodule.py` is an identical copy in the new layout.

```python
@dataclass
class Pulse2PulseMIMICConfig:
    data_dir: str
    batch_size: int = 128
    num_workers: int = 4
    max_samples: Optional[int] = None
    skip_missing_check: bool = True     # note: CLI default is False
    num_channels: int = 8
    seq_length: int = 5000
```

`setup()` builds two `MIMICIVECGDataset`s from `ecgen.data.mimic_dataset`. Both always load
`num_leads=12`, using the dataset's internal seeded split:

- **train**: `split="train"`, `max_samples=config.max_samples`
- **val**: `split="val"`, `max_samples=min(config.max_samples or 1000, 1000)`, which caps
  validation at 1000 records

Each dataset is wrapped in `ECGDatasetAdapter`. The adapter drops the conditioning features,
keeps the first `num_channels` leads (I, II, III, aVR, aVL, aVF, V1, V2 for 8), casts to float,
and returns `{"ecg_signals": tensor}` of shape `(num_channels, seq_length)`. Training
shuffles, validation does not, and `pin_memory` is enabled when CUDA is available. There is
no test dataloader.

Records are z-score normalised per record by `MIMICIVECGDataset` (see
[MIMIC-IV-ECG](../datasets/mimic.md)).

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
├── tb/                                # TensorBoard logs
└── wandb/                             # only with --wandb
```

Unlike the VAE/VQ-VAE scripts, this script does **not** write `config_resolved.yaml` or
`metadata.json`.

!!! note "Sample callbacks rarely fire with the default schedule"
    Both callbacks run in `on_validation_epoch_end` and only act when
    `trainer.current_epoch % every_n_epochs == 0`. With `--check-val-every-n-epoch 5`,
    validation runs at `current_epoch` 4, 9, 14, … These values are never multiples of 10 or
    25, so in practice the callbacks only write files during the **sanity-check validation at
    epoch 0** (verified with Lightning 2.0). To get periodic plots, use values that line up,
    e.g. `--check-val-every-n-epoch 1`, or pick `every_n_epochs` so that `current_epoch + 1`
    is a multiple of the validation interval.

For the meaning of the logged `train/*` and `val/*` metrics, see
[Pulse2Pulse GAN → Logged metrics](../models/gan.md#logged-metrics).

## Generating samples

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
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

!!! bug "Checkpoint loading fails as written"
    The script calls `Pulse2PulseGAN.load_from_checkpoint(args.checkpoint, map_location=device)`.
    That call raises `TypeError: __init__() missing 1 required positional argument: 'config'`,
    because the module saves its config fields, not a `config` argument, as hyperparameters.
    Until that is fixed, load the model as shown in
    [Pulse2Pulse GAN → Loading a checkpoint](../models/gan.md#loading-a-checkpoint), by passing
    `config=Pulse2PulseConfig(**hparams)` and `strict=False`, and then call
    `model.generate_samples(n)`.

## Setup check

`scripts/utils/test_pulse2pulse_setup.py` instantiates the model, data config, callbacks,
losses and metrics from the `ecgen.*` package as a quick smoke test. It has the same
`sys.path` issue, so run it with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/utils/test_pulse2pulse_setup.py
```

## See also

- [Pulse2Pulse GAN](../models/gan.md): architecture, loss, known issues
- [Callbacks](callbacks.md)
- [Training Guide](guide.md)
- [Evaluation Metrics](../evaluation/metrics.md)
