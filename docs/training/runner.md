# Training with `ecgen train`

Every model family is trained by one YAML-driven runner,
[`ecgen.training.runner`](../api/training.md#ecgentrainingrunner), exposed as the `ecgen` command.
There are no per-model training scripts: the experiment YAML says which family, dataset and
callbacks to build.

## Command line

```bash
pip install -e .        # installs the `ecgen` console script

ecgen train configs/experiments/vae_mimic.yaml
ecgen train configs/experiments/vae_mimic.yaml trainer.max_epochs=5 data.dataset.params.max_samples=1000
ecgen train configs/experiments/vae_mimic.yaml --resume runs/vae_mimic/seed_42/checkpoints/last.ckpt
ecgen train configs/experiments/vae_mimic.yaml --print-config     # show the resolved config, do not train
ecgen list                                                         # registered families, datasets, callbacks

python -m ecgen train configs/experiments/vae_mimic.yaml          # same as `ecgen train`
```

| Argument | Meaning |
|---|---|
| `config` | Experiment YAML (`configs/experiments/*.yaml`). |
| `dotted.key=value ...` | Overrides, applied **after** the YAML, so they always win. |
| `--resume <ckpt>` | Resume from a checkpoint (same as the override `resume=<ckpt>`). |
| `--print-config` | Print the resolved config (defaults filled, `${...}` interpolated) and exit. |

The shell wrappers in `scripts/shell/` are thin conveniences around the same command; extra
arguments are passed through as overrides:

| Wrapper | Runs |
|---|---|
| `run_train_vae_mimic_config.sh [config] [overrides]` | `python -m ecgen train configs/experiments/vae_mimic.yaml ...` |
| `run_train_vqvae_mimic.sh [config] [overrides]` | VQ-VAE stage 1 (`vqvae_mimic.yaml`) |
| `run_train_vqvae_prior_mimic.sh <stage1.ckpt> [config] [overrides]` | prior (`prior_mimic.yaml`) with `model.params.vqvae_checkpoint=<stage1.ckpt>` |
| `run_train_pulse2pulse_mimic.sh [config] [overrides]` | Pulse2Pulse (`pulse2pulse_mimic.yaml`) |
| `run_train_pulse2pulse_standalone.sh` | Pulse2Pulse with settings from env vars (`DATA_DIR`, `EXP_NAME`, `BATCH_SIZE`, `MAX_EPOCHS`, `GPU_ID`, `NUM_WORKERS`) |
| `run_sweep.sh [overrides]` | every config in `configs/experiments/`, then the prior on `runs/vqvae_mimic/seed_42/checkpoints/last.ckpt` if it exists |

### Overrides

Each override is `dotted.key=value`; the value is parsed as YAML (`1e-3` is read as a float).
Integer path components index into lists.

```bash
ecgen train configs/experiments/vae_mimic.yaml \
    model.params.lr=1e-3 \
    trainer.devices=[1] \
    wandb.enabled=false \
    callbacks.2.params.num_samples=2 \
    data.dataset.params.max_samples=null
```

## Config schema

The runner accepts exactly these top-level sections; any other key (for example the old
`visualization:` section) is rejected with an error.

```yaml
experiment:
  name: vae_mimic            # run directory: <runs_root>/<name>/seed_<seed>/
  seed: 42                   # global seed (default 42)
  runs_root: runs            # default "runs"

model:
  name: vae                  # registry name: vae | vqvae | vqvae_prior | pulse2pulse
  # target: my_pkg.models.MyModel   # ...or an import path instead of `name`
  params:                    # the family's config dataclass fields, flat
    in_channels: 12          # must match the dataset's lead count
    base_channels: 64
    latent_channels: 8
    channel_multipliers: [1, 2, 4, 4]
    num_res_blocks: 2
    lr: 0.0001
    kl_weight: 0.0001
    # legacy form `params: {config: {...}}` is still accepted

data:                        # -> ecgen.data.datamodule.ECGDataModule
  dataset:
    name: mimic_iv_ecg       # registry name: mimic_iv_ecg | synthetic (or `target:`)
    params:                  # dataset constructor args; `split` is filled in per split
      root: /path/to/mimic-iv-ecg
      max_samples: 1000      # null = full split
      leads: null            # null = 12 leads in file order; "independent8"; or a list of names
      seq_length: 5000
      scale: zscore          # per-record z-score; "none" = mV; or a number
  batch_size: 32
  num_workers: 4
  pin_memory: null           # default: torch.cuda.is_available()
  split_params:              # per-split overrides of dataset.params
    val: {max_samples: 1000}

checkpoint:                  # -> ModelCheckpoint (defaults shown)
  monitor: val_loss
  mode: min
  save_top_k: 1
  save_last: true
  filename: "epoch{epoch:03d}-step{step:06d}"

callbacks:                   # built in order, after the checkpoint callback
  - name: early_stopping
    params: {monitor: val_loss, patience: 10, mode: min}
  - name: lr_monitor
    params: {logging_interval: step}
  - name: vae_visualization  # save_dir defaults to ${run_dir}/samples
    params: {log_every_n_epochs: 5, num_samples: 4}

trainer:                     # passed to pl.Trainer as keyword arguments
  max_epochs: 100
  accelerator: gpu
  devices: [0]
  check_val_every_n_epoch: 5
  gradient_clip_val: 1.0

wandb:                       # optional; TensorBoard is always on
  enabled: false
  project: ecg-vae
  entity: null
  run_name: null             # default <name>_seed<seed>
  tags: []
  notes: ""
  log_model: false

resume: null                 # checkpoint to resume from (or --resume)
```

`model` and `data` are required; everything else has defaults. String values anywhere in the
config may use `${run_dir}`, `${name}` and `${seed}`, for example
`callbacks: [{name: generated_samples, params: {save_dir: "${run_dir}/gen"}}]`.

The shipped configs are the reference examples:
[`vae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vae_mimic.yaml),
[`vqvae_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/vqvae_mimic.yaml),
[`prior_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/prior_mimic.yaml),
[`pulse2pulse_mimic.yaml`](https://github.com/vlbthambawita/ECGEN/blob/main/configs/experiments/pulse2pulse_mimic.yaml).

## Registry names

`ecgen list` prints the current registries. Built in:

| Kind | Name | Class |
|---|---|---|
| family | `vae` | `ecgen.families.vae.VAELightning` |
| family | `vqvae` | `ecgen.families.vqvae.VQVAELightning` |
| family | `vqvae_prior` | `ecgen.families.vqvae.PriorLightning` |
| family | `pulse2pulse` | `ecgen.families.gan.Pulse2PulseGAN` |
| dataset | `mimic_iv_ecg` | `ecgen.data.datasets.MIMICIVECG` |
| dataset | `synthetic` | `ecgen.data.datasets.SyntheticECG` |
| callback | `vae_visualization` | `VAEVisualizationCallback` (reconstruction plots; VAE and VQ-VAE) |
| callback | `ecg_visualization` | `ECGVisualizationCallback` (real vs generated; Pulse2Pulse) |
| callback | `generated_samples` | `GeneratedSamplesCallback` (periodic sample grids; Pulse2Pulse) |
| callback | `early_stopping` | `pytorch_lightning.callbacks.EarlyStopping` |
| callback | `lr_monitor` | `pytorch_lightning.callbacks.LearningRateMonitor` |
| callback | `model_checkpoint` | `pytorch_lightning.callbacks.ModelCheckpoint` (an extra one; the runner always adds its own from `checkpoint:`) |

A callback whose constructor takes `save_dir` and does not get one writes to `${run_dir}/samples`.
See [Callbacks](callbacks.md) for their parameters.

## Run directory

```text
runs/<experiment.name>/seed_<seed>/
├── config_resolved.yaml         # config after overrides, defaults and ${...} interpolation
├── metadata.json                # command, config path, Python/package versions, git info
├── checkpoints/
│   ├── epoch###-step######.ckpt # top-k by checkpoint.monitor
│   └── last.ckpt
├── samples/                     # visualization callback output
├── metrics/
│   └── train_val_metrics.json   # final callback/logged metrics and best_model_path
├── tb/version_N/                # TensorBoard logs
└── wandb/                       # only when wandb.enabled
```

There is no `best.ckpt`; the best path is recorded in `metrics/train_val_metrics.json`.
Each checkpoint also stores the datamodule state under the `ECGDataModule` key: `lead_names`,
`fs`, `seq_length` and the train split's `feature_normalizer`.

## Compatibility check

After building the model and the train/val datasets, and before training starts, the runner
compares the model's channel count (`in_channels` / `num_channels`) and `seq_length` (where the
config declares one) with the dataset's leads and length. A mismatch fails immediately, e.g.:

```text
ValueError: Model expects 12 channels but the dataset yields 8 leads ['I', 'II', 'V1', ...];
set data.dataset.params.leads or the model's channel count
```

For the prior, the check uses the frozen stage-1 VQ-VAE's input.

## Two-stage VQ-VAE

```bash
# Stage 1: VQ-VAE
ecgen train configs/experiments/vqvae_mimic.yaml

# Stage 2: PixelCNN prior over the frozen stage-1 codes
ecgen train configs/experiments/prior_mimic.yaml \
    model.params.vqvae_checkpoint=runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
# or
./scripts/shell/run_train_vqvae_prior_mimic.sh runs/vqvae_mimic/seed_42/checkpoints/last.ckpt
```

`model.params.num_embeddings` and the dataset's `leads` / `seq_length` in `prior_mimic.yaml` must
match the stage-1 run. See [VQ-VAE training](vqvae_training.md).

## Smoke runs without data

The `synthetic` dataset generates deterministic ECG-like signals, so a config can be tested
without MIMIC-IV-ECG:

```bash
ecgen train configs/experiments/vae_mimic.yaml \
    data.dataset.name=synthetic \
    'data.dataset.params={num_samples: 64, seq_length: 5000}' \
    data.batch_size=8 data.num_workers=0 \
    trainer.max_epochs=1 trainer.check_val_every_n_epoch=1 \
    trainer.accelerator=cpu trainer.devices=1 \
    wandb.enabled=false experiment.name=smoke
```

Keep `trainer.check_val_every_n_epoch` no larger than `trainer.max_epochs`: checkpoints monitor
`val_loss`, so a run that never validates writes no checkpoint at all.

`SyntheticECG` takes `split`, `num_samples`, `leads`, `seq_length`, `fs`, `scale` and `seed`, so
the MIMIC-specific params (e.g. `root`) must be replaced rather than merged, as above.

## Extending: new families, datasets and callbacks

Components are looked up in three registries in [`ecgen.registry`](../api/training.md#ecgenregistry):
`FAMILIES`, `DATASETS` and `CALLBACKS`. A spec is either `{name: <registered>, params: {...}}` or
`{target: <import.path.Class>, params: {...}}`.

**Without registering** — point `target` at any importable class:

```yaml
model:
  target: my_project.models.MyModel
  params: {in_channels: 12, lr: 1e-4}
data:
  dataset:
    target: my_project.data.MyDataset
    params: {root: /data/mine}
callbacks:
  - target: my_project.callbacks.MyCallback
    params: {every_n_epochs: 5}
```

**Registering a name** (in library code, or in a module that is imported before training):

```python
from ecgen.families.base import GenerativeModule
from ecgen.registry import FAMILIES

@FAMILIES.register("my_model")
class MyModel(GenerativeModule):
    def __init__(self, config=None, **kwargs):
        ...
```

Built-in names come from `ecgen.registry.load_builtins()`, which imports `ecgen.families`,
`ecgen.data.datasets` and `ecgen.training.callbacks`. A family should:

- subclass [`GenerativeModule`](../api/models.md#ecgenfamiliesbase), take `config=None, **kwargs`
  and resolve them with `ecgen.utils.config.resolve_config`, so `load_from_checkpoint(path)` works;
- read batches with `self.get_signal(batch)` — batches are
  [`ECGBatch`](../api/data.md#ecgentypes) objects (`signal [B, C, T]`, `lead_names`, `fs`, `cond`,
  `cond_missing`, `meta`); `get_signal` also accepts the legacy dict and tuple formats;
- expose `config.in_channels` (or `num_channels`) and, if fixed, `config.seq_length` so the
  compatibility check can run;
- implement `sample(n_samples, ...)` returning `[n, C, T]`.

For datasets see [Adding a Dataset](../datasets/custom.md).
