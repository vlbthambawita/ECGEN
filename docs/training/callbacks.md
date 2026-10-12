# Training Callbacks

Callbacks live in
[`src/ecgen/training/callbacks/visualization.py`](https://github.com/vlbthambawita/ECGEN/blob/main/src/ecgen/training/callbacks/visualization.py)
and are exported from `ecgen.training.callbacks`:

```python
from ecgen.training.callbacks import (
    VAEVisualizationCallback,
    ECGVisualizationCallback,
    GeneratedSamplesCallback,
)
```

`ecgen.training.callbacks.checkpointing` (`CustomCheckpointCallback`) and
`ecgen.training.callbacks.logging` (`CustomLoggingCallback`) are empty placeholders.

## Registry names

The runner builds the `callbacks:` list of an experiment YAML in order. Each entry is
`{name: <registered name>, params: {...}}` or `{target: <import.path.Class>, params: {...}}`:

| Name | Class |
|---|---|
| `vae_visualization` | `VAEVisualizationCallback` |
| `ecg_visualization` | `ECGVisualizationCallback` |
| `generated_samples` | `GeneratedSamplesCallback` |
| `early_stopping` | `pytorch_lightning.callbacks.EarlyStopping` |
| `lr_monitor` | `pytorch_lightning.callbacks.LearningRateMonitor` |
| `model_checkpoint` | `pytorch_lightning.callbacks.ModelCheckpoint` |

A callback whose constructor takes `save_dir` gets `${run_dir}/samples` if `params` does not set it.
The main `ModelCheckpoint` is always added by the runner from the `checkpoint:` section, so
`model_checkpoint` is only for an additional one.

All three visualization callbacks act in `on_validation_epoch_end`, so they only fire on epochs where
validation runs (`check_val_every_n_epoch`). They save PNGs with matplotlib's `Agg` backend.

## VAEVisualizationCallback

Plots real vs reconstructed ECGs for autoencoders (used by `vae_mimic.yaml` and `vqvae_mimic.yaml`).

```python
VAEVisualizationCallback(
    save_dir,                  # str | Path, created if missing
    log_every_n_epochs=5,
    num_samples=4,
    plot_all_leads=True,       # True: one subplot per lead; False: all leads overlaid
    log_to_tensorboard=True,
    log_to_wandb=False,
)
```

Behaviour:

- Skipped during the sanity check and when `current_epoch % log_every_n_epochs != 0`.
- Takes the **first batch** of `trainer.val_dataloaders`, reads the signal with
  `ecgen.types.get_signal` (an `ECGBatch`, a `{"signal"}` / legacy `{"ecg_signals"}` dict, or an
  `(ecg, features)` tuple), and keeps the first `num_samples`.
- Reconstructs with `pl_module.vae(ecg)` if the module has a `vae` attribute, else
  `pl_module.vqvae(ecg)`, else `pl_module(ecg)`; the first element of the returned tuple is used.
- Saves `save_dir/epoch_{epoch:04d}.png` (150 dpi), logs the figure to TensorBoard under the tag
  `vae_reconstruction` and, if enabled, to W&B under `vae_reconstruction`.

See [Visualization](../visualization/guide.md) for the plot layouts.

## ECGVisualizationCallback

Plots one real vs one generated ECG for GANs.

```python
ECGVisualizationCallback(save_dir, every_n_epochs=10, n_samples=1)
```

Reads `pl_module._val_real_sample` and `pl_module._val_fake_sample` (set by
`Pulse2PulseGAN.validation_step`) and does nothing if they are missing. Plots up to 8 leads in two
columns and saves `save_dir/sample_epoch_{epoch:04d}.png`; also logs `samples/real_vs_fake` to any
logger whose `experiment` has a `log` method (W&B). `n_samples` is stored but not used.

## GeneratedSamplesCallback

Periodically generates and saves samples.

```python
GeneratedSamplesCallback(save_dir, every_n_epochs=25, n_samples=16)
```

Calls `pl_module.generate_samples(n_samples)` (does nothing if the module lacks that method — only
`Pulse2PulseGAN` has it), saves the tensor to `generated_epoch_{epoch:04d}.pt` and a plot of the first
sample (up to 8 leads) to `generated_epoch_{epoch:04d}.png`, and logs `samples/generated` to W&B.

!!! note
    `ECGVisualizationCallback` and `GeneratedSamplesCallback` take `every_n_epochs`, whereas
    `VAEVisualizationCallback` takes `log_every_n_epochs`.

## Using callbacks

In an experiment YAML (from `vae_mimic.yaml`):

```yaml
checkpoint:            # the runner's ModelCheckpoint
  monitor: val_loss
  mode: min
  save_top_k: 3

callbacks:
  - name: early_stopping
    params: {monitor: val_loss, patience: 10, mode: min, verbose: true}
  - name: lr_monitor
    params: {logging_interval: step}
  - name: vae_visualization          # save_dir defaults to ${run_dir}/samples
    params:
      log_every_n_epochs: 5
      num_samples: 4
      log_to_wandb: true
```

Individual values can be changed from the command line, e.g.
`ecgen train configs/experiments/vae_mimic.yaml callbacks.2.params.log_every_n_epochs=1`.
The VAE, VQ-VAE and prior log `val_loss`; Pulse2Pulse checkpoints on `val/d_wasserstein`
(`mode: max`).

## Custom callbacks

Subclass `pytorch_lightning.Callback`:

```python
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ...
```

then reference it from the YAML with `target: my_project.callbacks.MyCallback`, or register a name
for it:

```python
from ecgen.registry import CALLBACKS

CALLBACKS.add("my_callback", MyCallback)   # or @CALLBACKS.register("my_callback") on the class
```

Shared callbacks belong in `src/ecgen/training/callbacks/`, registered in its `__init__.py`.
