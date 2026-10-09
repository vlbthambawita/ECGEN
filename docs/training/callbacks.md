# Training Callbacks

Callbacks live in
[`training/callbacks/visualization.py`](https://github.com/vlbthambawita/ECGEN/blob/main/training/callbacks/visualization.py)
and are exported from `training.callbacks`:

```python
from training.callbacks import (
    VAEVisualizationCallback,
    ECGVisualizationCallback,
    GeneratedSamplesCallback,
)
```

`training/callbacks/checkpointing.py` (`CustomCheckpointCallback`) and
`training/callbacks/logging.py` (`CustomLoggingCallback`) are empty placeholders.

All three visualization callbacks act in `on_validation_epoch_end`, so they only fire on epochs where
validation runs (`check_val_every_n_epoch`). They save PNGs with matplotlib's `Agg` backend.

## VAEVisualizationCallback

Plots real vs reconstructed ECGs for autoencoders (used by the VAE and VQ-VAE training scripts).

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
- Takes the **first batch** of `trainer.val_dataloaders`, unpacked as `ecg, _ = batch` (i.e. it
  expects `(ecg, features)` tuples), and keeps the first `num_samples`.
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

The training scripts build their callbacks in code. In your own script:

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from training.callbacks import VAEVisualizationCallback

callbacks = [
    ModelCheckpoint(
        dirpath="runs/my_exp/seed_42/checkpoints",
        filename="epoch{epoch:03d}-step{step:06d}",
        auto_insert_metric_name=False,
        monitor="val_loss", mode="min", save_top_k=3, save_last=True,
    ),
    EarlyStopping(monitor="val_loss", mode="min", patience=10),
    LearningRateMonitor(logging_interval="step"),
    VAEVisualizationCallback(save_dir="runs/my_exp/seed_42/samples", log_every_n_epochs=5),
]

trainer = pl.Trainer(callbacks=callbacks, max_epochs=100)
```

All ECGEN LightningModules log `val_loss`, so it is the metric to monitor.

### From YAML

The `callbacks:` list in the experiment files (`target` + `params`) is only instantiated by
`training/trainers/gan_trainer.py`. `train_vae_mimic.py`, `train_vqvae_mimic.py` and
`train_prior_mimic.py` ignore it; they configure visualization through the `visualization:` section
and early stopping / top-k through `--patience` / `--save-top-k`.

## Custom callbacks

Subclass `pytorch_lightning.Callback`:

```python
import pytorch_lightning as pl

class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ...
```

and add it to `training/callbacks/` and `training/callbacks/__init__.py` if it should be shared.
