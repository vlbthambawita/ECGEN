# Standalone VQ-VAE (`single_file/`)

`single_file/` holds a **self-contained** two-stage VQ-VAE for MIMIC-IV-ECG. All model, data, loss, callback and training code is in one Python file, so you can copy the directory to another machine and run it without installing ECGEN.

| File | Purpose |
|------|---------|
| `train_vqvae_standalone.py` | Everything: models, dataset, Lightning modules, callback, CLI. `--stage 1` trains the VQ-VAE and `--stage 2` trains the PixelCNN prior. |
| `run_train_vqvae.sh` | Bash wrapper configured through environment variables. Runs stage 1, stage 2, or both in sequence. |
| `test_data_loading.py` | Pre-flight check of a MIMIC-IV-ECG directory. |
| `README.md`, `QUICK_START.md`, `SUMMARY.md`, `FIXES.md`, `WANDB_GUIDE.md` | Older notes that this page replaces. Parts of them are out of date (see [Corrections](#corrections-to-the-older-notes)). |

!!! warning "Independent copy: changes do not propagate"
    The standalone file imports nothing from `models/`, `data/`, `training/` or `utils/`. It **re-implements** the same architecture as the packaged [VQ-VAE](../models/vqvae.md) (`models/vqvae/`). A fix or change in one is **not** reflected in the other. Checkpoints are also not interchangeable without care, because the module class paths differ. See [Differences from the packaged implementation](#differences-from-the-packaged-implementation).

## Requirements

```bash
pip install torch pytorch-lightning matplotlib numpy pandas scikit-learn wfdb
pip install wandb   # optional; only used with --wandb
```

`wandb` is imported in a `try/except`. If it is missing, `--wandb` prints a warning and training continues with TensorBoard only.

## Quick start

```bash
cd single_file

# 1. Check the data
python test_data_loading.py --data-dir /path/to/mimic-iv-ecg

# 2. Smoke test: 100 records/split, 10 epochs per stage, no W&B
DATA_DIR=/path/to/mimic-iv-ecg \
MAX_SAMPLES=100 MAX_EPOCHS_STAGE1=10 MAX_EPOCHS_STAGE2=10 \
WANDB_ENABLED=false \
./run_train_vqvae.sh both

# 3. Monitor
tensorboard --logdir runs
```

## Validating the data

```bash
python test_data_loading.py [--data-dir PATH]
```

| Flag | Default |
|------|---------|
| `--data-dir` | `$DATA_DIR` if set, otherwise the hard-coded cluster path `/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0` |

The checks run in this order and stop at the first hard failure (exit code 1):

1. The directory exists.
2. `machine_measurements.csv` exists (its size is printed).
3. The CSV loads with pandas (row count and columns are printed).
4. The required columns `subject_id`, `study_id`, and the 9 measurement columns are present.
5. `files/` exists, and the `.hea`/`.dat` pair is present for the **first 10 rows**. If none are found, the check fails. If only some are, it prints a warning.
6. `wfdb.rdrecord` loads the first readable record among the first 100 rows, and the script prints its shape and sampling frequency. If `wfdb` is not installed, this is only a warning.

On success it prints suggested `run_train_vqvae.sh` commands.

## Running with the shell wrapper

```bash
./run_train_vqvae.sh [1|2|both]     # default: both
```

- `1`: stage 1 only.
- `2`: stage 2 only. Requires `VQVAE_CHECKPOINT` to point to an existing file.
- `both`: stage 1, then stage 2 using the checkpoint stage 1 just produced.

The wrapper calls `train_vqvae_standalone.py` by absolute path, so you can run it from any directory. However, `RUNS_ROOT` (default `runs`) is resolved **relative to your current directory**.

After stage 1, the wrapper picks the stage-2 input with `find <ckpt_dir> -name "epoch*.ckpt" | sort | tail -n 1`, falling back to `last.ckpt`. This gives the **highest-epoch** retained top-k checkpoint, which is not necessarily the one with the lowest `val_loss`. To choose explicitly, run stage 2 separately with `VQVAE_CHECKPOINT=...`.

### Environment variables

All variables are optional. The defaults below are copied from the script.

=== "Data & experiment"

    | Variable | Default | Maps to |
    |----------|---------|---------|
    | `DATA_DIR` | cluster path (see above) | `--data-dir`; the script exits if the directory does not exist |
    | `EXP_NAME_STAGE1` | `vqvae_mimic_standalone` | `--exp-name` (stage 1) |
    | `EXP_NAME_STAGE2` | `prior_mimic_standalone` | `--exp-name` (stage 2) |
    | `SEED` | `42` | `--seed` |
    | `RUNS_ROOT` | `runs` | `--runs-root` |
    | `BATCH_SIZE` | `32` | `--batch-size` |
    | `NUM_WORKERS` | `4` | `--num-workers` |
    | `MAX_SAMPLES` | `1000` | `--max-samples`; set `MAX_SAMPLES=null` for the full dataset (an empty value falls back to 1000) |
    | `VAL_SPLIT` | `0.1` | `--val-split` |
    | `TEST_SPLIT` | `0.1` | `--test-split` |

=== "Model"

    | Variable | Default | Maps to |
    |----------|---------|---------|
    | `IN_CHANNELS` | `12` | `--in-channels` (stage 1 only) |
    | `BASE_CHANNELS` | `64` | `--base-channels` (stage 1) |
    | `LATENT_CHANNELS` | `64` | `--latent-channels` (stage 1) |
    | `NUM_RES_BLOCKS` | `2` | `--num-res-blocks` (stage 1) |
    | `NUM_EMBEDDINGS` | `512` | `--num-embeddings` (both stages; must match) |
    | `COMMITMENT_COST` | `0.25` | `--commitment-cost` (stage 1) |
    | `SEQ_LENGTH` | `5000` | `--seq-length` (both) |
    | `HIDDEN_DIM` | `128` | `--hidden-dim` (stage 2) |
    | `NUM_LAYERS` | `10` | `--num-layers` (stage 2). There are no variables for `--kernel-size` or `--dilation-cycle`; their script defaults (2 and 10) are used |
    | `VQVAE_CHECKPOINT` | unset | `--vqvae-checkpoint`; required for `2`, set automatically by `both` |

=== "Training"

    | Variable | Default | Maps to |
    |----------|---------|---------|
    | `LR_STAGE1` | `0.0001` | `--lr` (stage 1) |
    | `LR_STAGE2` | `0.001` | `--lr` (stage 2) |
    | `MAX_EPOCHS_STAGE1` | `100` | `--max-epochs` (stage 1) |
    | `MAX_EPOCHS_STAGE2` | `100` | `--max-epochs` (stage 2) |
    | `ACCELERATOR` | `gpu` | `--accelerator` |
    | `DEVICES` | `0` | `--devices` (space-separated IDs, e.g. `"0 1"`) |
    | `LOG_EVERY_N_STEPS` | `50` | `--log-every-n-steps` |
    | `CHECK_VAL_EVERY_N_EPOCH` | `1` | `--check-val-every-n-epoch` |
    | `GRADIENT_CLIP` | `1.0` | `--gradient-clip` |
    | `PATIENCE` | `10` | `--patience` |
    | `SAVE_TOP_K` | `3` | `--save-top-k` |
    | `VIZ_EVERY_N_EPOCHS` | `5` | `--viz-every-n-epochs` (stage 1) |
    | `VIZ_NUM_SAMPLES` | `4` | `--viz-num-samples` (stage 1) |

=== "Weights & Biases"

    | Variable | Default | Effect |
    |----------|---------|--------|
    | `WANDB_ENABLED` | **`true`** | Adds `--wandb`. Only the exact string `true` enables it. |
    | `WANDB_PROJECT` | `ecg-vqvae` | `--wandb-project` |
    | `WANDB_ENTITY` | empty | `--wandb-entity` if non-empty |
    | `WANDB_RUN_NAME` | empty | `--wandb-run-name`; stage 2 gets `<name>_stage2` |
    | `WANDB_TAGS` | empty | `--wandb-tags` (space-separated) |

The wrapper has no variable for `--skip-missing-check`, `--b1` or `--b2`. Call the Python script directly to use them.

### Examples

```bash
# Full dataset, larger codebook
DATA_DIR=/path/to/mimic-iv-ecg MAX_SAMPLES=null \
MAX_EPOCHS_STAGE1=200 MAX_EPOCHS_STAGE2=200 BATCH_SIZE=64 NUM_EMBEDDINGS=1024 \
./run_train_vqvae.sh both

# Stage 2 from a chosen checkpoint
VQVAE_CHECKPOINT=runs/vqvae_mimic_standalone/seed_42/checkpoints/last.ckpt \
./run_train_vqvae.sh 2

# Codebook-size sweep (stage 1 only)
for K in 256 512 1024 2048; do
  NUM_EMBEDDINGS=$K EXP_NAME_STAGE1="vqvae_emb${K}" ./run_train_vqvae.sh 1
done
```

## Running the Python script directly

```bash
# Stage 1
python train_vqvae_standalone.py --stage 1 \
    --data-dir /path/to/mimic-iv-ecg --exp-name vqvae_exp \
    --batch-size 32 --max-epochs 100 --num-embeddings 512

# Stage 2
python train_vqvae_standalone.py --stage 2 \
    --data-dir /path/to/mimic-iv-ecg --exp-name prior_exp \
    --vqvae-checkpoint runs/vqvae_exp/seed_42/checkpoints/last.ckpt \
    --num-embeddings 512
```

### CLI reference

| Flag | Type / default | Notes |
|------|----------------|-------|
| `--stage` | `1` or `2`, **required** | |
| `--exp-name` | `vqvae_mimic` | Run dir: `<runs-root>/<exp-name>/seed_<seed>/` |
| `--seed` | `42` | Seeds Python, NumPy, torch, and the data split |
| `--runs-root` | `runs` | |
| `--data-dir` | **required** | Must contain `machine_measurements.csv` |
| `--batch-size` | `32` | |
| `--num-workers` | `4` | |
| `--max-samples` | `None` | Per split (train **and** val) |
| `--skip-missing-check` | flag | Skip the `.hea`/`.dat` existence scan |
| `--val-split` / `--test-split` | `0.1` / `0.1` | Subject-level fractions |
| `--in-channels` | `12` | Stage 1; also the dataset lead count. Stage 2 hard-codes 12 leads |
| `--base-channels` | `64` | Stage 1 |
| `--latent-channels` | `64` | Stage 1; also the codebook vector size |
| `--num-res-blocks` | `2` | Stage 1 |
| `--num-embeddings` | `512` | Codebook size; must match between stages |
| `--commitment-cost` | `0.25` | Stage 1 |
| `--seq-length` | `5000` | Must be divisible by 8 |
| `--hidden-dim` | `128` | Stage 2 |
| `--num-layers` | `10` | Stage 2: number of `CausalGatedBlock` layers |
| `--kernel-size` | `2` | Stage 2: kernel size of the causal convolutions |
| `--dilation-cycle` | `10` | Stage 2: layer `i` uses dilation `2 ** (i % dilation_cycle)` |
| `--vqvae-checkpoint` | `None` | **Required** for stage 2 |
| `--lr` | `1e-4` | In stage 2, a value of exactly `1e-4` is silently replaced by `1e-3` |
| `--b1` / `--b2` | `0.9` / `0.999` | Adam betas |
| `--max-epochs` | `100` | |
| `--accelerator` | `gpu` | |
| `--devices` | `0` (list of ints, `nargs="+"`) | |
| `--log-every-n-steps` | `50` | |
| `--check-val-every-n-epoch` | `1` | |
| `--gradient-clip` | `1.0` | `gradient_clip_val` |
| `--patience` | `10` | Early stopping on `val_loss` |
| `--save-top-k` | `3` | Checkpoints ranked by `val_loss` |
| `--viz-every-n-epochs` | `5` | Stage 1 |
| `--viz-num-samples` | `4` | Accepted but **unused**: the callback plots a single validation example |
| `--wandb` | flag | |
| `--wandb-project` | `ecg-vqvae` | |
| `--wandb-entity` | `None` | |
| `--wandb-run-name` | `None` → `<exp-name>_seed<seed>` | |
| `--wandb-tags` | `None` → `vqvae ecg stage1` / `vqvae prior pixelcnn stage2` | `nargs="*"` |

!!! note "There is no `--resume` flag"
    The older README shows `--resume ...`, but the script does not define it, and argparse will reject it. `trainer.fit` is always called without `ckpt_path`.

!!! warning "CPU runs"
    `--devices` is always parsed as a list of ints. Lightning's CPU accelerator rejects a list (`devices selected with CPUAccelerator should be an int > 0`), so `--accelerator cpu` does not currently work with this script.

## What the script does

### Data

The standalone file has its own `MIMICIVECGDataset`, similar to but **not identical to** the packaged [`data.datasets.mimic.MIMICIVECGDataset`](../datasets/mimic.md):

- Splits by `subject_id` with two seeded `train_test_split` calls (test first, then val from the remainder). This gives the same subject sets as the packaged dataset only when the input rows are identical, and they are not, because the packaged dataset drops NaN feature rows before splitting.
- **Does not** drop rows with NaN measurements. Feature statistics use `nanmean`/`nanstd`.
- The missing-file check requires both `.hea` **and** `.dat`, and prints how many it found. If the split is empty after filtering and `max_samples`, the constructor raises `ValueError: Dataset is empty after filtering!` with hints.
- If a record fails to load, it prints the error and returns an **all-zero** ECG instead of raising.
- Extra leads are truncated, but fewer than `num_leads` leads are **not** padded. Time is zero-padded or truncated to `seq_length`.
- Per-sample normalization `(x - x.mean()) / (x.std() + 1e-6)` over all leads.
- Returns `(ecg[num_leads, seq_length], features[9])`.

`VQVAEMIMICDataModule` builds only train and val splits (no test), with `shuffle=True` for train and `pin_memory=True`.

### Stage 1: VQ-VAE

```text
x [B, 12, 5000]
 └─ Encoder1D: conv7 → (2 ResBlocks per level, stride-2 conv between levels) × 4 levels, mults (1,2,4,4)
               → mid ResBlock ×2 → GroupNorm → SiLU → conv3 → 2·latent channels; first half kept
 z [B, 64, 625]          (3 downsamplings: 5000 / 8 = 625)
 └─ VectorQuantizer (K = 512 codes, dim 64, nearest neighbour, straight-through)
 z_q [B, 64, 625], indices [B, 625]
 └─ Decoder1D (mirror, ConvTranspose1d upsampling)
 x̂ [B, 12, 5000]
```

With default settings the model has about 6.4 M parameters.

Loss (`vqvae_loss`): `MSE(x̂, x) + vq_loss`. In the code,

```python
codebook_loss   = F.mse_loss(quantized.detach(), z)      # gradient flows to the encoder
commitment_loss = F.mse_loss(quantized, z.detach())      # gradient flows to the codebook
vq_loss = codebook_loss + commitment_cost * commitment_loss
```

So `commitment_cost` (β) scales the term that updates the **codebook**, and the encoder-side term has weight 1. This is the reverse of the usual VQ-VAE convention (`‖sg[z] − e‖² + β‖z − sg[e]‖²`). Keep this in mind when tuning `--commitment-cost`. The packaged implementation has the same behaviour.

Optimizer: Adam(`lr`, `(b1, b2)`), with no scheduler.

### Stage 2: PixelCNN prior

```text
indices [B, 625]  (from the frozen stage-1 encoder, computed on the fly each batch)
 └─ Embedding(K, hidden_dim) → shift right by one (zeros at t = 0)
 └─ CausalGatedBlock × num_layers   (left-padded causal conv, kernel 2, dilation 2^(i % dilation_cycle) = 1, 2, …, 512;
                                     tanh·sigmoid gate → 1×1 residual (+ input) and 1×1 skip)
 └─ Σ skips → ReLU → Conv1d 1×1 → ReLU → Conv1d 1×1
 logits [B, K, 625]   (logit t predicts code t from codes < t; receptive field 1024 codes)
```

With default settings the prior has about 1.14 M parameters. `sample()` draws the 625 codes one at a time and feeds each step only the last `receptive_field` codes.

Loss: `prior_loss(logits, indices)`, the cross-entropy of each code given the codes before it (no target shift, so training matches sampling). The stage-1 model is frozen (`eval()` + `freeze()`).

!!! warning "Old stage-2 checkpoints"
    Prior checkpoints saved before the switch from mask-A/B `GatedMaskedConv1d` layers to `CausalGatedBlock` have different `state_dict` keys and cannot be loaded. Retrain stage 2. Stage-1 checkpoints are unaffected.

### Callbacks and logging

| | Stage 1 | Stage 2 |
|---|---|---|
| `ModelCheckpoint` | `epoch{epoch:03d}-step{step:06d}.ckpt`, `last.ckpt`, top-k by `val_loss` | same |
| `EarlyStopping` | `val_loss`, `patience` | same |
| `LearningRateMonitor` | per step | per step |
| Reconstruction plots | `samples/epoch_XXXX.png` every `viz-every-n-epochs` (real vs. recon of the first val example, up to 8 leads) | — |
| Trainer | `deterministic=True` | `deterministic=True` |

Logged metrics:

- **Stage 1:** `train/total_loss`, `train/recon_loss`, `train/vq_loss`, `train/unique_codes` and `train/codebook_usage` (every 100th batch), `val/total_loss`, `val/recon_loss`, `val/vq_loss`, `val/unique_codes`, `val/codebook_usage`, `val_loss`.
- **Stage 2:** `train/loss`, `val/loss`, `val_loss`.

Useful signals: `val/recon_loss` should go down, and `val/codebook_usage` (unique codes / K per batch) shows codebook collapse when it stays very low.

### Outputs

```text
<runs-root>/
├── vqvae_mimic_standalone/seed_42/
│   ├── checkpoints/   epoch000-step000032.ckpt … last.ckpt
│   ├── samples/       epoch_0000.png, epoch_0005.png, …
│   ├── tb/version_N/  TensorBoard events
│   └── wandb/         only if a W&B run was created
└── prior_mimic_standalone/seed_42/
    ├── checkpoints/
    ├── samples/       (created but unused)
    └── tb/
```

## Weights & Biases

```bash
pip install wandb && wandb login           # once
WANDB_PROJECT=my-ecg WANDB_ENTITY=my-team WANDB_TAGS="sweep v2" ./run_train_vqvae.sh 1
WANDB_ENABLED=false ./run_train_vqvae.sh both   # disable (the wrapper enables W&B by default)
WANDB_MODE=offline ./run_train_vqvae.sh both    # offline; later: wandb sync runs/<exp>/seed_42/wandb/
```

When `--wandb` is set, the script creates a `WandbLogger` with `save_dir=<run_dir>` and a config dict containing the stage, the architecture, `lr`, `batch_size`, `max_epochs`, `seed` and, for stage 2, `vqvae_checkpoint`. If initialization raises an error, or `wandb` is not installed, it prints a warning and continues with TensorBoard. TensorBoard is always on.

!!! bug "Stage 1 metrics do not reach W&B"
    In `train_stage1_vqvae` the trainer is built with `logger=tb_logger` instead of `logger=loggers_list`. The `WandbLogger` is constructed but never attached, so stage-1 metrics go to TensorBoard only. Stage 2 passes `loggers_list` and logs to both.

## Loading checkpoints and sampling

`VQVAELightning` and `PriorLightning` rebuild their config from the flat hyperparameters stored in the checkpoint, so a plain `load_from_checkpoint(path)` works (passing `config=` explicitly is still accepted). When stage 2 loads the stage-1 checkpoint, it records the VQ-VAE's config as `vqvae_config` in the prior's hyperparameters; together with the `vqvae.*` weights already stored in the prior checkpoint, this makes the prior checkpoint self-contained, so loading it does not read the stage-1 file. (Prior checkpoints saved before `vqvae_config` was added still load the VQ-VAE from the stored `vqvae_checkpoint` path.)

`PriorLightning.sample()` derives the latent length from the VQ-VAE (`VQVAELightning.latent_length(seq_length)`, a downsampling factor of 8 by default), so samples have the requested length. `seq_length` must be a multiple of that factor; otherwise it raises `ValueError`.

```python
# run from inside single_file/
import torch
from train_vqvae_standalone import VQVAELightning, PriorLightning

# Stage 1 on its own (reconstruction, encode_to_indices, decode_from_indices)
vqvae = VQVAELightning.load_from_checkpoint(
    "runs/vqvae_mimic_standalone/seed_42/checkpoints/last.ckpt", map_location="cpu"
).eval()

# Prior: also restores its frozen VQ-VAE from the prior checkpoint
prior = PriorLightning.load_from_checkpoint(
    "runs/prior_mimic_standalone/seed_42/checkpoints/last.ckpt", map_location="cuda"
).eval()

samples = prior.sample(n_samples=16, seq_length=5000, temperature=1.0)  # [16, 12, 5000], per-sample normalized units
torch.save(samples.cpu(), "generated_ecgs.pt")
```

## Differences from the packaged implementation

| Aspect | `single_file/` | Packaged (`models/vqvae/`, `scripts/train/train_vqvae_mimic.py`, `train_prior_mimic.py`) |
|--------|----------------|------------------------------------------------------------|
| Code location | One file, no repo imports | `models.vqvae`, `data.datasets.mimic`, `training.callbacks`, `utils.seed` |
| Installation | none (plain `pip` deps) | `pip install -e .` |
| Configuration | CLI flags / env vars only | `--config configs/experiments/{vqvae,prior}_mimic.yaml` plus CLI flags |
| Entry point | one script with `--stage 1` or `--stage 2`; `run_train_vqvae.sh` with `1`, `2` or `both` | separate scripts per stage; `scripts/shell/run_train_vqvae_mimic.sh`, `run_train_vqvae_prior_mimic.sh` |
| Dataset | own copy (keeps NaN-feature rows, zero ECG on read errors, no lead padding, train/val only) | `MIMICIVECGDataset` (drops NaN-feature rows, raises on read errors, pads leads, train/val/test) |
| Batch format | tuple only | VQ-VAE/prior accept tuple **or** `{"ecg_signals": ...}` |
| Visualization | minimal built-in callback (1 example, PNG only) | `training.callbacks.visualization.VAEVisualizationCallback` (`num_samples`, overlay/all-leads, TensorBoard/W&B logging) |
| Resume | not supported | `--resume <ckpt>` |
| W&B | stage 1 logger not attached (bug) | attached via `logger=loggers_list` |
| Default run names | `vqvae_mimic_standalone`, `prior_mimic_standalone` (wrapper) | `vqvae_mimic`, `prior_mimic` |
| Shared behaviour | same architecture, defaults, metric names, checkpoint naming, config-from-hyperparameters checkpoint loading (self-contained prior checkpoints), and sampling via `VQVAELightning.latent_length()` | |

The `nn.Module` attribute names match (`vqvae.encoder`, `vqvae.vq`, `vqvae.decoder`, `prior.*`), so state-dict weights can in principle be moved between the two by loading the `state_dict` into the other class. `load_from_checkpoint` across implementations is not supported. For the packaged workflow, see [VQ-VAE](../models/vqvae.md) and [VQ-VAE training](../training/vqvae_training.md).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Data directory not found` (wrapper) / `ERROR: Data directory does not exist` | Set `DATA_DIR` / `--data-dir` to the folder containing `machine_measurements.csv`. |
| `Dataset is empty after filtering!` | Run `test_data_loading.py`. Check that `files/p*/p*/s*/*.hea` and `.dat` exist. Try `--skip-missing-check` (Python script only). |
| `TypeError: Unexpected VQVAEConfig fields: [...]` (or `PriorConfig`) | A keyword argument passed to the constructor or `load_from_checkpoint` is not a config field. Check its spelling. |
| `devices selected with CPUAccelerator should be an int > 0` | CPU runs are not supported by the current `--devices` parsing. |
| CUDA out of memory | Lower `BATCH_SIZE`, or `BASE_CHANNELS`/`LATENT_CHANNELS`. |
| Low `codebook_usage` | Try a smaller `NUM_EMBEDDINGS`. Note the [β convention](#stage-1-vq-vae) before tuning `COMMITMENT_COST`. |
| Slow start on the full dataset | The file-existence scan touches every record. Use `--skip-missing-check` once the data is known to be complete. |
| `unrecognized arguments: --resume` | Not supported here. Use the packaged scripts. |

## Corrections to the older notes

The Markdown files in `single_file/` predate several changes. Compared with the code:

- **`WANDB_ENABLED`**: `WANDB_GUIDE.md` lists the default as `false`. The wrapper default is `true`.
- **W&B support**: `FIXES.md` says W&B "was never included". It is included, but stage 1 does not attach the logger.
- **`--resume`**: shown in `README.md`, but the flag does not exist.
- **Sampling snippet**: works as written, returning `[16, 12, 5000]`. See [above](#loading-checkpoints-and-sampling).
- **Low codebook usage → "increase commitment cost"**: with the β placement in this code, β scales the codebook update, not encoder commitment.
- The README omits `LOG_EVERY_N_STEPS` and `CHECK_VAL_EVERY_N_EPOCH`, which the wrapper does support.

## Citation

```bibtex
@inproceedings{oord2017neural,
  title     = {Neural Discrete Representation Learning},
  author    = {van den Oord, Aaron and Vinyals, Oriol and Kavukcuoglu, Koray},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
```
