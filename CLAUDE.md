# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ECGEN is a research framework for generative modeling of 12-lead ECGs (MIMIC-IV-ECG, PTB-XL), built on PyTorch Lightning. Model families (all conditional-capable): VAE, VQ-VAE + PixelCNN prior, Pulse2Pulse GAN, diffusion and flow matching (pixel and latent space).

The repo also ships the `ecgen` package (`src/ecgen/`) published to PyPI — `ecgen.generate`, `ecgen.plot_ecg`, and HF upload helpers; see "Published `ecgen` package" below.

## Commands

```bash
pip install -e ".[dev]"     # installs src/ecgen (+ pytest, black, flake8, mypy); extras: [hf], [plot], [all]

# pytest also sees the top-level research packages via [tool.pytest.ini_options] pythonpath = ["."].
pytest tests/                                          # all tests (no MIMIC needed)
pytest tests/test_models/test_vae.py::<test_name>      # single test
black . && flake8 .

# Training: one runner for every family; dotted key=value overrides win over the YAML
ecgen train configs/experiments/vae_mimic.yaml [key=value ...] [--resume ckpt] [--print-config]
ecgen train configs/experiments/vqvae_mimic.yaml                     # VQ-VAE stage 1
ecgen train configs/experiments/prior_mimic.yaml model.params.vqvae_checkpoint=<stage1.ckpt>
ecgen train configs/experiments/pulse2pulse_mimic.yaml
ecgen list                                             # registered families / datasets / callbacks
ecgen sample <ckpt> -n 16 -o out/ --format csv [--cond field=v1,v2] [--guidance 2] [--steps 50]
ecgen evaluate out/samples.npy <config.yaml>           # Fréchet / MMD / precision-recall vs real test ECGs
python -m ecgen train ...                              # same without the console script
./scripts/shell/run_train_*.sh [config] [key=value ...]  # thin wrappers around the runner
```

Training runs on GPU against MIMIC-IV-ECG at a hard-coded cluster path (`data.dataset.params.root` in the experiment YAMLs). Use `data.dataset.params.max_samples=1000` for quick runs, or `data.dataset.name=synthetic` (plus `trainer.accelerator=cpu`) to run without MIMIC.

## Architecture

`PLAN_TRAINING_LIBRARY.md` is the roadmap; Phases 0–8 are done (released as v1.1.0) and the shims were removed afterwards. Read §3–§7 before adding a family, conditioner, or dataset.

**Where code lives.**
- **`src/ecgen/` (imported as `ecgen.*`, the published package)** now holds the training stack: `ecgen.families.{vae,vqvae,gan}` (Lightning modules, configs, losses), `ecgen.families.base` (`GenerativeModule`, `Capabilities`), `ecgen.nn` (shared conv blocks), `ecgen.data` (datasets, `ECGDataModule`, collation, lead handling, `FeatureNormalizer`), `ecgen.training.{runner,callbacks}`, `ecgen.registry`, `ecgen.types`, `ecgen.cli`, plus `generate`/`plot`/`f`.
- **Top-level research code** (not installed): `evaluation/`, `visualization/`, `training/metrics/`, `training/losses/common.py`, `scripts/`. The old root shim packages (`models/`, `data/`, `utils/`, `training/callbacks|trainers`) were removed after the v1.1.0 release; import from `ecgen.*`. `ecgen.models.{vae,pulse2pulse}` is a deprecated alias of `ecgen.families` that warns on import (removal in 2.0) — don't import it from library code. Don't add top-level packages to the wheel — their generic names (`models`, `data`, `utils`) would clash in site-packages.
- **Legacy tuple loaders** `ecgen.data.mimic_dataset.MIMICIVECGDataset` and `ecgen.data.pulse2pulse_mimic` remain for old scripts; new code uses `ecgen.data.datasets.MIMICIVECG`.
- **`single_file/`:** a self-contained VQ-VAE (both stages) in `train_vqvae_standalone.py`; independent of the package — changes to `ecgen.families.vqvae` are not reflected there and vice versa.

**Conditioning** (`ecgen.conditioning`, docs/training/conditioning.md): a conditional family takes `model.params.conditioning: {vector_dim, token_dim, encoders: [{field, name, params}], dropout: {field: p}}`, calls `self.init_conditioning(config.conditioning)` in `__init__`, and gets a `ConditionOutput` (`vector` / `tokens` / `dense`) from `self.condition(batch)` in training or `self.encode_condition(n, cond=raw, obs=, obs_mask=, drop=)` at sampling time. Encoders: `categorical`, `multilabel`, `numeric`, `fourier`, `masked_signal` (field `obs`, needs `data.masks`), `text_hash`, `text` (frozen HF model held outside the module tree so its weights never enter checkpoints; `ecgen[text]`). String fields (reports) collate to `list[str]`. Dropped/absent fields use learned null embeddings (`drop="all"` = the unconditional CFG pass). Injection layers in `ecgen.nn.injection` are zero-initialised. Families declare `supports.conditional` and `accepts` (slots); `init_conditioning` and the runner validate both. The runner calls `model.set_data_info(datamodule.state_dict())`; `data_info` (lead names, fs, per-field normalizers) is saved in checkpoints as `ecgen_data_info`, so `prepare_condition` takes raw units. Subclasses overriding `on_load_checkpoint` must call `super()`.

**Conditional families** (docs/training/conditioning.md#conditional-families): `vae` (CVAE: FiLM in encoder/decoder, dense concat, `learned_prior`, `latent_type: spatial|global` — only `global` gives coherent prior samples; spatial is an autoencoder/first stage), `pulse2pulse` (FiLM generator + projection critic with its own `d_conditioner`; dense `masked_signal` = paired lead reconstruction with `l1_weight`; `output_scale` stretches the tanh output — use ~8 for z-scored data), `vqvae_prior` (conditional gated blocks, `sample(guidance=w)` = CFG on logits). Without `conditioning` all three keep their original parameters. `fixed_seq_length` (P2P, global VAE) is checked by the runner. Never call `next(module.parameters())` inside a training step path for modules that may have no parameters: the `StopIteration` silently ends the Lightning epoch. Condition adherence: `ecgen.evaluation.measure.heart_rate` + `scripts/evaluate/condition_adherence.py`.

**Continuous families** (`ecgen.families.continuous`, docs/models/diffusion.md): `ContinuousGenerativeModule` = process (`ecgen.processes`, registry `PROCESSES`: `ddpm`, `rectified_flow`) + backbone (`ecgen.nn.backbones`, registry `BACKBONES`: `unet1d`, `dit1d`, `ssm` (S4D), `mamba` (needs mamba-ssm), `forward(x, t, cond)`, declares `accepts`) + conditioner; registered as `diffusion` (default process `ddpm`) and `flow_matching` (`FlowMatchingConfig`, default process `rectified_flow`; Euler/Heun ODE sampling, optional minibatch OT coupling; `inpaint="replace"` only). It owns EMA (`ecgen.training.ema.EMA`, stored as `checkpoint["ema"]`, used by `sample(use_ema=True)`), AdamW + warm-up, `loss_on`, `clean_observed` (SSSD-style), CFG (`guidance`), and training-free `inpaint="replace"|"repaint"`. Latent variants `latent_diffusion` / `latent_flow_matching` (`ecgen/families/continuous/latent.py`) run the process on a frozen stage-1 `vae` (spatial, unconditional) or `vqvae`, store `autoencoder_config` + weights in the checkpoint, and fix `latent_scale` from the first batch; subclasses customise via the `to_model_space` / `from_model_space` / `model_space_shape` / `adapt_condition` hooks. A new diffusion/flow method is a new `Process` (`training_pair`, `sample`), not a new family. Evaluate imputation with `scripts/evaluate/lead_reconstruction.py`.

**Family pattern** (`ecgen/families/<family>/`): a dataclass config (`config.py`), a plain `nn.Module` (`*_1d.py`), `losses.py`, and a `GenerativeModule` subclass registered with `@FAMILIES.register("<name>")` that owns its own training logic (loss, optimizers, manual vs. automatic optimization) and `sample()`. Read batches with `self.get_signal(batch)` (accepts `ECGBatch` and the legacy dict/tuple formats). Registered names: `vae`, `vqvae`, `vqvae_prior`, `pulse2pulse`.

**VQ-VAE is two-stage:** `VQVAELightning` (encoder → `VectorQuantizer` → decoder) is trained first; `PriorLightning` then loads a frozen stage-1 checkpoint via `model.params.vqvae_checkpoint` and trains a `PixelCNNPrior` over the discrete code indices. The prior is WaveNet-style (embedding shifted right by one, then dilated causal gated blocks), so logits at position t predict code t from codes < t — train with `PriorLightning.compute_loss`, never shift targets. Its `receptive_field` (1024 codes by default) should cover the latent length (625 for 5000 samples). Sampling = prior samples codes → VQ-VAE decodes.

**Configs and runner:** `configs/experiments/*.yaml` are the entry points, read by `ecgen.training.runner` (module docstring documents the schema). Sections: `experiment`, `model: {name|target, params}` (flat config-dataclass fields), `data: {dataset: {name|target, params}, batch_size, num_workers, split_params}`, `checkpoint`, `callbacks: [{name|target, params}]`, `trainer` (pl.Trainer kwargs), `wandb`, `resume`; unknown sections are rejected. Components are built from the registries in `ecgen.registry` (`FAMILIES`, `DATASETS`, `CALLBACKS`) by `name`, or by import path via `target`. Strings may use `${run_dir}`, `${name}`, `${seed}`; callbacks taking `save_dir` default to `${run_dir}/samples`. The runner checks the model's channel count / `seq_length` against the dataset before training.

**Datasets:** `mimic_iv_ecg` (`report: true` adds machine-report text), `ptbxl` (`ecgen/data/datasets/ptbxl.py`: `scp` multi-hot with `labels: all|superclass|subclass|diagnostic|form|rhythm`, `demographics`, `sex`, `report`, PTB-XL+ `features`; strat_fold 1–8/9/10; data at /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1, PTB-XL+ at /work/vajira/DATA/ptbxl_plus/physionet.org/files/ptb-xl-plus/1.0.1), `synthetic`.

**Data:** datasets subclass `ecgen.data.datasets.BaseECGDataset`, return dict samples `{"signal": [C,T], "cond": {...}, "cond_missing": {...}, "meta": {...}}`, and expose `lead_names`, `fs`, `seq_length`; `ECGDataModule` collates them into `ecgen.types.ECGBatch`. `MIMICIVECG` (`mimic_iv_ecg`) does the seeded subject-level split internally, fits the measurement normalizer on the train split only, and selects leads by name (`leads: null` = file order `I, II, III, aVR, aVF, aVL, V1–V6` — MIMIC stores aVF before aVL; aliases `standard12`, `independent8`). `scale: zscore` (per record, legacy) / `none` (mV) / number. `missing: drop` (legacy) vs `mask` (keeps rows; changes the split). `SyntheticECG` (`synthetic`) is for tests and smoke runs.

**Outputs:** runs go to `runs/<exp_name>/seed_<seed>/` (`checkpoints/epoch###-step######.ckpt` plus `last.ckpt` — there is no `best.ckpt`; `samples/`, `metrics/train_val_metrics.json`, `tb/`, `config_resolved.yaml`, `metadata.json`). Each checkpoint also stores the `ECGDataModule` state (lead names, fs, seq_length, feature normalizer). W&B logging is optional via the `wandb` section.

**Checkpoint loading:** wrappers save the config as flat hyperparameters (`save_hyperparameters(config.__dict__)`) and accept `config=None, **kwargs` via `ecgen.utils.config.resolve_config`, so plain `X.load_from_checkpoint(path)` works; keep that signature on new wrappers. `PriorLightning` records the stage-1 config in `vqvae_config`, so prior checkpoints restore the frozen VQ-VAE from their own weights. `tests/test_models/test_checkpoint_loading.py` and `tests/test_training/test_runner.py` cover this.

**Known pitfalls:** The VQ-VAE/VAE downsample by `2 ** (len(channel_multipliers) - 1)` (8× by default) — use `VQVAELightning.downsample_factor` / `latent_length()` rather than hard-coding it. PyYAML reads `1e-3` as a string; the runner's override parser converts it, but in YAML files write `0.001`. Pulse2Pulse configs use `leads: independent8` (I, II, V1–V6, the CSV columns `ecgen.generate` writes); checkpoints trained by the old script saw the first 8 file leads (I, II, III, aVR, aVF, aVL, V1, V2) instead.

## Documentation

MkDocs Material site in `docs/` (config: `mkdocs.yml`; API pages in `docs/api/` are rendered by mkdocstrings from Google-style docstrings). Preview with `pip install -r requirements-docs.txt && mkdocs serve`; check with `mkdocs build --strict` **without `-q`** (quiet mode hides the warnings that make `--strict` fail; griffe warns on grouped docstring params like `a, b: ...` — give each parameter its own line); CI (`.github/workflows/docs.yml`) runs `mkdocs build --strict` and publishes to the `gh-pages` branch on pushes to `main`/`simple`. New pages must be added to `nav` in `mkdocs.yml`.

## Published `ecgen` package

- **Sampling any trained model:** `ecgen.sample(ckpt, n, cond=..., obs=..., guidance=..., steps=..., output_dir=, format=npy|csv|wfdb, derive_12=, plot=)` / `ecgen.load_model(ckpt)` (`src/ecgen/api.py`); checkpoints record `ecgen_family` and `ecgen_data_info` (incl. `scale`). A condition value with a leading dim of `n` (or `n` strings, or `n` ints) is per-sample, otherwise broadcast.
- **Evaluation:** `ecgen.evaluation` (`measure`, `features`, `metrics`, `classifier`): distribution metrics, HR / label adherence, TSTR, reconstruction, inter-lead correlation error, Einthoven residual, memorisation. `ecgen.data.leads.derive_standard12` completes 8 independent leads to 12.

- **Generation:** `ecgen.generate(model_path, n_samples, output_dir, ...)` (`src/ecgen/generate.py`) loads a Pulse2Pulse checkpoint and writes one CSV per sample (5000 rows × 8 leads `I, II, V1–V6`), optionally with a plot. Original Pulse2Pulse-repo checkpoints use `Conv1dTrans`/`ppfilter1` keys, which `_remap_state_dict()` rewrites. `hf://<repo_id>/<path>` model paths go through `_resolve_model_path()`, which uses `HF_TOKEN` from the environment or a `.env` in CWD. Use `denorm=6000.0` for models trained on `ECGDataSimple`.
- **Plots:** `ecgen.plot_ecg` (`src/ecgen/plot.py`) renders an ECG grid sheet as self-contained D3 HTML or SVG (pure Python), or PDF via `cairosvg` (`ecgen[plot]`).
- **HF upload:** `ecgen.f.upload_hf(config_or_dict, dry_run=...)` / `upload_hf_single(...)` (`src/ecgen/f/upload_hf.py`, needs `ecgen[hf]`); manifest in `configs/upload_checkpoints.yaml`; CLI `scripts/upload_checkpoints_to_hf.py [--dry-run]`. `upload_hf` raises `RuntimeError` on failure rather than exiting.
- **Releasing:** the version comes from git tags via `setuptools-scm` (no `__version__` literal; `src/ecgen/_version.py` is generated and gitignored). Pushing a `v*` tag runs `.github/workflows/publish.yml`, which publishes to PyPI with OIDC trusted publishing. Don't add the top-level research packages to the wheel — their generic names (`models`, `data`, `utils`) would clash in site-packages.
- `tests/test_upload_hf.py` covers the upload helpers.
