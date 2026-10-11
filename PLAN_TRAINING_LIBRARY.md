# Plan: Training and Conditioning as Part of the `ecgen` Library

Status: **proposal, to be executed later** · Written: 2026-10-11 · Branch at time of writing: `docs/visual-guide`

## 1. Question and verdict

**Question.** Can training (not only generation) live inside the `ecgen` library, given that
(a) each generator family trains differently (VAE, GAN, VQ-VAE + prior, diffusion, flow matching, …), and
(b) each conditioning type (tabular, measured parameters, partial ECG, leads, text, …) also changes
training? And can the design stay open to new approaches later?

**Verdict: yes, it is feasible.** One rule makes it work: **do not try to write one universal
training loop.** Split the problem into independent, pluggable axes, and let each axis own only
what actually differs:

| Axis | Owns | Examples |
|---|---|---|
| **Family** (objective + sampler) | `training_step` logic, loss, optimizers, `sample()` | VAE, GAN, VQ-VAE+AR prior, DDPM/DDIM, flow matching, latent diffusion |
| **Backbone** (network) | `forward(x, t, cond)` | UNet1d, WaveGAN/Pulse2Pulse U-Net, DiT-1d, S4/Mamba, WaveNet |
| **Conditioner** (encode conditions) | raw condition fields → standard `ConditionOutput` | tabular MLP, label embedding, text encoder, masked-signal encoder |
| **Injection** (how cond enters backbone) | layers inside the backbone | concat, FiLM / AdaGN / AdaLN-Zero, cross-attention, projection discriminator |
| **Condition-time strategy** | conditioning at train vs sample time | condition dropout + CFG, masked training, RePaint, classifier/DPS guidance |
| **Data** | dataset → `ECGBatch` (signal + condition fields + masks) | MIMIC-IV-ECG, PTB-XL, custom CSV |
| **Runner** | Lightning `Trainer`, run dirs, logging, checkpoints | one YAML-driven runner for all |

PyTorch Lightning already splits things this way. The `Trainer` (the loop) is generic; the
`LightningModule` (the training logic) is per family. This repo already uses that split:
`Pulse2PulseGAN` does its own WGAN-GP manual optimization, `VAELightning` and `VQVAELightning`
use automatic optimization, and `PriorLightning` is a frozen two-stage setup. There is also already
a generic, YAML-driven runner (`training/trainers/gan_trainer.py`, duplicated as
`src/ecgen/training/train.py`) that builds the model, data and callbacks from `target:` import strings.
So the "different training methods" problem is already solved at the right layer. What is missing is:

1. a **shared contract**: batch format, condition format and a `sample()` signature. Today the
   batch format is inconsistent: `MIMICIVECGDataset` returns tuples, while `VAELightning` and
   Pulse2Pulse expect dicts.
2. a **conditioning subsystem** that is separate from the families;
3. one **runner and registry** in place of four ~450-line copy-paste train scripts;
4. a **packaging decision**: the research packages (`models`, `data`, …) cannot be put into the wheel
   under those generic names (see §3).

### What is *not* possible or sensible, and the reasons

These limits do not block the plan. Each one is reflected in the design.

- **A single generic `training_step` for every family is not possible.** GANs need alternating
  optimizers and an `n_critic` schedule. Diffusion and flow need timestep sampling and EMA. VQ-VAE
  needs two stages with a frozen encoder. Distillation or consistency models need a teacher.
  → Each family keeps its own `LightningModule`, and they share a thin base class.
- **Not every conditioning mechanism works with every family.** For example, classifier-free
  guidance needs a model whose output can be mixed across conditional and unconditional passes
  (diffusion, flow, AR prior), so it does not apply to a one-shot GAN generator. RePaint-style
  inpainting needs an iterative sampler. → Use a **compatibility matrix** (§6) that is checked
  when the config is built, so the error is clear instead of a silent wrong result.
- **Some conditioning makes the data contract dataset-specific.** MIMIC has 9 machine measurements
  and free-text machine reports. PTB-XL has SCP labels, age, sex, height and weight. → Condition
  fields are declared per dataset with a `ConditionSpec`. The library does not hard-code them.
- **Raw data and credentialed datasets cannot ship with the library** (PhysioNet credentialing for
  MIMIC-IV-ECG). Only loaders and preprocessing ship.
- **Text conditioning needs heavy optional dependencies** (`transformers`). → Put them in an
  optional extra (`ecgen[text]`), never in the core install.

## 2. Current state (inventory used for this plan)

- Families: `models/vae/` (VAE), `models/vqvae/` (VQ-VAE + `PixelCNNPrior`, two-stage),
  `models/gan/pulse2pulse.py` (WGAN-GP, U-Net generator). `models/diffusion/` and `models/ssm/` are
  empty placeholders.
- Pattern: config dataclass + `nn.Module` + `LightningModule` wrapper; `utils.config.resolve_config`
  makes `load_from_checkpoint` work from flat hyperparameters. **Keep this**: every new module
  must keep `config=None, **kwargs`.
- Data: `MIMICIVECGDataset` returns `(ecg[12,5000], features[9])`. The 9 machine measurements
  (`rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis`) are already
  a ready-made tabular condition, but **no model uses them yet**.
- Known data issues that matter for conditioning (fix these in Phase 0):
  - `compute_feature_stats()` runs **per split**, so val and test features are normalized with
    their own statistics. A conditional model would see shifted conditions at evaluation time.
    Statistics must be fit on train and reused.
  - Per-sample ECG normalization (`(x - mean)/std` per record) throws away absolute amplitude, so
    amplitude-related conditions (e.g. voltage criteria for LVH) cannot be learned. Conditional
    models need a fixed physical scale (mV, optional global constant).
  - Rows with missing measurements are dropped (`dropna`). Conditioning should instead allow
    missingness, using a value plus a missing mask.
- Train scripts: `scripts/train/train_{vae_mimic,vqvae_mimic,prior_mimic,pulse2pulse}.py`. Each one
  defines its own inline DataModule and argparse, and reads specific YAML keys. The YAML
  `callbacks:` section is ignored.
- Package: only `src/ecgen` is published. It contains an *older copy* of the VAE and Pulse2Pulse code
  plus `generate`, `plot` and `f.upload_hf`.

## 3. Key structural decision: where the library code lives

The research packages (`models/`, `data/`, `training/`, `utils/`) cannot go into the wheel as they
are, because their generic names would collide in `site-packages` (see CLAUDE.md). So training as part
of the library means **moving the code under the `ecgen.` namespace**.

**Recommendation:** grow the library inside `src/ecgen/` as these subpackages:

```
src/ecgen/
  registry.py              # Registry[T]: register("name")(cls), build({"name"|"target", params})
  types.py                 # ECGBatch, ConditionOutput, SampleRequest (dataclasses)
  data/
    datasets/  base.py  mimic.py  ptbxl.py  csv_folder.py
    fields.py              # ConditionSpec / FieldSpec declarations (what a dataset can emit)
    transforms.py          # scaling (mV / fixed), lead select/reorder, crop, resample, filter
    normalizers.py         # FeatureNormalizer fit on train → saved in checkpoint
    masks.py               # MaskGenerator: lead subsets, time spans, forecast, random points
    datamodule.py          # ECGDataModule(dataset=..., fields=[...], masks=..., batch_size...)
    collate.py             # → ECGBatch
  conditioning/
    base.py                # Conditioner(nn.Module) -> ConditionOutput; NullCondition
    encoders/              # categorical, multilabel, numeric(tabular, missing-aware),
                           # fourier-scalar, text(frozen LM), signal(masked ECG / leads),
                           # other-modality(PPG), identity/beat-template
    composite.py           # combine several encoders → one ConditionOutput
    dropout.py             # per-field condition dropout (enables CFG), learned null tokens
  nn/
    blocks.py  embeddings.py (timestep, sinusoidal/Fourier)
    injection.py           # ConcatInput, FiLM, AdaGN, AdaLNZero, CrossAttention1d, AddToTimeEmb
    backbones/ unet1d.py  wavegan.py  dit1d.py  wavenet.py  ssm.py
  families/
    base.py                # GenerativeModule(pl.LightningModule) – shared contract (§4.3)
    vae/  vqvae/  gan/  diffusion/  flow/  latent/
  sampling/
    schedulers.py  samplers.py (ancestral, DDIM, DPM-Solver, Euler/Heun ODE)
    guidance.py            # CFG, classifier guidance, DPS / reconstruction guidance
    inpainting.py          # replacement, RePaint, masked-conditional
  training/
    runner.py              # the single YAML-driven runner (merge of gan_trainer.py / train.py)
    callbacks/             # EMA, sample grids per condition, visualization, eval-on-val
    losses/
  evaluation/              # unconditional + conditional metrics (§8)
  cli.py                   # `ecgen train --config x.yaml`, `ecgen sample --ckpt ... --cond ...`
  generate.py  plot.py  f/ # existing public API, kept stable
```

Migration of the top-level packages works the same way as `MIGRATION_GUIDE.md`: move the code,
leave thin re-export shims (`models/vae/__init__.py` → `from ecgen.families.vae import *`) for one
release, then delete. `single_file/` stays independent. Keep `ecgen.generate()` backward
compatible for Pulse2Pulse checkpoints, and add a generic `ecgen.sample(ckpt, n, cond=...)` that
uses the registry.

*Alternative if moving is too disruptive:* keep the research layout and publish a second
distribution `ecgen-train` with a namespaced import root. This is not recommended because it keeps
the duplication between `models/` and `src/ecgen/models/` alive.

## 4. Core contracts

These four interfaces are what make the system extensible. Every family, backbone, conditioner and
dataset only talks to these.

### 4.1 `ECGBatch` (data → model)

```python
@dataclass
class ECGBatch:
    signal: Tensor                      # [B, C_leads, T], fixed physical scale
    lead_names: list[str]               # e.g. ["I","II","III","aVR",...]; order is a contract
    fs: int                             # sampling rate (Hz)
    cond: dict[str, Tensor]             # raw condition fields by name (see FieldSpec)
    cond_missing: dict[str, Tensor]     # [B] or [B, F] bool, True = missing
    obs_mask: Tensor | None             # [B, C, T] 1 = observed (partial-ECG / lead conditioning)
    meta: dict[str, Any]                # ids, split, etc. (never fed to model)
```

Datasets return dicts; `collate` builds `ECGBatch`. The tuple and dict inconsistency goes away.
Unconditional training simply has `cond == {}` and `obs_mask is None`.

### 4.2 `FieldSpec` / `ConditionSpec` (declare what a dataset can provide)

```yaml
data:
  name: mimic_iv_ecg
  params: {root: ..., scale: mv, seq_length: 5000, leads: all12}
  fields:
    - {name: measurements, kind: numeric, columns: [rr_interval, qrs_onset, qrs_end, ...],
       normalize: standard, missing: mask}
    - {name: report, kind: text, column: report_0..report_17, join: " "}
    - {name: sex, kind: categorical, num_classes: 2}         # PTB-XL
    - {name: scp, kind: multilabel, vocab: ptbxl_superclass} # PTB-XL
  masks: {kind: lead_subset, keep: [[I], [I, II], [II, V1]], p_unconditional: 0.1}
```

`FeatureNormalizer` statistics are fitted on the train split, stored in the DataModule state, and
**saved into the checkpoint**, so `sample(cond=raw_values)` works without the dataset.

### 4.3 `Conditioner` → `ConditionOutput` (conditions → network)

```python
@dataclass
class ConditionOutput:
    vector: Tensor | None   # [B, D]       global: labels, tabular, pooled text  → FiLM/AdaGN/AdaLN/+temb
    tokens: Tensor | None   # [B, N, D]    sequence: text tokens, per-field tokens → cross-attention
    tokens_mask: Tensor | None
    dense: Tensor | None    # [B, C', T]   time-aligned: masked ECG + mask, PPG, beat template → concat
```

Backbones declare the slots they consume (`accepts = {"vector", "tokens", "dense"}`), and the
builder checks this against the configured conditioners. A `CompositeConditioner` merges several
encoders: vectors are summed or concatenated then projected, tokens are concatenated, dense is
concatenated on channels. `ConditionDropout` replaces fields with learned null embeddings with
probability `p` during training. This is what CFG needs, and it also gives one model that works
with any subset of conditions at inference time.

### 4.4 `GenerativeModule` (family base class)

```python
class GenerativeModule(pl.LightningModule):
    family: ClassVar[str]
    supports: ClassVar[Capabilities]   # e.g. {cfg, inpainting, likelihood, latent, n_stages}
    def __init__(self, config=None, **kwargs)        # resolve_config pattern (checkpoint-safe)
    def build_condition(self, batch) -> ConditionOutput
    def compute_loss(self, batch: ECGBatch) -> dict[str, Tensor]   # "loss" + extra logs
    def training_step / validation_step              # default: compute_loss + log; GAN overrides
    @torch.no_grad()
    def sample(self, n: int, cond: dict | None = None, obs: Tensor | None = None,
               obs_mask: Tensor | None = None, guidance: float = 1.0,
               steps: int | None = None, generator=None) -> Tensor   # [n, C, T] in physical units
    def configure_optimizers                         # from OptimConfig (AdamW, sched, warmup)
```

The shared base handles EMA weights (on/off via config), logging names (`train/…`, `val/…`, `val_loss`),
saving normalizer statistics and `ConditionSpec` into the checkpoint, and moving outputs back to
physical units. Each family overrides only what differs.

## 5. Family-specific training (what each family module owns)

| Family | Training objective | Special training mechanics | Sampling | Conditioning hook |
|---|---|---|---|---|
| **VAE / CVAE** | recon + β·KL (+ optional spectral / perceptual) | KL annealing, free bits | z ~ p(z\|c) → decode | encoder q(z\|x,c), decoder p(x\|z,c), optional learned prior p(z\|c) |
| **GAN** (Pulse2Pulse/WaveGAN, WGAN-GP) | adversarial (+ GP, + L1 for paired tasks) | manual optimization, `n_critic`, two optimizers | one forward pass | generator: FiLM/AdaIN or concat; discriminator: **projection** or **AC-GAN** head; paired tasks (leads→12 leads) use a pix2pix-style conditional D |
| **VQ-VAE + AR prior** | stage 1: recon + codebook + commitment; stage 2: CE over codes | two stages, frozen stage 1; teacher forcing | AR sampling, temperature, top-k | condition only the **prior** (class/vector → bias or FiLM in gated blocks; tokens → cross-attn); CFG works on logits |
| **Diffusion** (DDPM / v-pred, score SDE) | MSE on ε / v / x₀ at random t | timestep sampling, noise schedule, EMA, min-SNR weighting | ancestral, DDIM, DPM-Solver | t-emb + vector (AdaGN/AdaLN), cross-attn for tokens, concat for dense; CFG via cond dropout; masked training (SSSD-style) or RePaint for partial ECG |
| **Flow matching / rectified flow** | MSE on velocity v(x_t,t) = x₁ − x₀ | t sampling (uniform / logit-normal), optional OT coupling, EMA, reflow | ODE solver (Euler / Heun / RK45) | same backbone hooks as diffusion; CFG on velocity; partial-ECG by masked conditional flow or source = observed |
| **Latent (diffusion/flow)** | diffusion/flow loss in latent of a frozen VAE / VQ-VAE | two stages (reuse `PriorLightning` frozen-checkpoint pattern), latent scaling factor | sample latent → decode | as diffusion; dense conditions must be downsampled to latent length (`downsample_factor`) |
| **SSM backbone** (S4 / Mamba) | n/a – a backbone, used inside diffusion/flow/AR | long-sequence efficiency for T=5000 | – | FiLM / AdaLN per block |
| *Future*: consistency/distillation, normalizing flows, AR transformers on tokens | own module | teacher model, likelihood | own | reuse conditioners unchanged |

Diffusion and flow matching share almost everything (backbone, conditioner, EMA, CFG, inpainting).
Implement them as one `ContinuousGenerativeModule` with a pluggable **`Process`** (`DDPMProcess`,
`VPSDEProcess`, `RectifiedFlowProcess`) that provides `sample_t`, `corrupt(x0, noise, t)`, `target`
and a `solver` step. Adding a new continuous-time method then means adding a `Process`, not a family.

## 6. Conditioning types supported and how they change training

*(Detailed literature survey with references: see §11.)*

| Condition type | Example source | Encoder → slot | Injection | Training change | Families |
|---|---|---|---|---|---|
| Class / rhythm label | PTB-XL superclass, AF vs SR | embedding → `vector` | add to t-emb / AdaGN / FiLM; GAN: projection D | label dropout (CFG); class-balanced sampling | all |
| Multi-label diagnostic statements | PTB-XL SCP codes (71), MIMIC report-derived codes | multi-hot → MLP → `vector` (or per-label tokens) | AdaGN / AdaLN, or cross-attn over label tokens | per-label dropout; ACGAN BCE head for GANs | all |
| Tabular / demographics | age, sex, height, weight, BMI | numeric MLP + categorical emb, missing-aware → `vector` or per-field `tokens` | FiLM / AdaLN; cross-attn if many fields | missing mask + field dropout; normalizer fit on train | all |
| ECG measurement parameters | MIMIC machine measurements (RR, PR, QRS, QT-ish, axes), HR | Fourier features + MLP → `vector` | FiLM / AdaLN | as tabular; plus **consistency loss/eval**: re-measure generated ECG (NeuroKit2) vs requested | all |
| Free text / report | MIMIC machine reports, PTB-XL report | frozen text encoder (e.g. BioClinicalBERT / CLIP-style ECG-text) → `tokens` + pooled `vector` | cross-attention (+ pooled to AdaLN) | text dropout (CFG); optional precomputed embeddings cache | diffusion, flow, latent, AR prior; GAN/VAE via pooled vector only |
| Lead conditioning (generate missing leads from 1–3 leads) | single-lead wearable → 12-lead | masked signal + mask → `dense` | concat at input (and at each U-Net scale) | `MaskGenerator(lead_subset)`; loss on unobserved leads or all | GAN (pix2pix), diffusion, flow, VAE (CVAE), AR prior (partial) |
| Partial ECG in time (imputation / forecasting / extension) | missing segments, extend a 2.5 s strip to 10 s | masked signal + mask → `dense` | concat; or training-free RePaint / replacement / DPS at sampling | `MaskGenerator(time_span / forecast / random)`; SSSD-style loss on masked region | diffusion, flow (train-time or sample-time); AR (natural for forecasting); GAN/VAE (train-time only) |
| Other modality → ECG | PPG, BCG, PCG | modality encoder → `dense` (time-aligned) or `tokens` | concat / cross-attn | paired dataset requirement | diffusion, flow, GAN |
| Beat / template / rhythm-strip | median beat, lead II rhythm strip | signal encoder → `dense` or `vector` | concat / FiLM | data pipeline extracts template | all |
| Subject / style identity | patient's previous ECG | ECG encoder (contrastive) → `vector` | AdaLN | pairs from same subject at different times | diffusion, flow, VAE |
| Denoising / super-resolution | noisy or low-fs ECG | degraded signal → `dense` | concat | degradation transform in data pipeline | diffusion, flow, GAN |
| Counterfactual / editing | "this ECG but with LBBB" | source ECG + target label | SDEdit / DDIM inversion + CFG | no training change (sampling-time) | diffusion, flow |

**Classifier and DPS guidance** reuse a separately trained ECG classifier or measurement network. They
need no change to generator training and live in `sampling/guidance.py`.

**Compatibility check** (`Capabilities` + backbone `accepts`): the builder refuses, for example,
`gan + cfg`, `vae + repaint`, or `tokens` conditioning on a backbone without cross-attention, and
gives a clear error message.

## 7. Configuration and registry

A single YAML schema is shared by all models. Every component is chosen by registry **name** (short)
or `target` (import path, for user code outside the library, which gives extensibility without
forking):

```yaml
experiment: {name: ddpm_mimic_measurements, seed: 42, runs_root: runs}
data:   {name: mimic_iv_ecg, params: {...}, fields: [...], masks: null}
model:
  family: diffusion                   # registry key → ContinuousGenerativeModule
  process: {name: ddpm, params: {schedule: cosine, prediction: v, steps: 1000}}
  backbone: {name: unet1d, params: {channels: [64,128,256,256], attn_levels: [3]}}
  conditioning:
    encoders:
      - {field: measurements, name: numeric_mlp, params: {dim: 256}}
    injection: adagn                  # or film | adaln_zero | cross_attn | concat
    dropout: {measurements: 0.1}      # enables CFG
  ema: {decay: 0.9999}
  optim: {name: adamw, lr: 2.0e-4, weight_decay: 0.0, warmup_steps: 1000}
sampling: {sampler: ddim, steps: 50, guidance: 2.0}
callbacks: [{name: ema}, {name: ecg_samples, params: {every_n_epochs: 5, cond_from: val}}]
trainer: {max_epochs: 200, accelerator: gpu, devices: [0], precision: bf16-mixed}
wandb: {enabled: false}
```

- `ecgen.registry` has registries for `datasets, fields, masks, conditioners, injections,
  backbones, processes, families, samplers, callbacks, metrics`. Built-ins are registered at import
  time, and third-party packages can register through the entry point group `ecgen.plugins`.
- The runner **honours `callbacks:`** (today it is ignored), and **CLI flags never silently lose to
  YAML**. Use one override syntax (`ecgen train -c x.yaml model.optim.lr=1e-4`).
- The **resolved config** (after defaults and overrides) is written to `runs/<name>/seed_<n>/config_resolved.yaml`
  as today, and also embedded in the checkpoint.
- Optionally use Hydra or OmegaConf later. Do not start with it: a small dotted-override parser plus
  dataclass validation is enough, and it keeps dependencies light.

## 8. Evaluation of conditional generation

This extends `evaluation/` and must ship with the families. Without it, conditioning cannot be
judged.

- **Unconditional fidelity and diversity** (existing): distribution distances on features and
  embeddings (FID-style on a pretrained ECG encoder), MMD, precision/recall, and diversity.
- **Condition adherence:**
  - labels: accuracy / AUROC of a **pretrained ECG classifier** on generated samples for the
    requested labels;
  - parameters: |requested − measured| for HR / RR, PR, QRS, QT and axis, measured with a
    delineator (NeuroKit2) on the generated ECGs;
  - text: retrieval score with an ECG–text contrastive model.
- **Partial-ECG / lead tasks:** MSE / MAE and Pearson per lead on the masked region, plus
  consistency on the observed region (should be ≈ exact).
- **Utility:** TSTR / TRTR (train on synthetic, test on real), and augmentation gains on a downstream
  classifier.
- **Privacy:** nearest-neighbour distance to training set (memorization check).
- **CFG sweep:** adherence vs fidelity curve over guidance scales.

## 9. Execution plan (phased)

Each phase ends with green tests (`pytest tests/`), `black . && flake8 .`, a smoke run with
`data.params.max_samples: 1000`, and updated docs (`mkdocs build --strict`). Each phase is
independently mergeable.

### Phase 0 — Foundations (no new models)
1. Create `ecgen/types.py` (`ECGBatch`, `ConditionOutput`), `ecgen/registry.py`.
2. Port `MIMICIVECGDataset` → `ecgen.data.datasets.mimic` returning **dicts**:
   - fit `FeatureNormalizer` on **train only** and share it with val/test;
   - allow missing values (value + mask) in place of `dropna`; filtering stays optional;
   - fixed physical scaling option (`scale: mv`), with per-record z-score as a legacy option;
   - expose lead names and fs.
3. Generic `ECGDataModule` + `collate` → `ECGBatch`. Delete the four inline DataModules.
4. `ecgen.training.runner`: merge `training/trainers/gan_trainer.py` and `src/ecgen/training/train.py`
   (they are identical apart from imports); add registry names, honoured `callbacks:`, dotted overrides,
   and `ecgen train` CLI.
5. Move VAE, VQ-VAE+prior and Pulse2Pulse under `ecgen.families.*` behind `GenerativeModule`,
   unchanged numerically. Leave re-export shims in `models/`. Rewrite `configs/experiments/*.yaml`
   to the new schema, and keep the shell wrappers working by calling `ecgen train`.
6. Tests: batch/collate, normalizer is fit on train only, registry build, **old checkpoints still load**
   (`test_checkpoint_loading.py`), one-batch `fast_dev_run` per family on synthetic data
   (add a `SyntheticECGDataset` so tests need no MIMIC).
   *Acceptance:* the three existing families train via `ecgen train -c configs/experiments/<x>.yaml`
   with identical loss curves on a fixed seed.

### Phase 1 — Conditioning core
1. `conditioning/encoders`: `categorical`, `multilabel`, `numeric_mlp` (missing-aware),
   `fourier_scalar`, `masked_signal`. Add `CompositeConditioner` and `ConditionDropout` (learned null
   embeddings).
2. `nn/injection.py`: `ConcatInput`, `FiLM`, `AdaGN`, `AdaLNZero`, `CrossAttention1d`, `AddToTimeEmb`.
3. `data/masks.py`: `lead_subset`, `time_span`, `forecast`, `random_points`, `none`, plus a mixture.
4. `Capabilities` / `accepts` compatibility validation in the builder.
5. Tests: shape contracts for every encoder × injection, dropout produces null rows, mask statistics,
   and **round-trip `sample(cond=raw)`** with saved normalizer statistics.

### Phase 2 — Conditional versions of existing families
1. **CVAE**: condition the encoder and decoder via FiLM, with an optional learned prior `p(z|c)`.
2. **Conditional Pulse2Pulse**: (a) FiLM-conditioned generator + **projection discriminator** for
   vector conditions; (b) **lead-to-12-lead** paired mode (U-Net generator takes the masked signal, D
   sees `(signal, obs)`, + L1).
3. **Conditional PixelCNN prior**: global condition as bias or FiLM in `CausalGatedBlock`. CFG on
   logits. The VQ-VAE stays unconditional.
4. *Acceptance:* on MIMIC measurements, the measured RR / HR of generated ECGs tracks the requested
   value better than the unconditional baseline (Phase 7 metric, simple version first).

### Phase 3 — Diffusion family
1. `ContinuousGenerativeModule` + `DDPMProcess` (ε / v / x₀ prediction, linear / cosine schedules),
   EMA, min-SNR weighting.
2. `UNet1d` backbone with t-emb, AdaGN, optional attention at low resolution, and input concat for
   `dense`.
3. Samplers: ancestral, DDIM. CFG. Inpainting: masked training (SSSD-style, loss on unobserved only,
   configurable) **and** training-free replacement / RePaint for unconditional models.
4. *Target reproduction:* SSSD-ECG-style conditional generation on PTB-XL labels (needs the PTB-XL
   dataset; see Phase 6) and MIMIC measurement conditioning. Lead-reconstruction task: 1 lead →
   12 leads.

### Phase 4 — Flow matching family
1. `RectifiedFlowProcess` / conditional flow matching (linear interpolation, logit-normal t),
   optional minibatch-OT coupling.
2. ODE samplers: Euler, Heun (optional `torchdiffeq` extra). CFG on velocity.
3. Partial-ECG: masked conditional flow (dense concat) and guided / replacement sampling.
4. Reuses `UNet1d`, conditioners and callbacks unchanged. This proves the extensibility claim.

### Phase 5 — Latent generative models and stronger backbones
1. `latent` family: diffusion / flow in the latent of a frozen VAE / VQ-VAE (reuse the frozen-checkpoint
   pattern from `PriorLightning`, store the stage-1 config in the checkpoint, latent scale factor).
2. `DiT1d` (patchified transformer, AdaLN-Zero, cross-attn) and an **SSM** backbone (S4D / Mamba,
   optional extra) so `models/ssm/` stops being a placeholder.

### Phase 6 — More data and text conditioning
1. `ptbxl` dataset (SCP statements → superclass / subclass / multilabel, age, sex, height, weight, folds
   1–10 split as recommended by PTB-XL), optional PTB-XL+ features.
2. MIMIC machine reports as a text field. `text` encoder (frozen HF model; cache embeddings to disk)
   behind `ecgen[text]`.
3. Cross-attention conditioning on text for diffusion / flow / latent.

### Phase 7 — Evaluation and sampling API
1. `ecgen.evaluation`: the metrics in §8, with pretrained classifier and delineator wrappers.
2. Public API: `ecgen.sample(ckpt, n, cond=..., obs=..., obs_mask=..., guidance=..., steps=...)`,
   `ecgen sample` CLI; CSV / WFDB / plot outputs using the existing `ecgen.generate` writers.
3. Keep `ecgen.generate(...)` for Pulse2Pulse working (backward compatibility test).

### Phase 8 — Docs, packaging, release
1. Docs: "Concepts" (axes and contracts), "Conditioning guide", one page per family, and
   "Extending ECGEN" (below). Visual Guide pages via the `visual-docs` skill for diffusion and flow.
2. Extras: `ecgen[train]` (lightning, wfdb, sklearn, if we want a slimmer core), `ecgen[text]`,
   `ecgen[ssm]`, `ecgen[eval]` (neurokit2).
3. Remove the shims in `models/`, `data/` and `training/` after one release. Bump the minor version
   (tag `v0.x.0`).

## 10. Extending later: checklists

- **New family** (e.g. consistency model): subclass `GenerativeModule`, set `supports`, implement
  `compute_loss` and `sample`, add a config dataclass, `@families.register("consistency")`, a smoke
  test, and a docs page. Conditioners, backbones, data and runner are reused as-is.
- **New continuous-time method** (e.g. stochastic interpolants, EDM): add a `Process` with no new
  family.
- **New backbone**: implement `forward(x, t, cond: ConditionOutput)`, declare `accepts`, register it.
- **New conditioning type**: add a `FieldSpec` kind (dataset side) + an encoder producing
  `vector` / `tokens` / `dense`. No family changes.
- **New dataset**: implement `BaseECGDataset` returning the dict contract and its `ConditionSpec`.
- **New guidance / inpainting method**: add a function in `sampling/guidance.py` or `inpainting.py`
  and register it.
- **Out-of-tree**: any of the above can live in user code via `target: my_pkg.MyThing` or the
  `ecgen.plugins` entry point.

## 11. Conditioning research summary

This summary comes from a literature survey done on 2026-10-11. Tags: **[V]** means the
existence, venue and ID were checked by web search. **[K]** means a well-known reference taken
from background knowledge and not re-checked. **[U]** means uncertain. Before citing any of these
in a paper, check them again.

### 11.1 What the ECG literature does

Conditioning in recent ECG generation work comes down to three patterns, which match the three
`ConditionOutput` slots:

1. **Labels, demographics and measurements → a global vector.** The vector is added to the time
   embedding or fed through FiLM, AdaGN or AdaLN. SSSD-ECG, for example, projects the 71-dim
   PTB-XL multi-hot vector and adds it in every residual block.
2. **Text → a pretrained (usually frozen) embedding.** It enters as a pooled vector or as tokens
   for cross-attention. In VQ + AR models it enters as prefix tokens (Auto-TTE).
3. **Partial signals → input channels plus a mask, or a sampling-time inverse problem.** Partial
   signals include other leads, time segments, PPG and beat templates. The training-time route is
   CSDI/SSSD-style masked training. The training-free routes are RePaint and DPS/MGPS.

Design consequences adopted in this plan:

- **Generate 8 independent leads** (I, II, V1–V6) and derive III, aVR, aVL and aVF analytically
  (Einthoven/Goldberger). SSSD-ECG does this, and so does the existing `ecgen.generate` CSV. Add
  this as `transforms.leads: independent8` plus a derivation step in `sample()`. Report
  Einthoven-consistency for any model that outputs all 12 leads directly.
- **Masks are data, not models.** Imputation, forecasting and lead reconstruction are mask samplers
  in the DataModule. DPS and RePaint are sampler plug-ins that work on unconditional diffusion and
  flow checkpoints.
- **VQ-VAE: condition only the prior.** Use gated-PixelCNN bias terms for vectors, `h` upsampled to
  latent length for dense conditions, and prefix or cross-attention for tokens. Stage 1 stays
  unchanged (as in Auto-TTE).
- **Per-group null tokens** in `ConditionDropout` let each group get its own guidance weight
  (decoupled CFG), e.g. labels at w = 3 and demographics at w = 1. This keeps unrelated attributes
  fixed during counterfactual edits.
- **ControlNet-style adapters** (zero-init copy of the encoder, base model frozen) let you add
  dense control to an already trained model. Add this in Phase 5 as `conditioning.adapter: controlnet`.
- **Flow matching is cheap to add once diffusion exists.** FlowECG replaces SSSD-ECG's diffusion
  with FM under the same labels and gets comparable quality at 10–25 function evaluations instead
  of 200. This supports the shared `Process` design in §5.

### 11.2 Mechanisms (generic) → where they live in the library

| Mechanism | Module | Families |
|---|---|---|
| Concat to input (+ mask channel) | `nn/injection.ConcatInput` | all conv/SSM backbones |
| Concat to latent / CVAE prior p(z\|c) | `families/vae` | VAE |
| Conditional norm: CBN / AdaGN / FiLM / AdaIN | `nn/injection.{FiLM,AdaGN}` | all |
| AdaLN-Zero | `nn/injection.AdaLNZero` | DiT, SSM, FM/diffusion |
| Embedding added to time embedding | `nn/injection.AddToTimeEmb` | diffusion, FM |
| Cross-attention | `nn/injection.CrossAttention1d` | diffusion, FM, LDM, AR transformer |
| Prefix tokens | AR prior backbone | VQ + AR |
| Gated PixelCNN conditional bias | `families/vqvae` prior blocks | VQ + PixelCNN |
| Projection discriminator / AC-GAN / pix2pix D + L1 | `families/gan` heads | GAN |
| Classifier / regressor guidance | `sampling/guidance` | diffusion, FM |
| Classifier-free guidance (per-group) | `conditioning/dropout` + `sampling/guidance` | diffusion, FM, AR prior |
| ControlNet adapter | `conditioning/adapters` | diffusion, FM, LDM |
| Masked training (CSDI/SSSD) | `data/masks` + family loss mask | diffusion, FM, VAE, GAN |
| RePaint / replacement | `sampling/inpainting` | diffusion (FM approx.) |
| DPS / MGPS posterior sampling | `sampling/guidance` | diffusion, FM |
| SDEdit / DDIM inversion | `sampling/edit` | diffusion, FM |
| Conditional coupling layers | future `families/nf` | normalizing flows |

### 11.3 Condition fields available in the target datasets

- **MIMIC-IV-ECG** ([PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/); ~800k ECGs, ~160k patients, 10 s, 500 Hz).
  - **`machine_measurements.csv`:** `rr_interval`, `p_onset`, `p_end`, `qrs_onset`, `qrs_end`,
    `t_end` and the `p/qrs/t_axis` fields (all currently used), plus `report_0…report_17` (machine
    report text), `cart_id`, `bandwidth`, `filtering` and `ecg_time`.
    - Derived fields: HR = 60000 / RR; PR = qrs_onset − p_onset; QRS = qrs_end − qrs_onset;
      QT = t_end − qrs_onset.
    - Check the column list against `machine_measurements_data_dictionary.csv`.
  - **`waveform_note_links.csv`:** links to cardiologist reports in MIMIC-IV-Note.
  - **`subject_id`:** links to MIMIC-IV demographics, labs (K⁺, troponin) and ICD codes.
  - **Caveat:** ECG timestamps come from the machine clock. Lab or diagnosis conditioning
    therefore needs tolerant time-window matching. Only about 55% of ECGs overlap a hospital
    admission.
- **PTB-XL** (Wagner et al., *Sci Data* 2020 [K]; 21,799 records, 100 / 500 Hz).
  - `scp_codes`: 71 statements with likelihoods, mapped to super/subclass, form and rhythm
    through `scp_statements.csv`.
  - Demographics: `age`, `sex`, `height`, `weight`. Height and weight are often missing, so a
    missing-aware encoder is needed.
  - `heart_axis`, `pacemaker`, `extra_beats`.
  - Noise flags (`baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`).
  - `report` (mostly German) and `strat_fold` (the recommended split).
- **PTB-XL+** (Strodthoff et al., *Sci Data* 2023 [V]; [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/)):
  harmonised 12SL / Uni-G / ECGdeli features and median beats. This is the best source for
  measurement and beat-template conditioning.

### 11.4 ECG generation papers with conditioning (reproduction targets marked ★)

| Paper | Conditioning | Family | Ref |
|---|---|---|---|
| Pulse2Pulse / DeepFake ECG (Thambawita et al., *Sci Rep* 2021) | none (baseline in repo) | GAN | [V] doi:10.1038/s41598-021-01295-2 |
| ME-GAN (Chen et al., ICML 2022) | disease, mixup-normalization, view D; rFID | GAN | [V] arXiv:2207.10670 |
| ★ SSSD-ECG (Alcaraz & Strodthoff, 2023) | 71 PTB-XL statements, 8 leads | diffusion + S4 | [V] arXiv:2301.08227 |
| SSSD (imputation/forecasting) | masked training | diffusion + S4 | [K] arXiv:2208.09399 |
| Auto-TTE (Chung et al., ICASSP 2023) | text + patient embedding, prefix | VQ + AR | [V] arXiv:2303.09395 |
| DiffECG (Neifar et al., 2023/24) | generation / imputation / forecasting in one model | diffusion | [V] arXiv:2306.01875 |
| PulseDiff (2023) | template (beat) prior for imputation | diffusion | [V] arXiv:2310.15742 |
| BioDiffusion (Li et al., *Bioengineering* 2024) | label and signal conditioning | diffusion | [V] arXiv:2401.10282 |
| ECGT2T (2021) | 2 asynchronous leads → 10 leads | GAN | [V] arXiv:2103.00006 |
| Single → 12-lead GAN (Joo et al., MICCAI 2023; "EKGAN"? [U]) | lead conditioning | GAN | [V] doi:10.1007/978-3-031-43990-2_18 |
| Shifted diffusion, 12-lead from single lead (Liu et al., ECML-PKDD 2024) | lead conditioning | diffusion | [V] |
| BeatDiff / EM-BeatDiff (NeurIPS 2024); Bayesian ECG reconstruction | posterior sampling for missing leads | diffusion | [V] arXiv:2401.05388 |
| MGPS (ICLR 2025) | midpoint-guided posterior sampling | sampler | [V] |
| RDDM (Shome et al., AAAI 2024) | PPG → ECG | diffusion | [V] arXiv:2308.13568 |
| CardioGAN (Sarkar & Etemad, AAAI 2021) | PPG → ECG | GAN | [K] |
| PPGFlowECG (2025); CardioFlow (Nambu et al., 2025 [U venue]) | PPG → ECG | latent rectified flow | [V] arXiv:2509.19774 |
| ★ FlowECG (2025) | SSSD-ECG labels | flow matching | [V] arXiv:2509.10491 |
| ★ DiffuSETS (Lai et al., *Patterns* 2025) | LLM-embedded MIMIC reports + age, sex, HR | latent diffusion | [V] arXiv:2501.05932 |
| MIDT-ECG (NeurIPS 2025) | demographics + mel/STFT loss | diffusion | [V] arXiv:2510.05492 |
| SE-Diff (ICLR 2026) | text + ODE-simulator prior + retrieved EHR | latent diffusion | [V] arXiv:2511.09895 |
| ECGTwin (2025) | reference ECG (identity) + target condition | diffusion | [V] arXiv:2508.02720 |
| DADM (2025) | pre-dose ECG + drug (ControlNet for tabular) | diffusion | [V] arXiv:2502.07297 |
| DCDM-ECG (ICML 2026) | 71 SCP + age, sex, height, weight, HR | latent diffusion | [V/U] listing only |
| GCX (*EHJ Digital Health*); CoFE (2025) | counterfactual explanations | StyleGAN / other | [V] arXiv:2508.16033 (CoFE) |
| MERL (Liu et al., ICML 2024) | ECG–text contrastive; frozen text encoder / CLIP-score | encoder | [V] arXiv:2403.06659 |
| MedFlow (2026) | class-aware flow matching, medical time series | flow matching | [V] arXiv:2609.04804 |

Requested items that could not be verified, so they are left out: "SignalDiff", "Kardio LDM",
a generative "zero-shot ECG diffusion" paper, and CardioGPT.

### 11.5 General method references [K]

- **Conditioning a network:** CVAE (Sohn et al., NeurIPS 2015); Conditional BN (arXiv:1707.00683);
  FiLM (arXiv:1709.07871); DiT / AdaLN-Zero (arXiv:2212.09748); ControlNet (arXiv:2302.05543);
  conditional PixelCNN (arXiv:1606.05328).
- **Conditional GANs:** AC-GAN (arXiv:1610.09585); projection discriminator (arXiv:1802.05637);
  pix2pix (arXiv:1611.07004).
- **Guidance:** classifier guidance (arXiv:2105.05233); classifier-free guidance (arXiv:2207.12598);
  decoupled CFG (arXiv:2506.14399 [V]).
- **Inpainting, inverse problems and editing:** RePaint (arXiv:2201.09865); DPS (arXiv:2209.14687);
  SDEdit (arXiv:2108.01073); CSDI (arXiv:2107.03502).
- **Other model families:** latent diffusion (arXiv:2112.10752); flow matching (arXiv:2210.02747);
  rectified flow (arXiv:2209.03003); SRFlow, a conditional normalizing flow (arXiv:2006.14200).
- **SSM backbones:** S4 (arXiv:2111.00396); Mamba (arXiv:2312.00752).

### 11.6 Evaluation additions from the survey

These extend §8:
- **rFID** on a pretrained ECG encoder (ME-GAN).
- **Inter-lead correlation error** (MIDT-ECG) and **Einthoven consistency**.
- **CLIP-score / retrieval R@k** with MERL for text conditioning.
- **Diagnostic agreement** on reconstructed versus real ECGs for lead tasks.
- **Error at fiducial points.**
- **Membership inference** in addition to nearest-neighbour memorization checks.
- **Clinical Turing test** as the qualitative gold standard.
- The classifier used to score adherence must be trained on data separate from the generator's
  training split.

## 12. Risks and open questions

- **Scope.** Phases 0–3 are already a large refactor. Phase 0 is the critical one: if contracts are
  wrong there, every later phase pays for it. Review the contracts in §4 before coding.
- **Memory / compute.** 12 × 5000 with attention is expensive. Use attention only at downsampled
  resolutions, patchify for DiT, or use latent models.
- **Checkpoint compatibility.** Nested configs (conditioning, backbone) must still survive
  `save_hyperparameters` → `load_from_checkpoint`. Store the resolved nested config as one
  hyperparameter, and keep the `resolve_config` behaviour tested.
- **Data semantics.** Lead order and units must be part of the checkpoint. Otherwise a sampled ECG
  cannot be interpreted.
- **Open decisions for the maintainer:**
  1. Move the research code into `src/ecgen/` (recommended) vs a separate distribution?
  2. Keep PyTorch Lightning as a hard dependency of `ecgen`, or put training behind an `ecgen[train]`
     extra?
  3. Hand-rolled config overrides vs Hydra / OmegaConf?
  4. Which datasets come first after MIMIC (PTB-XL recommended, because it has labels and demographics)?
