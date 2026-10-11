# Diffusion

`DiffusionModule` (registry name `diffusion`) is a Gaussian diffusion model
(DDPM, Ho et al., 2020) over raw 12-lead ECGs. It is one instance of
`ContinuousGenerativeModule`, which separates three choices:

| Choice | Config key | Built-in |
|---|---|---|
| **Process** — how training pairs are corrupted and how samples are integrated | `model.params.process` | `ddpm` |
| **Backbone** — the network `f(x_t, t, cond)` | `model.params.backbone` | `unet1d` |
| **Conditioning** — what the model is conditioned on | `model.params.conditioning` | see [Conditioning](../training/conditioning.md) |

Flow matching (Phase 4) reuses the same module with another process.

## DDPM process (`ddpm`)

| Param | Default | Meaning |
|---|---|---|
| `num_steps` | `1000` | Diffusion steps `T` |
| `schedule` | `"cosine"` | `"cosine"` (Nichol & Dhariwal) or `"linear"` beta schedule |
| `prediction` | `"v"` | Network target: `"eps"`, `"v"` (Salimans & Ho) or `"x0"` |
| `min_snr_gamma` | `5.0` | Min-SNR-γ loss weighting; `null` = uniform |
| `clip_x0` | `null` | Clamp predicted `x0` while sampling (z-scored ECGs reach about ±10) |

Sampling is DDIM (Song et al., 2021) over `steps` evenly spaced steps;
`eta=0` is deterministic, `eta=1` ancestral.

## U-Net backbone (`unet1d`)

ADM-style 1D U-Net: residual blocks with AdaGN on the time (+ condition)
embedding, self-attention at `attn_levels`, cross-attention to condition tokens
at the same levels, and dense conditions concatenated to the input. The signal
length must be divisible by `2 ** (len(channel_mults) - 1)` (5000 works with
four levels).

| Param | Default |
|---|---|
| `base_channels` | `64` |
| `channel_mults` | `[1, 2, 4, 4]` |
| `num_res_blocks` | `2` |
| `attn_levels` | `[3]` (the lowest resolution) |
| `heads` | `4` |
| `dropout` | `0.0` |

## Module options (`ContinuousConfig`)

| Field | Default | Meaning |
|---|---|---|
| `lr`, `weight_decay`, `b1`, `b2` | `2e-4`, `0`, `0.9`, `0.999` | AdamW |
| `warmup_steps` | `500` | Linear LR warm-up |
| `ema_decay` | `0.999` | EMA of the weights, used for sampling (`0` = off); stored in checkpoints |
| `loss_on` | `"all"` | `"unobserved"`: loss only where `obs_mask` is 0 |
| `clean_observed` | `false` | SSSD-style: observed entries of `x_t` are the clean signal, in training and sampling |
| `sample_steps` | `50` | Default sampling steps |
| `guidance` | `1.0` | Default classifier-free guidance scale |

## Sampling

```python
from ecgen.families import DiffusionModule

model = DiffusionModule.load_from_checkpoint("runs/diffusion_mimic/seed_42/checkpoints/last.ckpt")
x = model.sample(16, steps=50, seed=0)                       # unconditional
x = model.sample(16, cond={"measurements": [...]}, guidance=2.0)  # conditional + CFG
x = model.sample(4, obs=ecg, obs_mask=mask)                  # model trained with a masked_signal condition
x = model.sample(4, obs=ecg, obs_mask=mask, inpaint="repaint", jumps=5)  # any model, training-free
```

Inpainting modes: `replace` overwrites the observed part with the observation
noised to the current level after every step; `repaint` additionally resamples
each step `jumps` times (Lugmayr et al., 2022). The observed part is pasted into
the output unless `paste_observed=False`.

## Example configs

- `configs/experiments/diffusion_mimic.yaml` — unconditional
- `configs/experiments/diffusion_mimic_conditional.yaml` — conditioned on the machine measurements, CFG
- `configs/experiments/diffusion_mimic_leads.yaml` — lead reconstruction / imputation

Evaluate lead reconstruction with
`python scripts/evaluate/lead_reconstruction.py <ckpt> --family diffusion --keep I`
(add `--inpaint repaint` for an unconditional model).

## Adding a process or backbone

A process subclasses `ecgen.processes.Process` (`training_pair`, `sample`) and
is registered with `@PROCESSES.register("name")`; a backbone implements
`forward(x, t, cond)`, declares `accepts` (condition slots) and takes
`in_channels`, `cond_dim`, `context_dim`, `dense_channels`; register it with
`@BACKBONES.register("name")`. Neither needs changes to the family.
