# Diffusion and flow matching

!!! tip "Visual guide"
    Animated diagrams of the noising process, the `UNet1d` backbone, guidance, the loss and
    sampling: [Diffusion (DDPM)](../visual/diffusion.md) and [Flow Matching](../visual/flow_matching.md).

`DiffusionModule` (registry name `diffusion`) is a Gaussian diffusion model
(DDPM, Ho et al., 2020) over raw 12-lead ECGs. It is one instance of
`ContinuousGenerativeModule`, which separates three choices:

| Choice | Config key | Built-in |
|---|---|---|
| **Process** — how training pairs are corrupted and how samples are integrated | `model.params.process` | `ddpm` |
| **Backbone** — the network `f(x_t, t, cond)` | `model.params.backbone` | `unet1d` |
| **Conditioning** — what the model is conditioned on | `model.params.conditioning` | see [Conditioning](../training/conditioning.md) |

`FlowMatchingModule` (registry name `flow_matching`) is the same module with the
`rectified_flow` process as its default — see [Flow matching](#flow-matching).

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

## Flow matching

Process `rectified_flow` (family `flow_matching`): time runs from noise
(`t = 0`) to data (`t = 1`) along straight paths
`x_t = t·x1 + (1 − (1 − σ_min)·t)·x0`; the network regresses the velocity
`x1 − (1 − σ_min)·x0`, and sampling integrates the ODE with Euler or Heun steps.
Typically 10–25 steps (Heun: two network calls per step) give samples
comparable to 50+ DDIM steps.

| Param | Default | Meaning |
|---|---|---|
| `t_sampling` | `"logit_normal"` | Training time distribution (`"uniform"` or logit-normal, Esser et al. 2024) |
| `logit_mean`, `logit_std` | `0.0`, `1.0` | Logit-normal parameters |
| `sigma_min` | `0.0` | Path width at `t = 1` (0 = rectified flow) |
| `ot_coupling` | `false` | Minibatch optimal-transport pairing of noise and data (OT-CFM) |
| `time_scale` | `1000.0` | Network time input = `t · time_scale` |
| `solver` | `"euler"` | Default ODE solver (`"euler"` or `"heun"`); override per call with `sample(solver=...)` |

Everything else — backbone, conditioning, CFG (on the velocity), EMA,
`loss_on`, `clean_observed` — is shared with diffusion. Training-free
inpainting supports `replace` (the observed part follows its straight path from
the initial noise); RePaint is diffusion-only.

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

## Other backbones

| Name | Idea | Condition slots | Notes |
|---|---|---|---|
| `unet1d` | Convolutional U-Net (above) | vector, tokens, dense | Default; length divisible by `2^(levels-1)` |
| `dit1d` | Transformer over patches of `patch_size` samples, AdaLN-Zero modulation, cross-attention in every block (Peebles & Xie, 2023) | vector, tokens, dense | Length divisible by `patch_size`; good for latents (e.g. 625 / 5 = 125 tokens) |
| `ssm` | SSSD-style stack of bidirectional S4D state-space layers at full resolution (Alcaraz & Strodthoff, 2023; Gu et al., 2022), pure PyTorch | vector, dense | Global receptive field via FFT convolution |
| `mamba` | `ssm` with bidirectional Mamba layers (Gu & Dao, 2023) | vector, dense | Needs `mamba-ssm` (CUDA) |

`dit1d`: `patch_size` (25), `dim` (256), `depth` (8), `heads` (8), `mlp_ratio` (4).
`ssm` / `mamba`: `channels` (128), `depth` (8), `state_size` (64 / 16), `bidirectional` (true), `dropout`.

## Latent diffusion and flow matching

`latent_diffusion` and `latent_flow_matching` run the same processes in the
latent space of a frozen stage-1 autoencoder (Rombach et al., 2022):

- `autoencoder_family: vae` — a spatial, unconditional `vae`; the posterior mean is the latent
  (`[latent_channels, T / 8]`);
- `autoencoder_family: vqvae` — the quantized embeddings are the latent; sampled latents are
  snapped to the codebook before decoding.

The stage-1 config is stored in the checkpoint (`autoencoder_config`) and its
weights in the state dict, so the stage-1 file is not needed after training
(the same pattern as the PixelCNN prior). Latents are multiplied by
`latent_scale` (default: `1 / std` of the first training batch, then fixed and
saved). Dense conditions are pooled to the latent length; `loss_on`,
`clean_observed` and training-free inpainting are signal-space options and are
rejected. Example: `configs/experiments/latent_diffusion_mimic.yaml`.

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
- `configs/experiments/flow_matching_mimic.yaml` — conditional flow matching
- `configs/experiments/diffusion_mimic_ssm.yaml` — SSSD-style S4D backbone
- `configs/experiments/latent_diffusion_mimic.yaml` — latent diffusion with a DiT backbone

Evaluate lead reconstruction with
`python scripts/evaluate/lead_reconstruction.py <ckpt> --family diffusion --keep I`
(add `--inpaint repaint` for an unconditional model).

## Adding a process or backbone

A process subclasses `ecgen.processes.Process` (`training_pair`, `sample`) and
is registered with `@PROCESSES.register("name")`; a backbone implements
`forward(x, t, cond)`, declares `accepts` (condition slots) and takes
`in_channels`, `cond_dim`, `context_dim`, `dense_channels`; register it with
`@BACKBONES.register("name")`. Neither needs changes to the family.
