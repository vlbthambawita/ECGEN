# Concepts

ECGEN trains every generative model through one runner (`ecgen train`) by
splitting a model into independent, swappable parts. Each part talks to the
others only through a small contract, so a new method, network, condition, or
dataset is one new class — not a new training script.

| Part | Owns | Contract | Registry | Built-ins |
|---|---|---|---|---|
| **Dataset** | Reading records, splits, condition fields, normalizers | Dict samples `{signal [C,T], cond, cond_missing, meta}` + `lead_names`, `fs`, `seq_length`, `field_normalizers` | `DATASETS` | `mimic_iv_ecg`, `ptbxl`, `synthetic` |
| **Masks** | Which part of an ECG is observed (partial-ECG tasks) | `obs_mask [C,T]`, 1 = observed | `MASKS` | `lead_subset`, `time_span`, `forecast`, `random_points`, `none`, `unobserved`, `mixture` |
| **Batch** | Collation | `ECGBatch(signal, lead_names, fs, cond, cond_missing, obs_mask, meta)` | — | `ECGDataModule` |
| **Condition encoders** | One field → one slot | `vector [B,D]`, `tokens [B,N,D]` + mask, or `dense [B,C',T]` | `CONDITION_ENCODERS` | `categorical`, `multilabel`, `numeric`, `fourier`, `masked_signal`, `text_hash`, `text` |
| **Conditioner** | Combining fields, dropout, null embeddings | `ConditionOutput(vector, tokens, tokens_mask, dense)` | — | `CompositeConditioner` |
| **Family** | The training objective and sampler | `GenerativeModule`: `__init__(config=None, **kw)`, `training_step`, `sample(n, cond=, obs=, …)`, `supports`, `accepts` | `FAMILIES` | `vae`, `vqvae`, `vqvae_prior`, `pulse2pulse`, `diffusion`, `flow_matching`, `latent_diffusion`, `latent_flow_matching` |
| **Process** | (continuous families) corruption + integration | `training_pair(x0, noise)`, `sample(model_fn, shape, steps, …)` | `PROCESSES` | `ddpm`, `rectified_flow` |
| **Backbone** | (continuous families) the network | `forward(x, t, cond)`, `accepts` | `BACKBONES` | `unet1d`, `dit1d`, `ssm`, `mamba` |
| **Callbacks** | Plots, early stopping, … | Lightning callbacks | `CALLBACKS` | `vae_visualization`, `ecg_visualization`, `generated_samples`, `early_stopping`, `lr_monitor`, `model_checkpoint` |
| **Runner** | Run dirs, loggers, checkpoints, compatibility checks | YAML config → `pl.Trainer.fit` | — | `ecgen train` |

Every registry also accepts `target: package.module.Class`, so code outside the
library plugs in without registration.

## What travels with a checkpoint

- the family's config (flat hyperparameters → `Family.load_from_checkpoint(path)` works);
- `ecgen_family`: the registry name, so `ecgen.sample(path)` needs nothing else;
- `ecgen_data_info`: lead names, sampling rate, length, scaling, and one normalizer per
  numeric condition field, so `sample(cond={...})` takes raw units;
- the `ECGDataModule` state, and EMA weights for continuous families;
- for two-stage models (PixelCNN prior, latent diffusion/flow) the frozen stage-1 config
  and weights, so stage 1 is not needed again.

## Checks before training

The runner refuses a config when the model's channel count or fixed length
disagrees with the dataset, a family that cannot be conditioned gets a
`conditioning` block, an encoder produces a slot the family or backbone cannot
consume, a condition field is not in the data, or an observation encoder is
configured without `data.masks`.

## Where to go next

- [Training with `ecgen train`](training/runner.md)
- [Conditioning](training/conditioning.md)
- [Sampling](training/sampling.md)
- [Diffusion & flow matching](models/diffusion.md)
- [Extending ECGEN](extending.md)
