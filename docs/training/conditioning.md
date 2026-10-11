# Conditioning

Conditional models generate ECGs *given* something: a label, patient data,
ECG measurements, or part of an ECG. Conditioning in ECGEN is separate from the
model families: a family that supports it takes a `conditioning:` block in its
`model.params`, and the same block works for every conditional family.

## The pieces

| Piece | Where | What it does |
|---|---|---|
| Condition **fields** | the dataset | Each sample carries `cond[field]` (and `cond_missing[field]`). MIMIC: `measurements` (9 machine measurements). Synthetic: `measurements` (heart rate, amplitude), `rhythm` (3 classes), `labels` (3 multi-hot). |
| Field **encoders** | `ecgen.conditioning.encoders` | Turn one field into one *slot*: a global `vector`, a sequence of `tokens`, or a time-aligned `dense` signal. |
| **CompositeConditioner** | `ecgen.conditioning.composite` | Runs all encoders, sums vectors, concatenates tokens and dense channels, and applies condition dropout. Output: `ConditionOutput(vector, tokens, tokens_mask, dense)`. |
| **Injection** layers | `ecgen.nn.injection` | How a backbone consumes the slots: `FiLM`, `AdaGN`, `AdaLNZero`, `AddToTimeEmb` (vector), `CrossAttention1d` (tokens), `concat_dense` (dense). All start as the identity (zero-initialised). |
| Observation **masks** | `ecgen.data.masks` | For partial-ECG conditioning: `data.masks` adds an `obs_mask` to every batch, and the `masked_signal` encoder conditions on `signal * obs_mask`. |

## Encoders

| `name` | Field value | Slot | Notes |
|---|---|---|---|
| `categorical` | class index `[B]` | vector | Missing / negative index → learned "unknown" embedding |
| `multilabel` | multi-hot `[B, L]` | vector (or tokens with `as_tokens: true`) | SSSD-ECG-style linear projection; tokens = one per present label |
| `numeric` | normalized values `[B, F]` + missing mask | vector (or tokens) | Missingness is an explicit input |
| `fourier` | normalized values `[B, F]` | vector | Random Fourier features, for sharp dependence on scalars like heart rate |
| `masked_signal` | reads `obs` / `obs_mask` (field name `obs`) | dense | `[x * mask, mask]`, optionally projected by a 1×1 conv |
| `text_hash` | strings `list[str]` | tokens (or vector) | Hashed bag-of-words embeddings, trained with the model; no dependencies |
| `text` | strings `list[str]` | tokens (or vector) | Frozen Hugging Face encoder (`model_name`, `max_length`); needs `ecgen[text]`; weights stay out of the checkpoint, embeddings are cached per distinct text |

## Text conditions

Report text is a string field: MIMIC (`report: true` joins the machine report
lines), PTB-XL (`report`), and the synthetic data (a templated
`"sinus tachycardia, high voltage"`). Strings are collated into lists and
encoded into tokens that the backbone attends to (cross-attention in `unet1d`
and `dit1d`). At sampling time pass one string for all samples or one per
sample: `sample(n, cond={"report": "Sinus bradycardia"})`. An empty string or a
missing field uses the null token.

## Config

```yaml
model:
  name: <a conditional family>
  params:
    conditioning:
      vector_dim: 256          # every vector encoder is projected to this and summed
      token_dim: 256
      encoders:
        - {field: measurements, name: numeric, params: {num_features: 9, dim: 256}}
        - {field: obs, name: masked_signal, params: {num_leads: 12}}
      dropout: {measurements: 0.1}   # per-field drop probability while training

data:
  masks:                        # only needed when an encoder reads `obs`
    name: mixture
    params:
      components:
        - {weight: 0.6, name: lead_subset, params: {keep: [[I], [I, II]]}}
        - {weight: 0.2, name: time_span, params: {min_frac: 0.1, max_frac: 0.5}}
        - {weight: 0.2, name: unobserved}
```

Mask generators: `none` (all observed), `unobserved`, `lead_subset` (`keep` lists of
lead names, or a random `min_leads..max_leads`), `time_span` (imputation),
`forecast` (observe a prefix), `random_points`, and `mixture`.

## Dropout, missing fields, and guidance

While training, each field is replaced by a learned **null embedding** with its
`dropout` probability. A field that is not given at all (at sampling time) uses
the same null. So one model can be sampled with any subset of its conditions,
and classifier-free guidance compares a pass with the conditions to a pass with
`drop="all"`.

## Checks before training

The runner refuses a config when:

- the family does not support conditioning (`Capabilities.conditional`);
- an encoder produces a slot the family cannot consume (`accepts`);
- an encoder's field is not in the dataset's samples;
- an encoder reads `obs` but `data.masks` is not set.

## Raw values at sampling time

Training data statistics travel with the checkpoint (`data_info`: lead names,
sampling rate, length, and a normalizer per numeric field). At sampling time,
pass numeric fields in their raw units and the model normalizes them:

```python
model = SomeFamily.load_from_checkpoint("runs/.../last.ckpt")
x = model.sample(8, cond={"measurements": [75.0, 1.2]})   # heart rate, amplitude
x = model.sample(8, cond={})                               # unconditional (null embeddings)
```

Use `NaN` (or `None`) for an unknown value inside a numeric field.

## Conditional families

| Family | `accepts` | How conditions enter | Extras |
|---|---|---|---|
| `vae` (CVAE) | vector, dense | FiLM after every encoder/decoder residual block; dense concatenated to the encoder input and (pooled) to the decoder input | `learned_prior: true` → `p(z \| c)`; `latent_type: global` → one latent vector per ECG, whose prior samples decode to coherent ECGs (`spatial` latents reconstruct well but their prior samples do not); `recon_on: unobserved` for imputation |
| `pulse2pulse` (cGAN) | vector, dense | Generator: FiLM on every decoder stage, dense concatenated to the input noise. Critic: projection head for vectors, dense concatenated to its input (paired / pix2pix critic) | `l1_weight` (e.g. 100) + `l1_on: unobserved` for lead reconstruction; the critic has its own conditioner copy (no dropout) |
| `vqvae_prior` (conditional PixelCNN) | vector, dense | Condition added to both gate pre-activations of every causal block (`V_f h`, `V_g h`); dense pooled to code resolution | `sample(..., guidance=w)` = classifier-free guidance on the logits; the stage-1 VQ-VAE stays unconditional |
| `diffusion` (and `flow_matching`) | vector, tokens, dense | Vector added to the time embedding (→ AdaGN in every block), tokens by cross-attention at the attention levels, dense concatenated to the input | `sample(guidance=w)` = CFG; `loss_on: unobserved` + `clean_observed` for SSSD-style imputation; training-free `inpaint="replace"` or `"repaint"` |

All three keep their original parameters when `conditioning` is absent, so
unconditional checkpoints load unchanged. Conditional models `paste_observed`
the given part of `obs` into their output by default.

Example configs: `configs/experiments/cvae_mimic.yaml`,
`pulse2pulse_lead_reconstruction.yaml`, `prior_mimic_conditional.yaml`.

## Checking condition adherence

`scripts/evaluate/condition_adherence.py` samples a model at a grid of
requested heart rates and measures the heart rate of the output
(`ecgen.evaluation.measure.heart_rate`, R-peak detection; `method="neurokit"`
uses NeuroKit2):

```bash
python scripts/evaluate/condition_adherence.py runs/cvae_mimic/seed_42/checkpoints/last.ckpt \
    --family vae --target rr_interval --rates 50,70,90,110,130
```

It reports MAE and Pearson r between requested and measured rates; an
unconditional checkpoint gives the baseline (r ≈ 0).
