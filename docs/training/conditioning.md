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
