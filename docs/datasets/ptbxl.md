# PTB-XL

`PTBXL` (registry name `ptbxl`) loads PTB-XL (Wagner et al., 2020; PhysioNet
`ptb-xl` 1.0.x): 21,799 ten-second 12-lead ECGs at 500 Hz (`records500/`) and
100 Hz (`records100/`), with diagnostic statements, demographics, and reports.
Optionally it joins PTB-XL+ (Strodthoff et al., 2023) 12SL measurements.

```yaml
data:
  dataset:
    name: ptbxl
    params:
      root: /path/to/ptb-xl/1.0.1
      sampling_rate: 500        # or 100
      leads: standard12         # stored order: I, II, III, aVR, aVL, aVF, V1–V6
      scale: zscore
      labels: all               # scp vocabulary, or null
      ptbxl_plus_root: /path/to/ptb-xl-plus/1.0.1   # optional: adds "features"
```

## Splits

The recommended stratified folds: `strat_fold` 1–8 train, 9 val, 10 test
(override with `folds: {train: [...], val: [...], test: [...]}`). All
normalizers are fit on the training folds.

## Condition fields

| Field | Value | Encoder to use | Switch |
|---|---|---|---|
| `scp` | multi-hot statements | `multilabel` (`num_labels` = vocabulary size) | `labels`: `all`, `superclass`, `subclass`, `diagnostic`, `form`, `rhythm`, or `null` |
| `demographics` | normalized `[age, height, weight]` + missing mask | `numeric` (`num_features: 3`) | `demographics: true` |
| `sex` | class (0 male, 1 female) | `categorical` (`num_classes: 2`) | `sex: true` |
| `report` | report text (mostly German) | `text` or `text_hash` | `report: true` |
| `features` | normalized PTB-XL+ 12SL measurements + missing mask | `numeric` / `fourier` | `ptbxl_plus_root`, `feature_columns` |

Vocabulary sizes: `all` 71, `superclass` 5 (NORM, MI, STTC, CD, HYP),
`subclass` 23, `rhythm` 12; the order is `dataset.label_names`.
`min_likelihood` drops statements below a likelihood (PTB-XL stores 0–100;
the default 0 keeps every listed statement, as SSSD-ECG does). Height and
weight are often missing — the `numeric` encoder sees the mask.

Default PTB-XL+ features: `HR_Ventr_Global`, `PR_Int_Global`, `QRS_Dur_Global`,
`QT_Int_Global`, `QT_IntBazett_Global`, `P_AxisFront_Global`,
`R_AxisFrontal_Global`, `T_AxisFront_Global`.

## Example configs

- `configs/experiments/ptbxl_sssd_ecg.yaml` — SSSD-ECG-style: 100 Hz, 8 independent leads,
  71 statements, S4D diffusion backbone.
- `configs/experiments/ptbxl_text_demographics.yaml` — flow matching conditioned on the
  report (frozen multilingual BERT), demographics and sex.
