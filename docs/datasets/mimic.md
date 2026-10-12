# MIMIC-IV-ECG Dataset

[MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) (the *Diagnostic Electrocardiogram Matched Subset*) is a large collection of 10-second, 12-lead diagnostic ECGs with automated machine measurements. It is the main training dataset in ECGEN.

The loader is `ecgen.data.datasets.MIMICIVECG` ([`src/ecgen/data/datasets/mimic.py`](https://github.com/vlbthambawita/ECGEN/blob/main/src/ecgen/data/datasets/mimic.py)), registered as **`mimic_iv_ecg`**. It follows the [`BaseECGDataset`](custom.md) dict contract and is wrapped by the generic `ecgen.data.datamodule.ECGDataModule`, which the training runner builds from the `data:` section of an experiment YAML. API details are in the [Data API reference](../api/data.md).

!!! note "Legacy loader"
    The older tuple-returning `ecgen.data.mimic_dataset.MIMICIVECGDataset` and `ecgen.data.pulse2pulse_mimic` are still present for
    legacy scripts such as `scripts/utils/test_pulse2pulse_setup.py`, but no training path uses them.
    With default parameters `MIMICIVECG` reproduces the legacy dataset's records, splits and
    signals exactly.

## Directory layout

`MIMICIVECG` expects the PhysioNet release unpacked as-is. The directory you pass as `root` must contain:

```text
<root>/
├── machine_measurements.csv        # required
└── files/
    └── p{first 4 digits of subject_id}/
        └── p{subject_id}/
            └── s{study_id}/
                ├── {study_id}.hea  # WFDB header
                └── {study_id}.dat  # WFDB signal
```

For example, subject `10000032`, study `40689238` resolves to
`files/p1000/p10000032/s40689238/40689238` (the WFDB record name without extension).

`machine_measurements.csv` must contain at least these columns, or the constructor raises `ValueError`:

- `subject_id`, `study_id`
- `rr_interval`, `p_onset`, `p_end`, `qrs_onset`, `qrs_end`, `t_end`, `p_axis`, `qrs_axis`, `t_axis`

!!! note "Where to put the data"
    Any location works. Pass it as `root` (Python) or `data.dataset.params.root` (experiment YAML or override). The shipped experiment configs in `configs/experiments/` point to a hard-coded cluster path, so change that value for your setup. `data/raw/mimic/` is the conventional location in this repo (and is gitignored).

## Getting the data

MIMIC-IV-ECG v1.0 is open access on PhysioNet under the ODC Open Database License (ODbL), so no credentialing is required:

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-ecg/1.0/
```

Then set `root` to the directory that contains `machine_measurements.csv` and `files/`.

No preprocessing step is needed. The dataset reads the raw WFDB files directly.

## Basic usage

```python
from ecgen.data.datasets import MIMICIVECG

train_ds = MIMICIVECG(
    root="/path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    split="train",
    max_samples=1000,          # small subset for a smoke test
)

sample = train_ds[0]           # a dict
sample["signal"].shape         # torch.Size([12, 5000]), float32
sample["cond"]["measurements"].shape          # torch.Size([9]), z-scored with train statistics
sample["cond_missing"]["measurements"]        # bool [9], True = missing
sample["meta"]                 # {"subject_id": ..., "study_id": ...}
train_ds.lead_names, train_ds.fs, train_ds.seq_length   # ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', ...], 500, 5000
```

With `ECGDataModule` the samples are collated into an `ecgen.types.ECGBatch` (`signal [B, C, T]`,
`lead_names`, `fs`, `cond`, `cond_missing`, `meta`). All families read the signal with
`get_signal(batch)`, which also accepts the legacy dict and tuple formats.

## Dataset parameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `root` | required | Root of the MIMIC-IV-ECG release (see [layout](#directory-layout)). |
| `split` | `"train"` | `"train"`, `"val"` or `"test"`; filled in by `ECGDataModule`. |
| `val_split` / `test_split` | `0.1` / `0.1` | Fractions of **subjects** in the validation / test split. |
| `seed` | `42` | `random_state` for the subject split. |
| `max_samples` | `None` | Keep only the first *N* records of the selected split, after filtering. Does not affect the measurement normalizer. |
| `skip_missing_check` | `False` | If `True`, skip the check that each record's `.hea` file exists (and emit a `UserWarning`). |
| `leads` | `None` | `None`: the file's leads in **file order** `I, II, III, aVR, aVF, aVL, V1–V6` (aVF before aVL), padded/truncated to `num_leads`. An alias, `"standard12"` (clinical order, aVL before aVF) or `"independent8"` (`I, II, V1–V6`), or a list of lead names, selects leads by name. |
| `num_leads` | `12` | Channel count when `leads` is `None`. |
| `seq_length` | `5000` | Output length in samples (zero-padded at the end or truncated). |
| `scale` | `"zscore"` | `"zscore"`: per record, `(x - mean) / max(std, eps)` over all selected leads (legacy default); `"none"`: physical units (mV); a number: `(x - mean) / number`. |
| `ecg_norm_eps` | `1e-6` | Lower bound on the std for `"zscore"`. |
| `measurements` | `None` | Columns of `machine_measurements.csv` returned in `cond["measurements"]`; `None` = the 9 standard measurements below; `[]` = none. |
| `missing` | `"drop"` | `"drop"`: drop rows with a NaN in any measurement column (legacy). `"mask"`: keep them; NaNs become 0 and are flagged in `cond_missing`. `"mask"` keeps more subjects, so its split differs from `"drop"`. |
| `report` | `false` | Add `cond["report"]`: the non-empty `report_0`…`report_17` machine statements joined by ". " (for text conditioning). |
| `feature_normalizer` | `None` | Use these measurement statistics (a `FeatureNormalizer` or its dict) instead of fitting them on the train split. |

In an experiment YAML these go under `data.dataset.params`; `data.split_params` overrides them per
split:

```yaml
data:
  dataset:
    name: mimic_iv_ecg
    params:
      root: /path/to/mimic-iv-ecg-...-1.0
      max_samples: 1000          # null = full split
      skip_missing_check: false
      val_split: 0.1
      test_split: 0.1
      leads: null                # or independent8, standard12, [I, II, V1]
      seq_length: 5000
      scale: zscore
  split_params:
    val: {max_samples: 1000}
  batch_size: 32
  num_workers: 4
```

The model's channel count (`in_channels` / `num_channels`) must equal the number of selected leads;
the runner checks this before training.

## What the constructor does

1. **Load measurements.** Reads `machine_measurements.csv` and checks the required columns. With
   `missing="drop"` it drops every row with a NaN in the 9 standard measurement columns (and in any
   extra `measurements` columns).
2. **Subject-level split.** Takes the unique `subject_id`s and runs two
   `sklearn.model_selection.train_test_split` calls with `random_state=seed`: first
   `test_size=test_split` → test subjects, then `test_size=val_split / (1 - test_split)` on the
   remainder → validation subjects. No subject appears in more than one split, and the split sizes
   are fractions of *subjects*, not of records.
3. **Measurement normalizer.** Fits a per-column z-score (`FeatureNormalizer`) on the **train**
   split's rows, whichever split is being loaded, unless `feature_normalizer` is given.
   `ECGDataModule` also passes the train dataset's normalizer to val/test and stores it in every
   checkpoint (under the `ECGDataModule` key, with `lead_names`, `fs` and `seq_length`).
4. **Filter to `split`**, then the **missing-file check** (unless `skip_missing_check=True`): keep
   only rows whose `<record>.hea` exists. If `<root>/files` does not exist, the check is skipped.
   This scans every record in the split, which can take a while on the full dataset.
5. **`max_samples`.** Truncates to the first *N* rows.

!!! info "Changed from the legacy loader"
    The legacy dataset computed the measurement statistics on whichever split it loaded, after
    `max_samples` truncation, so val/test features were normalized with different statistics from
    train. They now always come from the full train split.

## Per-sample processing (`__getitem__`)

1. Read the record with `wfdb.rdrecord(...)` and take `p_signal` (mV), transposed to `(leads, time)`.
2. Select leads (by name, or the first `num_leads` in file order) and pad/truncate time to
   `seq_length`.
3. Scale according to `scale` (default: one mean and std over all selected leads, so relative
   amplitudes between leads are kept).
4. Normalize the measurement columns with the train-split statistics; missing values become 0 with
   `cond_missing=True`.

There is no NaN handling for the waveform itself, and an unreadable record raises from `wfdb`.

### Output format

| Key | Shape | dtype | Notes |
|------|-------|-------|-------|
| `signal` | `(C, seq_length)`, default `(12, 5000)` | `float32` | Scaled per record |
| `cond["measurements"]` | `(F,)`, default `(9,)` | `float32` | Z-scored machine measurements |
| `cond_missing["measurements"]` | `(F,)` | `bool` | `True` = missing (only with `missing="mask"`) |
| `meta` | — | — | `subject_id`, `study_id` |

Signal properties of the release: 500 Hz (`fs`), 10 s (5000 samples).

### Conditioning features

Default `measurements`, in order:

| # | Feature | # | Feature | # | Feature |
|---|---------|---|---------|---|---------|
| 0 | `rr_interval` | 3 | `qrs_onset` | 6 | `p_axis` |
| 1 | `p_onset` | 4 | `qrs_end` | 7 | `qrs_axis` |
| 2 | `p_end` | 5 | `t_end` | 8 | `t_axis` |

The current models are unconditional and ignore them. They are returned for conditional models.

## Configuration fragment

`configs/dataset/mimic.yaml` documents defaults under a top-level `dataset:` key
(`data_dir`, `num_leads`, `seq_length`, `sampling_rate`, `normalize`, ...).

!!! warning "Not consumed by the runner"
    The runner does not load `configs/dataset/mimic.yaml`, and its key names differ from the
    `MIMICIVECG` parameters (`data_dir` vs `root`, `normalize` vs `scale`). Set dataset parameters
    in `data.dataset.params` of the experiment YAML. `configs/dataset/ptbxl.yaml` is an empty
    placeholder.

## Data preparation and validation scripts

`scripts/data/` has two CLI entry points. **Both are stubs right now.** They parse their arguments, print a "to be implemented" message and exit.

=== "prepare_mimic.py"

    ```bash
    python scripts/data/prepare_mimic.py \
        --raw_dir data/raw/mimic \
        --output_dir data/processed/mimic \
        [--validate]
    ```

    | Flag | Required | Description |
    |------|----------|-------------|
    | `--raw_dir` | yes | Path to raw MIMIC data |
    | `--output_dir` | yes | Output directory |
    | `--validate` | no | Flag: validate data integrity |

    Output: prints `MIMIC data preparation - to be implemented`. No files are written.

=== "validate_dataset.py"

    ```bash
    python scripts/data/validate_dataset.py \
        --data_dir data/raw/mimic \
        --dataset mimic
    ```

    | Flag | Required | Description |
    |------|----------|-------------|
    | `--data_dir` | yes | Dataset root |
    | `--dataset` | yes | `mimic` or `ptbxl` |

    Output: prints `Validating mimic dataset - to be implemented`.

!!! tip "Checking a download today"
    For a real check of the directory layout, CSV columns and WFDB readability, use `single_file/test_data_loading.py --data-dir <root>` (see [Standalone VQ-VAE](../standalone/vqvae_standalone.md#validating-the-data)), or build a dataset with `max_samples` set:

    ```bash
    python -c "from ecgen.data.datasets import MIMICIVECG as D; \
    d = D('/path/to/mimic', split='val', max_samples=10); s = d[0]; print(len(d), s['signal'].shape, d.lead_names)"
    ```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FileNotFoundError: machine_measurements.csv not found` | `root` should point at the directory that contains the CSV, not at `files/`. |
| `ValueError: machine_measurements.csv missing columns` | Wrong CSV (for example `record_list.csv`), or a `measurements` column that does not exist. |
| `ValueError: Leads [...] are not available` | A name in `leads` is misspelled; MIMIC uses `I, II, III, aVR, aVF, aVL, V1`–`V6`. |
| `ValueError: Model expects N channels but the dataset yields M leads` | Set `leads` / `num_leads` or the model's `in_channels` / `num_channels` so they agree. |
| `num_samples should be a positive integer value, but got num_samples=0` | The split is empty: no `.hea` files were found, or `max_samples=0`. |
| Slow startup on the full dataset | The missing-file check stats every record. Use `skip_missing_check: true` once you know the download is complete. |
| `NaN` losses | The dataset does not clean NaNs in `p_signal`. If some records contain them, filter those records or apply `np.nan_to_num` in a subclass. |

## Citation

If you use MIMIC-IV-ECG, cite the dataset and PhysioNet using the citation text on the [dataset page](https://physionet.org/content/mimic-iv-ecg/1.0/) (Gow *et al.*, "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset", version 1.0, PhysioNet, 2023).
