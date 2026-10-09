# MIMIC-IV-ECG Dataset

[MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) (the *Diagnostic Electrocardiogram Matched Subset*) is a large collection of 10-second, 12-lead diagnostic ECGs with automated machine measurements. It is the main training dataset in ECGEN.

The implementation is in `data/datasets/mimic/`:

| File | Contents |
|------|----------|
| `dataset.py` | `MIMICIVECGDataset`, the PyTorch `Dataset` used by all MIMIC training scripts |
| `datamodule.py` | `ECGDataModule`, a minimal **non-Lightning** dataclass placeholder (no training script uses it) |
| `pulse2pulse_datamodule.py` | `Pulse2PulseMIMICDataModule` + `ECGDatasetAdapter` for the GAN (still imports the legacy `ecgen` package, see below) |

API details are in the [Data API reference](../api/data.md).

## Directory layout

`MIMICIVECGDataset` expects the PhysioNet release unpacked as-is. The directory you pass as `mimic_path` must contain:

```text
<mimic_path>/
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
    Any location works. Pass it as `mimic_path` (Python) or `data.params.mimic_path` (experiment YAML). The shipped experiment configs in `configs/experiments/` point to a hard-coded cluster path, so change that value for your setup. `data/raw/mimic/` is the conventional location in this repo (and is gitignored).

## Getting the data

MIMIC-IV-ECG v1.0 is open access on PhysioNet under the ODC Open Database License (ODbL), so no credentialing is required:

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-ecg/1.0/
```

Then set `mimic_path` to the directory that contains `machine_measurements.csv` and `files/`.

No preprocessing step is needed. The dataset reads the raw WFDB files directly.

## Basic usage

```python
from data.datasets.mimic import MIMICIVECGDataset
# equivalent: from data.datasets.mimic.dataset import MIMICIVECGDataset

train_ds = MIMICIVECGDataset(
    mimic_path="/path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    split="train",
    max_samples=1000,          # small subset for a smoke test
)

ecg, features = train_ds[0]    # a tuple, not a dict
print(ecg.shape, ecg.dtype)    # torch.Size([12, 5000]) torch.float32
print(features.shape)          # torch.Size([9])
```

!!! warning "Return type is a tuple"
    `__getitem__` returns `(ecg, features)`, **not** a dict. A default `DataLoader` therefore yields `(ecg_batch, features_batch)` with shapes `(B, 12, 5000)` and `(B, 9)`.
    `VQVAELightning`/`PriorLightning` accept this tuple form directly, and so does the `VAEVisualizationCallback`. `VAELightning` and the Pulse2Pulse GAN expect `batch["ecg_signals"]` instead. `train_vae_mimic.py` uses its own tuple-based wrapper module, and the GAN wraps the dataset in `ECGDatasetAdapter`.

!!! note "Not a `BaseECGDataset` subclass"
    `MIMICIVECGDataset` subclasses `torch.utils.data.Dataset` directly, not `data.datasets.base.BaseECGDataset`. Its `num_leads` and `seq_length` are plain instance attributes. See [Adding a dataset](custom.md) for the base-class contract that new datasets should follow.

## Constructor arguments

```python
MIMICIVECGDataset(
    mimic_path: str,
    split: str = "train",
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42,
    skip_missing_check: bool = False,
    ecg_norm_eps: float = 1e-6,
    ecg_norm_factor: Optional[float] = None,
    num_leads: int = 12,
    seq_length: int = 5000,
)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `mimic_path` | required | Root of the MIMIC-IV-ECG release (see [layout](#directory-layout)). |
| `split` | `"train"` | One of `"train"`, `"val"`, `"test"`. Any other value matches no rows and gives an empty dataset. |
| `val_split` | `0.1` | Fraction of **subjects** in the validation split. |
| `test_split` | `0.1` | Fraction of **subjects** in the test split. |
| `max_samples` | `None` | Keep only the first *N* records of the selected split, after filtering. |
| `seed` | `42` | `random_state` for the subject split. Use the same seed for all three splits. |
| `skip_missing_check` | `False` | If `True`, skip the check that each record's `.hea` file exists (and emit a `UserWarning`). |
| `ecg_norm_eps` | `1e-6` | Lower bound on the per-sample std used for normalization. |
| `ecg_norm_factor` | `None` | If set, divide by this fixed scale instead of the per-sample std. |
| `num_leads` | `12` | Output lead count. Extra leads are dropped and missing leads are zero-padded. |
| `seq_length` | `5000` | Output length in samples. Longer signals are truncated and shorter ones zero-padded at the end. |

Other public attributes: `FEATURE_NAMES` (class attribute), `measurements` (the filtered `pandas.DataFrame`, including a `split` column), and `feature_stats` (per-feature `{"mean", "std"}`).

## What the constructor does

The steps run in this order:

1. **Load measurements.** Reads `machine_measurements.csv`, checks for the required columns, and **drops every row with a NaN in any of the 9 feature columns**.
2. **Subject-level split.** Takes the unique `subject_id`s and runs two `sklearn.model_selection.train_test_split` calls with `random_state=seed`:
    - first `test_size=test_split` → test subjects;
    - then `test_size=val_split / (1 - test_split)` on the remainder → validation subjects.

    So no subject appears in more than one split, and the split sizes are fractions of *subjects*, not of records.
3. **Filter to `split`.**
4. **Missing-file check** (unless `skip_missing_check=True`). Keeps only rows whose `<record>.hea` exists. If `<mimic_path>/files` does not exist, the check is skipped silently. This scans every record in the split, which can take a while on the full dataset.
5. **`max_samples`.** Truncates to the first *N* rows (`DataFrame.head`).
6. **Feature statistics.** Computes the mean and std (+1e-6) of each feature **on the selected split only, after truncation**.

!!! info "Each split is constructed independently"
    Every `MIMICIVECGDataset` instance re-reads the CSV and re-computes the split. Train, val and test agree because they share `seed`, `val_split` and `test_split`. Feature normalization statistics are per split, not taken from the training set.

## Per-sample processing (`__getitem__`)

1. Read the record with `wfdb.rdrecord(...)` and take `p_signal` (physical units, mV), transposed to `(leads, time)` as `float32`.
2. Pad or truncate leads to `num_leads`, then pad (with zeros at the end) or truncate time to `seq_length`.
3. **Normalize the ECG per sample, with one scalar for all leads:**

    ```python
    ecg = (ecg - ecg.mean()) / scale
    # scale = ecg_norm_factor            if ecg_norm_factor is not None
    #       = max(ecg.std(), ecg_norm_eps) otherwise
    ```

    By default each sample has mean 0 and std 1 over all leads. Relative amplitudes between leads are kept.
4. **Normalize features** with a z-score using the split's `feature_stats`.

There is no NaN handling for the waveform itself, and an unreadable record raises from `wfdb`.

### Output format

| Item | Shape | dtype | Notes |
|------|-------|-------|-------|
| `ecg` | `(num_leads, seq_length)`, default `(12, 5000)` | `float32` | Per-sample normalized |
| `features` | `(9,)` | `float32` | Z-scored machine measurements, in `FEATURE_NAMES` order |

Signal properties of the release: 500 Hz, 10 s (5000 samples). The lead order is whatever the WFDB header lists. For MIMIC-IV-ECG this is **I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6** (note aVF comes before aVL).

### Conditioning features

`MIMICIVECGDataset.FEATURE_NAMES`, in order:

| # | Feature | # | Feature | # | Feature |
|---|---------|---|---------|---|---------|
| 0 | `rr_interval` | 3 | `qrs_onset` | 6 | `p_axis` |
| 1 | `p_onset` | 4 | `qrs_end` | 7 | `qrs_axis` |
| 2 | `p_end` | 5 | `t_end` | 8 | `t_axis` |

The current models are unconditional and ignore the features. They are returned for future conditional models.

## Using it with Lightning

The repo has no shared Lightning DataModule for MIMIC. Each training script (`scripts/train/train_vae_mimic.py`, `train_vqvae_mimic.py`, `train_prior_mimic.py`) defines its own `LightningDataModule` inline. They all look like this:

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.datasets.mimic.dataset import MIMICIVECGDataset


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, **ds_kwargs):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers
        self.ds_kwargs = ds_kwargs  # val_split, test_split, max_samples, seed, ...

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = MIMICIVECGDataset(self.data_dir, split="train", **self.ds_kwargs)
            self.val_dataset = MIMICIVECGDataset(self.data_dir, split="val", **self.ds_kwargs)
        if stage in ("test", None):
            self.test_dataset = MIMICIVECGDataset(self.data_dir, split="test", **self.ds_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
```

!!! note
    These script datamodules pass the same `max_samples` to train, val **and** test, so `max_samples: 1000` gives up to 1000 records per split. `ecg_norm_eps` and `ecg_norm_factor` are not exposed by any training script, so the defaults always apply.

### Pulse2Pulse datamodule

`data/datasets/mimic/pulse2pulse_datamodule.py` defines `Pulse2PulseMIMICDataModule(config)`, which takes a `Pulse2PulseMIMICConfig` or dict with these fields:

| Field | Default |
|-------|---------|
| `data_dir` | required |
| `batch_size` | `128` |
| `num_workers` | `4` |
| `max_samples` | `None` (val is capped at `min(max_samples or 1000, 1000)`) |
| `skip_missing_check` | `True` |
| `num_channels` | `8` (first 8 leads are kept by `ECGDatasetAdapter`) |
| `seq_length` | `5000` |

`ECGDatasetAdapter` turns `(ecg, features)` into `{"ecg_signals": ecg[:num_leads]}`.

!!! warning "Legacy import"
    `Pulse2PulseMIMICDataModule.setup()` imports `from ecgen.data.mimic_dataset import MIMICIVECGDataset` (the legacy `src/ecgen` package), not the `data.datasets.mimic` version. That package is not in `pyproject.toml`, so this module only works when `src/` is importable.

## Configuration

### Experiment YAML (what the scripts actually read)

The training scripts read the dataset settings from `data.params` in `configs/experiments/*.yaml`:

```yaml
data:
  target: data.datasets.mimic.MIMICIVECGDataset   # informational only
  params:
    mimic_path: /path/to/mimic-iv-ecg-...-1.0
    batch_size: 32
    num_workers: 4
    max_samples: 1000          # null = full dataset
    skip_missing_check: false
    val_split: 0.1
    test_split: 0.1
    num_leads: 12              # not read; the lead count comes from model.params.config.in_channels
    seq_length: 5000
```

The VAE, VQ-VAE and prior scripts read `mimic_path`, `batch_size`, `num_workers`, `max_samples`, `skip_missing_check`, `val_split`, `test_split` and `seq_length`. `num_leads` is set from the model's `in_channels`. The dataset `seed` is `experiment.seed`. Equivalent CLI flags are `--data-dir`, `--batch-size`, `--num-workers`, `--max-samples`, `--skip-missing-check`, `--val-split`, `--test-split` and `--seq-length`. See the [Training guide](../training/guide.md).

### `configs/dataset/mimic.yaml` (reference fragment)

```yaml
dataset:
  name: mimic
  data_dir: data/raw/mimic
  num_leads: 12
  seq_length: 5000
  sampling_rate: 500  # Hz
  batch_size: 32
  num_workers: 4
  val_split: 0.1
  test_split: 0.1
  normalize: true
  skip_missing_check: false
  max_samples: null
```

!!! warning "Not consumed by any script"
    No training script loads `configs/dataset/mimic.yaml`. It documents the defaults only. Its key names (`data_dir`, `normalize`, `sampling_rate`) also differ from what the scripts read (`mimic_path`), and `normalize` has no effect because normalization is always on. `configs/dataset/ptbxl.yaml` is an empty placeholder.

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
    For a real check of the directory layout, CSV columns and WFDB readability, use `single_file/test_data_loading.py --data-dir <mimic_path>` (see [Standalone VQ-VAE](../standalone/vqvae_standalone.md#validating-the-data)), or build a dataset with `max_samples` set:

    ```bash
    python -c "from data.datasets.mimic import MIMICIVECGDataset as D; \
    d = D('/path/to/mimic', split='val', max_samples=10); e, f = d[0]; print(len(d), e.shape, f.shape)"
    ```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FileNotFoundError: machine_measurements.csv not found` | `mimic_path` should point at the directory that contains the CSV, not at `files/`. |
| `ValueError: machine_measurements.csv missing columns` | Wrong CSV (for example `record_list.csv`). Use `machine_measurements.csv`. |
| `num_samples should be a positive integer value, but got num_samples=0` | The split is empty: no `.hea` files were found, `split` is misspelled, or `max_samples=0`. |
| Slow startup on the full dataset | The missing-file check stats every record. Use `skip_missing_check=True` once you know the download is complete. |
| `NaN` losses | The dataset does not clean NaNs in `p_signal`. If some records contain them, filter those records or apply `np.nan_to_num` in a subclass or wrapper. |

## Citation

If you use MIMIC-IV-ECG, cite the dataset and PhysioNet using the citation text on the [dataset page](https://physionet.org/content/mimic-iv-ecg/1.0/) (Gow *et al.*, "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset", version 1.0, PhysioNet, 2023).
