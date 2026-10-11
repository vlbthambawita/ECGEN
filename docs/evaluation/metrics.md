# Evaluation Metrics

The `evaluation` package provides metrics for judging generated ECGs: signal quality,
diversity and fidelity. It also has a model benchmark helper and plotting utilities for metric
results. A small set of torch-based metrics for training-time monitoring lives separately in
`training/metrics/`.

```
evaluation/
├── metrics/
│   ├── __init__.py         # re-exports the 10 public metric functions
│   ├── signal_quality.py   # SNR, baseline wander, powerline interference
│   ├── diversity.py        # pairwise distance, per-class diversity, mode coverage
│   └── fidelity.py         # FID / Fréchet distance, reconstruction error, correlation
├── benchmark.py            # ModelBenchmark: params, latency, GPU memory
└── visualize.py            # bar chart and model×metric heat map
```

All `evaluation` metrics accept either a `torch.Tensor` (moved to CPU and converted) or a
NumPy array, and return plain Python `float`s or `dict`s. They are NumPy/SciPy/scikit-learn
code and are not differentiable.

```python
from evaluation.metrics import (
    signal_to_noise_ratio, baseline_wander, powerline_interference, calculate_quality_metrics,
    pairwise_distance_diversity, intra_class_diversity, mode_coverage,
    calculate_fid, reconstruction_error, structural_similarity,
)
from evaluation.metrics.fidelity import frechet_distance   # not re-exported
```

!!! warning "Python ≥ 3.10 required at import time"
    `signal_quality.py`, `diversity.py`, `fidelity.py` and `visualize.py` use `X | Y` type
    hints without `from __future__ import annotations`. Those hints are evaluated when the
    module is imported, so importing `evaluation.metrics` or `evaluation.visualize` fails on
    Python 3.8/3.9, even though `pyproject.toml` declares `requires-python = ">=3.8"`.

## `ecgen.evaluation` (packaged metrics)

The `ecgen` package ships the metrics from the training plan (§8). They take
`[N, C, T]` arrays or tensors.

| Function | What it measures |
|---|---|
| `distribution_metrics(real, fake, fs, extractor=None)` | Fréchet distance, MMD (RBF, median bandwidth) and k-NN precision / recall on standardized features — hand-crafted by default (`handcrafted_features`: heart rate, per-lead amplitude percentiles and band powers, inter-lead correlations), or any network via `module_features(model)` |
| `heart_rate(x, fs, method="simple" \| "neurokit")`, `adherence(requested, measured)` | Requested vs. measured heart rate (MAE, Pearson r) — condition adherence for measurement conditioning |
| `fit_classifier`, `label_adherence`, `tstr` | A small 1D-ResNet: train it on real labelled ECGs (separate from the generator's training data), then score generated ECGs against the labels they were generated for (macro AUROC), or train on synthetic and test on real (TSTR) |
| `reconstruction_metrics(pred, true, unobserved)` | MAE, RMSE, Pearson r on the unobserved part (lead reconstruction, imputation) |
| `inter_lead_correlation_error(real, fake)` | Mean absolute difference of the average lead-correlation matrices (MIDT-ECG) |
| `einthoven_residual(x, lead_names)` | Relative violation of III = II − I, aVR, aVL, aVF identities (≈0 for real 12-lead ECGs) |
| `memorization_ratio(fake, train, real_holdout)` | Nearest-neighbour distance of generated ECGs to the training set relative to held-out real ECGs (≪ 1 suggests copying) |

```bash
ecgen sample runs/.../last.ckpt -n 256 -o gen/ --format npy
ecgen evaluate gen/samples.npy configs/experiments/diffusion_mimic.yaml --split test -n 256
python scripts/evaluate/condition_adherence.py <ckpt> --family diffusion --target rr_interval
python scripts/evaluate/lead_reconstruction.py <ckpt> --family diffusion --keep I
```

The sections below document the older research-layout metrics in `evaluation/`.

## Signal quality (`evaluation/metrics/signal_quality.py`)

These are reference-free metrics computed on a single signal.

| Function | Signature | Returns |
|----------|-----------|---------|
| `signal_to_noise_ratio` | `(ecg) -> float` | SNR in dB |
| `baseline_wander` | `(ecg, sampling_rate: int = 500) -> float` | Std of the < 0.5 Hz component |
| `powerline_interference` | `(ecg, sampling_rate: int = 500) -> float` | Fraction of spectral power at 50 Hz + 60 Hz |
| `calculate_quality_metrics` | `(ecg, sampling_rate: int = 500) -> dict` | `{"snr", "baseline_wander", "powerline_interference"}` |

**`signal_to_noise_ratio(ecg)`** computes
`10·log10(mean(ecg²) / (std(diff(ecg))² + 1e-8))`. It estimates "noise" as the spread of
sample-to-sample differences along the last axis, which is a smoothness proxy rather than a
true noise measurement. It works on any shape. For `(C, L)` or `(B, C, L)` inputs it returns
one pooled value.

**`baseline_wander(ecg, sampling_rate=500)`** applies a 4th-order Butterworth high-pass at
0.5 Hz (`scipy.signal.filtfilt` along the last axis). It returns the standard deviation of the
removed low-frequency part, `ecg - filtered`, in the same units as the input. It works on 1-D
or multi-dimensional input (pooled std). The input must be longer than the filter's padding
length (more than 15 samples for this order-4 filter).

**`powerline_interference(ecg, sampling_rate=500)`** takes the FFT and returns
`(P(50 Hz) + P(60 Hz)) / ΣP`, using the nearest FFT bins to 50 and 60 Hz.

!!! bug "1-D input only"
    `powerline_interference` uses `len(ecg)` for the frequency axis and indexes `power[idx]`
    on the first axis. It is only correct for a **single-lead 1-D signal** of shape `(L,)`.
    With a `(C, L)` array it raises
    `TypeError: only length-1 arrays can be converted to Python scalars`.
    Since `calculate_quality_metrics` calls it, `calculate_quality_metrics` is also limited to
    1-D input. Loop over leads yourself:

    ```python
    ecg = samples[0]                      # (C, L)
    per_lead = [calculate_quality_metrics(ecg[c], sampling_rate=500) for c in range(ecg.shape[0])]
    ```

## Diversity (`evaluation/metrics/diversity.py`)

These metrics take a set of samples. Every sample is flattened to a vector, so `(N, C, L)`
becomes `(N, C·L)`.

| Function | Signature | Returns |
|----------|-----------|---------|
| `pairwise_distance_diversity` | `(samples, metric: str = "euclidean") -> float` | Mean pairwise distance |
| `intra_class_diversity` | `(samples, labels: np.ndarray) -> dict` | `{str(label): mean pairwise distance}` |
| `mode_coverage` | `(samples, real_samples, n_clusters: int = 10) -> float` | Fraction of real-data clusters hit, in `[0, 1]` |

**`pairwise_distance_diversity(samples, metric="euclidean")`** computes the mean of
`scipy.spatial.distance.pdist(samples.reshape(N, -1), metric=metric)`. `metric` can be any
`pdist` metric (`"euclidean"`, `"cosine"`, `"correlation"`, …). Higher values mean more
diverse samples. Cost is O(N²·C·L).

**`intra_class_diversity(samples, labels)`** calls `pairwise_distance_diversity` (Euclidean)
on `samples[labels == label]` for each unique label and skips classes with fewer than 2
samples. `labels` must be a NumPy array of length N, because it is used as a boolean mask.

**`mode_coverage(samples, real_samples, n_clusters=10)`** fits
`sklearn.cluster.KMeans(n_clusters, random_state=42)` on the flattened real samples, assigns
each generated sample to its nearest centroid, and returns
`#distinct clusters hit / n_clusters`. A value of 1.0 means every real-data mode received at
least one generated sample. You need at least `n_clusters` real samples.

## Fidelity (`evaluation/metrics/fidelity.py`)

These metrics compare generated (or reconstructed) data against real data.

| Function | Signature | Returns |
|----------|-----------|---------|
| `frechet_distance` | `(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float` | Fréchet distance between two Gaussians |
| `calculate_fid` | `(real_features, generated_features) -> float` | FID |
| `reconstruction_error` | `(real, reconstructed, metric: str = "mse") -> float` | MSE / MAE / RMSE |
| `structural_similarity` | `(real, generated) -> float` | Global normalised correlation |

**`calculate_fid(real_features, generated_features)`** expects 2-D feature matrices
`(N, D)` and `(M, D)`. It computes the mean and covariance of each (`np.cov(..., rowvar=False)`)
and calls `frechet_distance`:

$$
\mathrm{FD} = \lVert \mu_1-\mu_2\rVert^2 + \mathrm{Tr}\big(\Sigma_1+\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2}\big)
$$

If `sqrtm` is not finite, `frechet_distance` retries with `eps·I` added to both covariances.
It raises `ValueError` if the result has a significant imaginary part.

!!! note "No feature extractor is included"
    The package has no pretrained ECG encoder. You need to supply the embeddings yourself, for
    example the VAE/VQ-VAE encoder output or a classifier's penultimate layer. Passing raw
    flattened ECGs (`D = 12·5000`) is technically possible but builds 60000×60000 covariance
    matrices, and needs `N ≫ D` samples for a meaningful estimate. Lower is better.

**`reconstruction_error(real, reconstructed, metric="mse")`** computes the element-wise
`"mse"`, `"mae"` or `"rmse"` averaged over all elements. Any other value raises `ValueError`.
Shapes must match or broadcast. Use it for autoencoder reconstructions, for example
[VAE](../models/vae.md) / VQ-VAE.

**`structural_similarity(real, generated)`** z-normalises each array with its *global* mean
and std, then returns `mean(real_norm * generated_norm)`, i.e. a Pearson-style correlation
over all elements. Despite the name, it is **not** SSIM. It needs paired, same-shaped inputs
(such as reconstructions) and returns a value of roughly 1.0 for identical signals.

## Example: evaluating generated samples

```python
import torch
from evaluation.metrics import (
    calculate_quality_metrics, pairwise_distance_diversity, mode_coverage, calculate_fid,
)

generated = model.generate_samples(256).cpu()      # e.g. Pulse2Pulse: (256, 8, 5000)
real = torch.stack([ds[i][0][:8] for i in range(256)])  # matching real ECGs

results = {
    **calculate_quality_metrics(generated[0, 1]),  # lead II of sample 0 (1-D!)
    "diversity": pairwise_distance_diversity(generated),
    "mode_coverage": mode_coverage(generated, real, n_clusters=10),
    "fid": calculate_fid(encoder(real), encoder(generated)),   # encoder = your feature extractor
}
```

## Benchmarking (`evaluation/benchmark.py`)

`ModelBenchmark` measures size and speed for any `nn.Module` whose `forward` accepts a single
`torch.randn(input_shape)` tensor.

```python
ModelBenchmark(model, device: str = "cuda")
```

| Method | What it records in `self.results` |
|--------|-----------------------------------|
| `count_parameters()` | `total_parameters`, `trainable_parameters`, `total_parameters_millions` |
| `measure_inference_time(input_shape, n_runs=100, warmup=10)` | `inference_time_ms` (mean per forward), `throughput_samples_per_sec` (`input_shape[0]` / mean time) |
| `measure_memory_usage(input_shape)` | `memory_allocated_mb`, `memory_reserved_mb` (peak); no-op unless `device == "cuda"` |
| `run_full_benchmark(input_shape, save_path=None)` | Runs all of the above, optionally writes JSON to `save_path`, returns `results` |
| `print_results()` | Pretty-prints `results` |

```python
from evaluation.benchmark import ModelBenchmark
from ecgen.families.gan.pulse2pulse import Pulse2PulseGAN, Pulse2PulseConfig

model = Pulse2PulseGAN(Pulse2PulseConfig()).netG.to("cuda")   # move it yourself
bench = ModelBenchmark(model, device="cuda")
bench.run_full_benchmark(input_shape=(32, 8, 5000), save_path="benchmark_results.json")
bench.print_results()
```

!!! note
    `ModelBenchmark` does **not** move the model to `device`; it only moves the dummy input.
    CUDA synchronisation and memory measurement only happen when `device` is exactly the
    string `"cuda"`, so `"cuda:1"` skips both.

## Visualisation (`evaluation/visualize.py`)

| Function | Signature | Output |
|----------|-----------|--------|
| `plot_evaluation_results` | `(metrics: dict, save_path=None, figsize=(12, 8)) -> plt.Figure` | Bar chart of `{name: value}` with value labels |
| `plot_comparison_matrix` | `(models: list[str], metrics: dict[str, list[float]], save_path=None, figsize=(12, 8)) -> plt.Figure` | Metrics × models heat map (`RdYlGn_r`) with annotations |

For `plot_comparison_matrix`, each `metrics[name]` must be a list with one value per model, in
the order of `models`. When `save_path` is given, both functions save the figure at 300 dpi
with `bbox_inches="tight"`. Bar colours in `plot_evaluation_results` form a fixed gradient by
position and do not reflect the values. For ECG waveform plots, see
[Visualization](../visualization/guide.md).

```python
from evaluation.visualize import plot_evaluation_results, plot_comparison_matrix

plot_evaluation_results({"snr": 24.0, "diversity": 40.4, "fid": 4.6}, save_path="eval.png")
plot_comparison_matrix(
    models=["VAE", "VQ-VAE", "Pulse2Pulse"],
    metrics={"fid": [12.1, 8.3, 9.7], "diversity": [35.2, 41.0, 40.4]},
    save_path="comparison.png",
)
```

## Evaluation scripts (`scripts/evaluate/`)

!!! warning "Placeholders"
    Both scripts only parse their arguments and print a "to be implemented" message. They do
    not load checkpoints or compute any metrics yet. Use the Python API above instead.

| Script | Flags | Current behaviour |
|--------|-------|-------------------|
| `evaluate_model.py` | `--checkpoint` (required), `--data_dir` (required), `--output_dir` (default `evaluation_results`) | Prints `Model evaluation - to be implemented` |
| `benchmark_models.py` | `--checkpoints` (required, one or more), `--data_dir` (required), `--output_dir` (default `benchmark_results`) | Prints `Model benchmarking - to be implemented` |

```bash
python scripts/evaluate/evaluate_model.py --checkpoint path/to/model.ckpt --data_dir /path/to/data
python scripts/evaluate/benchmark_models.py --checkpoints a.ckpt b.ckpt --data_dir /path/to/data
```

These scripts use underscore flags (`--data_dir`), while the training scripts use hyphens
(`--data-dir`). Each script adds the repo root to `sys.path`, so they run from any working
directory.

## Training-time metrics (`training/metrics/`)

These are torch functions that return tensors, intended for use inside training/validation
steps. `training/metrics/__init__.py` re-exports nothing, so import from the submodules.

| Module | Function | Description |
|--------|----------|-------------|
| `training.metrics.reconstruction` | `reconstruction_mse(pred, target) -> Tensor` | `mean((pred - target)²)` |
| `training.metrics.reconstruction` | `reconstruction_mae(pred, target) -> Tensor` | `mean(|pred - target|)` |
| `training.metrics.generation` | `signal_to_noise_ratio(signal, noise) -> Tensor` | `10·log10(mean(signal²) / (mean(noise²) + 1e-8))`, given an explicit noise tensor |
| `training.metrics.clinical` | `wasserstein_distance(d_real, d_fake) -> Tensor` | `d_real.mean() - d_fake.mean()`, the WGAN critic estimate (higher means the critic separates real from fake better) |
| `training.metrics.clinical` | `gradient_penalty_loss(discriminator, real_data, fake_data, lmbda=10.0) -> Tensor` | WGAN-GP penalty on random interpolates of `(B, C, L)` tensors; same as `calc_gradient_penalty` in the [Pulse2Pulse GAN](../models/gan.md#training-objective-wgan-gp) |

!!! note "Two different `signal_to_noise_ratio`s"
    `training.metrics.generation.signal_to_noise_ratio(signal, noise)` takes an explicit noise
    tensor. `evaluation.metrics.signal_to_noise_ratio(ecg)` estimates noise from first
    differences. Despite its name, `training/metrics/clinical.py` contains only GAN critic
    metrics, not clinical ones such as intervals or heart rate.

None of the current Lightning modules import `training.metrics`. `Pulse2PulseGAN` computes
its Wasserstein estimate and gradient penalty inline.

## See also

- API reference: [Evaluation](../api/evaluation.md), [Training](../api/training.md)
- [Pulse2Pulse GAN](../models/gan.md), [VAE](../models/vae.md)
- [Visualization](../visualization/guide.md)
