# Evaluation

Evaluation tools and metrics for model assessment.

## Structure

- `metrics/` - Evaluation metrics
  - `signal_quality.py` - ECG signal quality metrics (SNR, baseline wander, etc.)
  - `diversity.py` - Generation diversity metrics
  - `fidelity.py` - Fidelity metrics (FID, reconstruction error, etc.)
- `visualize.py` - Evaluation visualization
- `benchmark.py` - Model benchmarking utilities

## Usage

### Signal Quality Metrics

```python
from evaluation.metrics.signal_quality import calculate_quality_metrics

metrics = calculate_quality_metrics(ecg_signal, sampling_rate=500)
print(f"SNR: {metrics['snr']:.2f} dB")
```

### Diversity Metrics

```python
from evaluation.metrics.diversity import pairwise_distance_diversity

diversity = pairwise_distance_diversity(generated_samples)
print(f"Diversity: {diversity:.4f}")
```

### Fidelity Metrics

```python
from evaluation.metrics.fidelity import calculate_fid

fid_score = calculate_fid(real_features, generated_features)
print(f"FID: {fid_score:.4f}")
```

### Benchmarking

```python
from evaluation.benchmark import ModelBenchmark

benchmark = ModelBenchmark(model, device="cuda")
results = benchmark.run_full_benchmark(
    input_shape=(32, 12, 5000),
    save_path="benchmark_results.json"
)
benchmark.print_results()
```

### Visualization

```python
from evaluation.visualize import plot_evaluation_results

metrics = {
    "SNR": 25.3,
    "FID": 12.5,
    "Diversity": 0.85,
}

fig = plot_evaluation_results(metrics, save_path="eval_results.png")
```

## Comprehensive Evaluation

```python
# Evaluate model comprehensively
from evaluation.metrics.signal_quality import calculate_quality_metrics
from evaluation.metrics.diversity import pairwise_distance_diversity
from evaluation.metrics.fidelity import calculate_fid, reconstruction_error

# Generate samples
generated_samples = model.generate(n_samples=100)
real_samples = test_dataset[:100]

# Calculate metrics
quality = calculate_quality_metrics(generated_samples[0])
diversity = pairwise_distance_diversity(generated_samples)
fid = calculate_fid(extract_features(real_samples), 
                   extract_features(generated_samples))
recon_error = reconstruction_error(real_samples, reconstructed_samples)

# Combine results
evaluation_results = {
    **quality,
    "diversity": diversity,
    "fid": fid,
    "reconstruction_error": recon_error,
}

# Visualize
from evaluation.visualize import plot_evaluation_results
fig = plot_evaluation_results(evaluation_results)
```

## Adding Custom Metrics

Create new metric functions following the pattern:

```python
def my_custom_metric(samples: Tensor | np.ndarray) -> float:
    """
    Calculate custom metric.
    
    Args:
        samples: Input samples
        
    Returns:
        Metric value
    """
    # Your metric calculation
    return metric_value
```
