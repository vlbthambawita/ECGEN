"""Benchmarking utilities for model evaluation."""

import time
import torch
from pathlib import Path
from typing import Any, Callable
import json


class ModelBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, model: Any, device: str = "cuda"):
        self.model = model
        self.device = device
        self.results = {}
    
    def measure_inference_time(self, input_shape: tuple, n_runs: int = 100, warmup: int = 10):
        """
        Measure inference time.
        
        Args:
            input_shape: Input tensor shape
            n_runs: Number of runs for averaging
            warmup: Number of warmup runs
        """
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # Measure
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_runs
        
        self.results["inference_time_ms"] = avg_time * 1000
        self.results["throughput_samples_per_sec"] = input_shape[0] / avg_time
    
    def measure_memory_usage(self, input_shape: tuple):
        """Measure GPU memory usage."""
        if self.device != "cuda":
            return
        
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        
        self.results["memory_allocated_mb"] = memory_allocated
        self.results["memory_reserved_mb"] = memory_reserved
    
    def count_parameters(self):
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.results["total_parameters"] = total_params
        self.results["trainable_parameters"] = trainable_params
        self.results["total_parameters_millions"] = total_params / 1e6
    
    def run_full_benchmark(self, input_shape: tuple, save_path: Path = None):
        """Run full benchmark suite."""
        print("Running benchmark...")
        
        print("  - Counting parameters...")
        self.count_parameters()
        
        print("  - Measuring inference time...")
        self.measure_inference_time(input_shape)
        
        if self.device == "cuda":
            print("  - Measuring memory usage...")
            self.measure_memory_usage(input_shape)
        
        print("Benchmark complete!")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {save_path}")
        
        return self.results
    
    def print_results(self):
        """Print benchmark results."""
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*50 + "\n")
