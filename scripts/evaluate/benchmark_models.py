#!/usr/bin/env python3
"""
Benchmark multiple models on standard metrics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    # TODO: Implement benchmarking
    print("Model benchmarking - to be implemented")


if __name__ == "__main__":
    main()
