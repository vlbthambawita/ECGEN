#!/usr/bin/env python3
"""
Evaluate trained model on test set.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    # TODO: Implement evaluation
    print("Model evaluation - to be implemented")


if __name__ == "__main__":
    main()
