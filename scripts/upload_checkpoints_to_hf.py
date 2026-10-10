#!/usr/bin/env python3
"""
CLI wrapper for ecgen.f.upload_hf.

Usage:
    python scripts/upload_checkpoints_to_hf.py
    python scripts/upload_checkpoints_to_hf.py --config configs/upload_checkpoints.yaml
    python scripts/upload_checkpoints_to_hf.py --dry-run

Authentication:
    huggingface-cli login
    # or: export HF_TOKEN=hf_...

Install with HF support:
    pip install 'ecgen[hf]'
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    default_config = Path(__file__).parent.parent / "configs" / "upload_checkpoints.yaml"

    parser = argparse.ArgumentParser(description="Upload model checkpoints to HuggingFace.")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to the YAML config file (default: configs/upload_checkpoints.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    from ecgen.f import upload_hf

    try:
        upload_hf(config_path, dry_run=args.dry_run)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
