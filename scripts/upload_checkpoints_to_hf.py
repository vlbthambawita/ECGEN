#!/usr/bin/env python3
"""
Upload model checkpoints to a HuggingFace repository.

Usage:
    python scripts/upload_checkpoints_to_hf.py
    python scripts/upload_checkpoints_to_hf.py --config configs/upload_checkpoints.yaml
    python scripts/upload_checkpoints_to_hf.py --dry-run

Requirements:
    pip install huggingface_hub pyyaml

Authentication:
    huggingface-cli login
    # or set environment variable: HF_TOKEN=your_token
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to the repo root (two levels up from scripts/)
        repo_root = Path(__file__).parent.parent
        p = repo_root / p
    return p.resolve()


def upload_checkpoints(config: dict, dry_run: bool = False) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = config["hf_repo_id"]
    repo_type = config.get("hf_repo_type", "model")
    commit_message = config.get("commit_message", "Upload model checkpoints")
    private = config.get("private", False)
    checkpoints = config.get("checkpoints", [])

    if not checkpoints:
        print("No checkpoints defined in config. Nothing to upload.")
        return

    # Ensure repo_id includes a namespace (username/repo or org/repo)
    if "/" not in repo_id:
        print(
            f"WARNING: repo_id '{repo_id}' has no namespace. "
            "It will be uploaded under your HF account. "
            "Set it to 'your_username/ECGEN' in the config to be explicit."
        )

    api = HfApi()

    if not dry_run:
        print(f"Creating/verifying repository '{repo_id}' ({repo_type}, private={private}) ...")
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            print(f"  Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"ERROR creating repository: {e}")
            sys.exit(1)
    else:
        print(f"[DRY RUN] Would create/verify repository: '{repo_id}'")

    print(f"\nUploading {len(checkpoints)} checkpoint(s) ...\n")

    success, skipped, failed = 0, 0, 0

    for entry in checkpoints:
        local_path = resolve_path(entry["local_path"])
        repo_path = entry["repo_path"]
        description = entry.get("description", "")

        label = f"  {local_path} -> {repo_id}/{repo_path}"
        if description:
            label += f"\n    ({description})"

        if not local_path.exists():
            print(f"SKIP {label}\n    Path does not exist: {local_path}\n")
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY RUN] Would upload: {label}\n")
            continue

        print(f"Uploading: {label}")
        try:
            if local_path.is_dir():
                api.upload_folder(
                    folder_path=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=f"{commit_message} [{repo_path}]",
                )
            else:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=f"{commit_message} [{repo_path}]",
                )
            print(f"  OK\n")
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    if dry_run:
        valid = sum(1 for e in checkpoints if resolve_path(e["local_path"]).exists())
        print(f"[DRY RUN] {valid}/{len(checkpoints)} paths exist and would be uploaded.")
    else:
        print(f"Done. Success: {success}  Skipped: {skipped}  Failed: {failed}")
        if success:
            print(f"View at: https://huggingface.co/{repo_id}")
    if failed:
        sys.exit(1)


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

    config = load_config(str(config_path))
    upload_checkpoints(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
