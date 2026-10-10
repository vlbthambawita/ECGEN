"""
ecgen.f.upload_hf — Upload model checkpoints to a HuggingFace repository.

Programmatic API:

    from ecgen.f import upload_hf, upload_hf_single

    # Upload using a config dict or a path to a YAML config file
    upload_hf("configs/upload_checkpoints.yaml")
    upload_hf({"hf_repo_id": "user/ECGEN", "checkpoints": [...]})

    # Upload a single checkpoint directly
    upload_hf_single(
        local_path="/runs/vae/ptbxl/best.ckpt",
        repo_path="vae/ptbxl/best.ckpt",
        repo_id="user/ECGEN",
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, TypedDict, Union

import yaml

# Module-level lazy import so @patch("ecgen.f.upload_hf.HfApi") works in tests
# and dry_run mode doesn't require huggingface_hub to be installed.
try:
    from huggingface_hub import HfApi, create_repo
    _HF_AVAILABLE = True
except ImportError:
    HfApi = None       # type: ignore[assignment,misc]
    create_repo = None  # type: ignore[assignment]
    _HF_AVAILABLE = False


def _load_token(env_file: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Load HF_TOKEN from a .env file or the environment.

    Resolution order:
      1. ``env_file`` argument (explicit path)
      2. ``HF_ENV_FILE`` environment variable (path to a .env file)
      3. ``.env`` in the current working directory
      4. ``HF_TOKEN`` already set in the environment (no file needed)

    Returns the token string, or None if not found.
    """
    # Determine which .env file to try
    candidate: Optional[Path] = None
    if env_file is not None:
        candidate = Path(env_file)
    elif os.environ.get("HF_ENV_FILE"):
        candidate = Path(os.environ["HF_ENV_FILE"])
    else:
        default = Path.cwd() / ".env"
        if default.exists():
            candidate = default

    if candidate is not None:
        try:
            from dotenv import load_dotenv
            load_dotenv(candidate, override=False)  # don't overwrite already-set vars
        except ImportError:
            # dotenv not installed — fall back to manual parse
            _parse_env_file(candidate)

    return os.environ.get("HF_TOKEN")


def _parse_env_file(path: Path) -> None:
    """Minimal .env parser used when python-dotenv is not installed."""
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


class CheckpointEntry(TypedDict, total=False):
    local_path: str   # required
    repo_path: str    # required
    description: str  # optional


def _load_config(config_path: Union[str, Path]) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_path(path: str) -> Path:
    """Resolve a path: absolute as-is, relative resolved from CWD."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (Path.cwd() / p).resolve()


def upload_hf(
    config: Union[str, Path, dict],
    *,
    dry_run: bool = False,
    env_file: Optional[Union[str, Path]] = None,
) -> dict:
    """Upload model checkpoints to a HuggingFace repository.

    Parameters
    ----------
    config:
        Path (str or Path) to a YAML config file, or a plain dict with the same structure.
        Required keys:
            hf_repo_id (str)         — e.g. "username/ECGEN"
            checkpoints (list[dict]) — each entry must have ``local_path`` and ``repo_path``
        Optional keys:
            hf_repo_type (str)  — "model" (default), "dataset", or "space"
            commit_message (str)
            private (bool)      — default False
            env_file (str)      — path to .env file containing HF_TOKEN (can also be set
                                  here instead of via the ``env_file`` argument)
    dry_run:
        Print what would be uploaded without actually uploading.
        Does not require ``huggingface_hub`` to be installed.
    env_file:
        Path to a .env file that contains ``HF_TOKEN=hf_...``.
        Falls back to ``HF_ENV_FILE`` env var, then ``.env`` in CWD, then
        ``HF_TOKEN`` already present in the environment.

    Returns
    -------
    dict with keys ``success``, ``skipped``, ``failed`` (counts).

    Raises
    ------
    ImportError  if ``huggingface_hub`` is not installed and ``dry_run=False``.
    ValueError   if required config keys are missing.
    RuntimeError if the HF repository cannot be created or any upload fails.
    """
    if isinstance(config, (str, Path)):
        config = _load_config(config)

    # Token resolution: argument > config key > .env file > environment
    resolved_env_file = env_file or config.get("env_file")
    token: Optional[str] = _load_token(resolved_env_file)

    repo_id: str = config.get("hf_repo_id", "")
    if not repo_id:
        raise ValueError("Config must contain 'hf_repo_id'.")

    checkpoints: List[CheckpointEntry] = config.get("checkpoints", [])
    if not checkpoints:
        print("No checkpoints defined in config. Nothing to upload.")
        return {"success": 0, "skipped": 0, "failed": 0}

    # Defer HF import check until we actually need to upload
    if not dry_run and not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for upload_hf. "
            "Install it with: pip install 'ecgen[hf]'  or  pip install huggingface_hub"
        )

    repo_type: str = config.get("hf_repo_type", "model")
    commit_message: str = config.get("commit_message", "Upload model checkpoints")
    private: bool = config.get("private", False)

    if "/" not in repo_id:
        print(
            f"WARNING: repo_id '{repo_id}' has no namespace. "
            "Set it to 'your_username/ECGEN' to be explicit."
        )

    if not dry_run:
        api = HfApi(token=token)
        print(f"Creating/verifying repository '{repo_id}' ({repo_type}, private={private}) ...")
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True,
                        token=token)
            print(f"  Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create HuggingFace repository '{repo_id}': {e}"
            ) from e
    else:
        api = None
        print(f"[DRY RUN] Would create/verify repository: '{repo_id}'")

    print(f"\nUploading {len(checkpoints)} checkpoint(s) ...\n")

    success = skipped = failed = 0

    for entry in checkpoints:
        local_path = _resolve_path(entry["local_path"])
        repo_path: str = entry["repo_path"]
        description: str = entry.get("description", "")

        label = f"  {local_path}  ->  {repo_id}/{repo_path}"
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
            print("  OK\n")
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    if dry_run:
        valid = sum(1 for e in checkpoints if _resolve_path(e["local_path"]).exists())
        print(f"[DRY RUN] {valid}/{len(checkpoints)} paths exist and would be uploaded.")
    else:
        print(f"Done.  Success: {success}  Skipped: {skipped}  Failed: {failed}")
        if success:
            print(f"View at: https://huggingface.co/{repo_id}")

    if failed:
        raise RuntimeError(f"{failed} checkpoint(s) failed to upload. See output above.")

    return {"success": success, "skipped": skipped, "failed": failed}


def upload_hf_single(
    local_path: Union[str, Path],
    repo_path: str,
    repo_id: str,
    *,
    repo_type: str = "model",
    commit_message: str = "Upload checkpoint",
    private: bool = False,
    dry_run: bool = False,
    env_file: Optional[Union[str, Path]] = None,
) -> dict:
    """Upload a single checkpoint file or directory to HuggingFace.

    Convenience wrapper around :func:`upload_hf` for one-liner use.

    Parameters
    ----------
    local_path:
        Path to the local file or directory to upload.
    repo_path:
        Destination path inside the HF repository (e.g. ``"vae/ptbxl/best.ckpt"``).
    repo_id:
        HuggingFace repository ID, e.g. ``"username/ECGEN"``.
    repo_type:
        ``"model"``, ``"dataset"``, or ``"space"``. Default ``"model"``.
    commit_message:
        Commit message for the upload.
    private:
        Create the repo as private if it does not yet exist.
    dry_run:
        Preview without uploading.
    env_file:
        Path to a .env file containing ``HF_TOKEN=hf_...``.

    Returns
    -------
    dict with keys ``success``, ``skipped``, ``failed``.
    """
    config = {
        "hf_repo_id": repo_id,
        "hf_repo_type": repo_type,
        "commit_message": commit_message,
        "private": private,
        "checkpoints": [
            {"local_path": str(local_path), "repo_path": repo_path}
        ],
    }
    return upload_hf(config, dry_run=dry_run, env_file=env_file)
