from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict


def _safe_import_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except Exception:
        return None


def _find_git_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _git_info(start: Path) -> Dict[str, Any]:
    root = _find_git_root(start)
    if root is None:
        return {}

    info: Dict[str, Any] = {}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        info["commit"] = commit
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["is_dirty"] = bool(status.strip())
    except Exception:
        pass

    if root is not None:
        info["root"] = str(root)

    return info


def collect_run_metadata(
    *,
    config_path: Path,
    cfg: Dict[str, Any],
    run_dir: Path,
    argv: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Collect lightweight metadata about the current run.
    """

    py_version = sys.version.replace("\n", " ")

    metadata: Dict[str, Any] = {
        "command": " ".join(argv or sys.argv),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "python": {
            "version": py_version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "torch": _safe_import_version("torch"),
            "pytorch_lightning": _safe_import_version("pytorch-lightning"),
            "ecgen": _safe_import_version("ecgen"),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": _git_info(config_path.resolve()),
    }

    # Store a shallow copy of the experiment section for quick reference
    exp_cfg = cfg.get("experiment", {})
    metadata["experiment"] = {
        "name": exp_cfg.get("name"),
        "seed": exp_cfg.get("seed"),
    }

    return metadata

