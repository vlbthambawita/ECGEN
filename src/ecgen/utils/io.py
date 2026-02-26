from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> Any:
    """
    Load a YAML file and return the parsed object.
    """

    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Any, path: str | Path) -> None:
    """
    Serialize an object to YAML.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(obj, f)
