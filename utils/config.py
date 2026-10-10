from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")


def resolve_config(
    config_cls: Type[T],
    config: Optional[T | dict[str, Any]],
    flat_kwargs: dict[str, Any],
) -> T:
    """
    Build a config dataclass for a LightningModule constructor.

    Wrappers save their config as flat hyperparameters, so
    ``load_from_checkpoint`` calls ``__init__(**hparams)`` with one keyword per
    config field instead of ``config=...``. This accepts either form.

    Args:
        config_cls: Config dataclass type.
        config: A config instance, a dict of its fields, or None.
        flat_kwargs: Extra keyword arguments passed to ``__init__``.

    Returns:
        A ``config_cls`` instance. An explicit ``config`` takes precedence over
        ``flat_kwargs`` (Lightning passes both when ``config=`` is given to
        ``load_from_checkpoint``).
    """
    valid = {f.name for f in fields(config_cls)}
    unknown = set(flat_kwargs) - valid
    if unknown:
        raise TypeError(f"Unexpected {config_cls.__name__} fields: {sorted(unknown)}")

    if config is None:
        return config_cls(**flat_kwargs)
    if isinstance(config, dict):
        return config_cls(**config)
    return config
