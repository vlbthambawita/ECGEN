"""
Name → class registries used to build components from YAML.

A component spec is a mapping with either a registry ``name`` or an import
path ``target``, plus optional ``params``::

    {"name": "vae", "params": {...}}
    {"target": "my_pkg.models.MyModel", "params": {...}}

``target`` lets user code outside the library plug in without registering.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Generic, Iterator, Mapping, Optional, TypeVar

T = TypeVar("T")


def import_from_string(path: str) -> Any:
    """Import ``package.module.attr`` and return ``attr``."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class Registry(Generic[T]):
    """
    A named collection of classes (or factories) of one kind.

    Example::

        FAMILIES = Registry("family")

        @FAMILIES.register("vae")
        class VAELightning(...): ...

        FAMILIES.get("vae")  # -> VAELightning
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self.add(name, obj)
            return obj

        return decorator

    def add(self, name: str, obj: T) -> None:
        existing = self._items.get(name)
        if existing is not None and existing is not obj:
            raise KeyError(
                f"{self.kind} {name!r} is already registered to {existing!r}"
            )
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Registered: {known}"
            ) from None

    def resolve(self, spec: Mapping[str, Any]) -> T:
        """Return the class named by a spec's ``name`` or ``target``."""
        if "target" in spec and "name" in spec:
            raise ValueError(
                f"{self.kind} spec has both 'name' and 'target': {dict(spec)}"
            )
        if "target" in spec:
            return import_from_string(spec["target"])
        if "name" in spec:
            return self.get(spec["name"])
        raise ValueError(f"{self.kind} spec needs 'name' or 'target': {dict(spec)}")

    def build(self, spec: Mapping[str, Any], **extra: Any) -> Any:
        """Instantiate ``resolve(spec)(**spec['params'], **extra)``."""
        params: Optional[Mapping[str, Any]] = spec.get("params")
        return self.resolve(spec)(**dict(params or {}), **extra)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {sorted(self._items)})"


DATASETS: Registry[Any] = Registry("dataset")
FAMILIES: Registry[Any] = Registry("family")
CALLBACKS: Registry[Any] = Registry("callback")
CONDITION_ENCODERS: Registry[Any] = Registry("condition encoder")
MASKS: Registry[Any] = Registry("mask generator")
BACKBONES: Registry[Any] = Registry("backbone")
PROCESSES: Registry[Any] = Registry("process")


def load_builtins() -> None:
    """Import the built-in modules so their ``@register`` decorators run."""
    import ecgen.conditioning  # noqa: F401
    import ecgen.data.datasets  # noqa: F401
    import ecgen.data.masks  # noqa: F401
    import ecgen.families  # noqa: F401
    import ecgen.nn.backbones  # noqa: F401
    import ecgen.processes  # noqa: F401
    import ecgen.training.callbacks  # noqa: F401
