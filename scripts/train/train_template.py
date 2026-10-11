#!/usr/bin/env python3
"""
Training now goes through one runner for every model family:

    ecgen train configs/experiments/<experiment>.yaml [dotted.key=value ...]
    python -m ecgen train ...          # same, without installing the entry point

To train a new model:

1. Write a ``GenerativeModule`` subclass (see ``ecgen.families.base``) with a config
   dataclass and the ``__init__(self, config=None, **kwargs)`` / ``resolve_config``
   pattern, and register it::

       from ecgen.registry import FAMILIES

       @FAMILIES.register("my_model")
       class MyModel(GenerativeModule): ...

   Code outside ``ecgen`` can skip registration and use ``model: {target: my_pkg.MyModel}``.
2. Copy a config from ``configs/experiments/`` and set ``model.name`` (or ``target``)
   and ``model.params``.
3. ``ecgen train <your config>``.

See docs/training/runner.md.
"""

from ecgen.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
