import pytest

from ecgen.registry import CALLBACKS, DATASETS, FAMILIES, Registry, load_builtins


class Thing:
    def __init__(self, a=1, b=2):
        self.a, self.b = a, b


def test_register_get_build():
    reg = Registry("thing")
    reg.register("thing")(Thing)
    assert reg.get("thing") is Thing
    obj = reg.build({"name": "thing", "params": {"a": 5}}, b=7)
    assert (obj.a, obj.b) == (5, 7)
    assert "thing" in reg and list(reg) == ["thing"]


def test_target_import_path():
    reg = Registry("thing")
    assert reg.resolve({"target": "collections.OrderedDict"}).__name__ == "OrderedDict"


def test_errors():
    reg = Registry("thing")
    reg.add("thing", Thing)
    with pytest.raises(KeyError, match="already registered"):
        reg.add("thing", dict)
    reg.add("thing", Thing)  # re-registering the same object is fine (module reloads)
    with pytest.raises(KeyError, match="Unknown thing 'nope'. Registered: thing"):
        reg.get("nope")
    with pytest.raises(ValueError, match="both"):
        reg.resolve({"name": "thing", "target": "x.y"})
    with pytest.raises(ValueError, match="needs 'name' or 'target'"):
        reg.resolve({"params": {}})


def test_builtins_registered():
    load_builtins()
    assert {"vae", "vqvae", "vqvae_prior", "pulse2pulse"} <= set(FAMILIES)
    assert {"mimic_iv_ecg", "synthetic"} <= set(DATASETS)
    assert {
        "vae_visualization",
        "ecg_visualization",
        "generated_samples",
        "early_stopping",
    } <= set(CALLBACKS)
