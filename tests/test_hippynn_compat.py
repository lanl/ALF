"""ALF finds hippynn's checkpoint module under its old name (``serialization``) or its new one (``checkpoints``)."""

import sys
from types import ModuleType

import pytest

from alframework.ml_interfaces import _hippynn_compat
from alframework.ml_interfaces._hippynn_compat import CHECKPOINT_MODULES, checkpoints


def _hide(monkeypatch, name):
    """Make ``import name`` fail the way a hippynn without that module does."""
    real = _hippynn_compat.importlib.import_module

    def import_module(target):
        if target == name:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(target)

    monkeypatch.setattr(_hippynn_compat.importlib, "import_module", import_module)


@pytest.mark.parametrize("present", CHECKPOINT_MODULES)
def test_either_name_is_found(monkeypatch, present):
    fake = ModuleType(present)
    for name in CHECKPOINT_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, present, fake)
    if present != CHECKPOINT_MODULES[0]:
        _hide(monkeypatch, CHECKPOINT_MODULES[0])
    assert checkpoints() is fake


def test_new_name_wins_when_both_exist(monkeypatch):
    new, old = (ModuleType(name) for name in CHECKPOINT_MODULES)
    monkeypatch.setitem(sys.modules, CHECKPOINT_MODULES[0], new)
    monkeypatch.setitem(sys.modules, CHECKPOINT_MODULES[1], old)
    assert checkpoints() is new


def test_an_error_inside_the_module_is_not_hidden(monkeypatch):
    def import_module(target):
        raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")

    monkeypatch.setattr(_hippynn_compat.importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        checkpoints()


def test_neither_name_raises(monkeypatch):
    def import_module(target):
        raise ModuleNotFoundError(f"No module named {target!r}", name=target)

    monkeypatch.setattr(_hippynn_compat.importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="provides none"):
        checkpoints()


def test_installed_hippynn_has_one_of_them():
    pytest.importorskip("hippynn")
    module = checkpoints()
    assert module.__name__ in CHECKPOINT_MODULES
    for function in ("load_checkpoint", "load_saved_tensors", "create_state", "create_structure_file"):
        assert callable(getattr(module, function))
