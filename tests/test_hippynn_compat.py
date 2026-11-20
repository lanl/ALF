"""ALF finds hippynn's checkpoint module under its old name (``serialization``) or its new one (``checkpoints``)."""

import sys
from types import ModuleType

import pytest

from alframework.ml_interfaces.hippynn import compat as _hippynn_compat
from alframework.ml_interfaces.hippynn.compat import CHECKPOINT_MODULES, checkpoints


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


def test_node_and_energy_names_work_with_both_hippynn_spellings():
    """Current hippynn has unique_node_from_name and system_energy; release 0.1.3 only the older names."""
    from alframework.ml_interfaces.hippynn.compat import node_from_name, system_energy

    class Current:
        output_names = ("system_energy", "atom_energies")
        system_energy = "current"

    class Released:
        _output_names = ("mol_energy", "atom_energies")
        mol_energy = "released"

    class CurrentGraph:
        def unique_node_from_name(self, name):
            return ("unique", name)

    class ReleasedGraph:
        def node_from_name(self, name):
            return ("released", name)

    assert system_energy(Current()) == "current" and system_energy(Released()) == "released"
    assert node_from_name(CurrentGraph(), "energy") == ("unique", "energy")
    assert node_from_name(ReleasedGraph(), "energy") == ("released", "energy")
