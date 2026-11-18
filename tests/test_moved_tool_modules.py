"""Old tool module paths resolve in a fresh interpreter."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("canonical_first", [True, False])
def test_renamed_tool_modules(canonical_first):
    script = f"""
import importlib
import pickle
import sys
import warnings

from ase import Atoms
from alframework.tools.config import load_module_from_string

renamed = [
    ("excited_state_tools", "excited_state_schema", "derive_state_property_table"),
    ("failed_tasks", "task_failure_cleanup", "releases_memory_on_failure"),
    ("node_setup", "worker_setup", "worker_setup"),
    ("labeling", "labeling_coordinator", "LabelingCoordinator"),
    ("molecules_class", "molecules", "MoleculesObject"),
]
for old, new, name in renamed:
    old = "alframework.tools." + old
    new = "alframework.tools." + new
    if {canonical_first!r}:
        importlib.import_module(new)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = pickle.loads(f"c{{old}}\\n{{name}}\\n.".encode())
    assert any(
        issubclass(w.category, FutureWarning) and old in str(w.message) and new in str(w.message)
        for w in caught
    ), caught
    canonical = importlib.import_module(new)
    assert canonical is importlib.import_module(old)
    assert canonical.__spec__.name == new
    assert loaded is load_module_from_string(f"{{old}}.{{name}}") is getattr(canonical, name)

from alframework.tools import molecules_class
from alframework.tools.molecules import MoleculesObject
assert molecules_class is importlib.import_module("alframework.tools.molecules")

molecule = MoleculesObject(Atoms("H2"), "legacy-id")
molecule.update_metadata({{"stage": "bootstrap"}})
current_data = pickle.dumps(molecule, protocol=0)
legacy_data = current_data.replace(b"alframework.tools.molecules\\n", b"alframework.tools.molecules_class\\n")
assert legacy_data != current_data
restored = pickle.loads(legacy_data)
assert type(restored) is MoleculesObject
assert restored.get_moleculeid() == "legacy-id"
assert restored.get_metadata()["stage"] == "bootstrap"
assert restored.get_atoms().get_chemical_formula() == "H2"
assert not ({{"torch", "torch_sim", "cupy"}} & sys.modules.keys())
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=60)
