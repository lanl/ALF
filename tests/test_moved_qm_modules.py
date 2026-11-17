"""QM modules that moved still import from their old paths, with a warning naming the new one."""

import importlib
import pickle
import subprocess
import sys
import warnings

import pytest

from alframework.tools.tools import load_module_from_string

PREFIX = "alframework.qm_interfaces."

# (old module, new module, a name each must give)
MOVED = [
    ("ase_calculator_interface", "ase.generic", "ase_calculator_task"),
    ("ase_calculator_interface", "ase.generic", "fresh_scratch_directory"),
    ("ase_calculator_interface", "ase.vasp", "VASP_ase_calculator_task"),
    ("orca5_interface", "legacy.orca5_interface", "orca_calculator_task"),
    ("orca5_interface", "legacy.orca5_interface", "orca_double_calculator_task"),
    ("orca5_interface", "legacy.orca5_interface", "orcaGenerator"),
    ("qchem_DFT_interface", "legacy.qchem_DFT_interface", "qchem_dft_calculator_task"),
    ("qchem_DFT_interface", "legacy.qchem_DFT_interface", "qchemGenerator"),
    ("vaspase_interface", "legacy.vaspase_interface", "VASPGenerator"),
]


def run_fresh(code):
    """Run ``code`` in a new interpreter, so modules imported by earlier tests don't hide the first import."""
    subprocess.run([sys.executable, "-B", "-W", "always::FutureWarning", "-c", code], check=True)


@pytest.mark.parametrize("old,new,name", MOVED)
def test_a_configured_old_path_gives_the_same_object_with_a_warning(old, new, name):
    run_fresh(f"""
import warnings
from alframework.tools.tools import load_module_from_string
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    moved = load_module_from_string("{PREFIX}{old}.{name}")
assert moved is load_module_from_string("{PREFIX}{new}.{name}")
assert any(issubclass(w.category, FutureWarning) and "{PREFIX}{new}" in str(w.message) for w in caught), caught
""")


@pytest.mark.parametrize("old,new,name", MOVED)
def test_an_object_pickled_by_its_old_path_still_loads(old, new, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # Checked above; a module warns on its first import only.
        loaded = pickle.loads(f"c{PREFIX}{old}\n{name}\n.".encode())
    assert loaded is load_module_from_string(f"{PREFIX}{new}.{name}")


def test_a_name_the_old_module_never_had_is_still_an_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_task'"):
        load_module_from_string(f"{PREFIX}ase_calculator_interface.not_a_task")


def test_a_module_that_never_existed_is_still_an_error():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{PREFIX}not_a_module")


@pytest.mark.parametrize("name", ["generic", "orca", "qchem", "vasp", "workflow"])
def test_ase_modules_import_without_torch_or_hippynn(name):
    run_fresh(f"""
import sys
import alframework.qm_interfaces.ase.{name}
assert "torch" not in sys.modules and "hippynn" not in sys.modules
""")
