"""Modules that moved after a release still import from their old paths, with a warning naming the new one.

Configurations, scripts and pickled objects (molecules, HIPPYNN checkpoints) may name a path that was on ALF's
released ``main``; each resolves to the module at its new path. The import checks run in one fresh interpreter,
because a module warns only on its first import, and modules imported by earlier tests would hide it.
"""

import importlib
import pickle
import subprocess
import sys
import warnings

import pytest
from ase import Atoms

from alframework.tools.config import load_module_from_config, load_module_from_string
from alframework.tools.molecules import MoleculesObject

# (old module, new module, a name each must give). A module that was split appears once per new home.
MOVED = [
    ("alframework.qm_interfaces.ase_calculator_interface", "alframework.qm_interfaces.ase.generic",
     "ase_calculator_task"),
    ("alframework.qm_interfaces.ase_calculator_interface", "alframework.qm_interfaces.ase.generic",
     "fresh_scratch_directory"),
    ("alframework.qm_interfaces.ase_calculator_interface", "alframework.qm_interfaces.ase.vasp",
     "VASP_ase_calculator_task"),
    ("alframework.qm_interfaces.orca5_interface", "alframework.qm_interfaces.legacy.orca5_interface",
     "orca_calculator_task"),
    ("alframework.qm_interfaces.orca5_interface", "alframework.qm_interfaces.legacy.orca5_interface",
     "orcaGenerator"),
    ("alframework.qm_interfaces.qchem_DFT_interface", "alframework.qm_interfaces.legacy.qchem_DFT_interface",
     "qchem_dft_calculator_task"),
    ("alframework.qm_interfaces.vaspase_interface", "alframework.qm_interfaces.legacy.vaspase_interface",
     "VASPGenerator"),
    ("alframework.tools.molecules_class", "alframework.tools.molecules", "MoleculesObject"),
    ("alframework.ml_interfaces.hippynn_interface", "alframework.ml_interfaces.hippynn.interface",
     "train_HIPPYNN_ensemble_task"),
    ("alframework.ml_interfaces.hippynn_interface", "alframework.ml_interfaces.hippynn.interface",
     "load_hippynn_batched_ensemble"),
]
# Paths that existed only on development, between releases, and no longer import.
RETIRED = [
    "alframework.samplers.packed_engine", "alframework.samplers.torchsim_engine", "alframework.samplers.alchemi_engine",
    "alframework.samplers.geometry", "alframework.tools.excited_state_tools", "alframework.tools.failed_tasks",
    "alframework.tools.node_setup", "alframework.tools.labeling", "alframework.ml_interfaces.hippynn_lightning",
    "alframework.ml_interfaces._hippynn_compat", "alframework.ml_interfaces.hippynn_final",
]
PACKAGES = ["alframework.ml_interfaces", "alframework.qm_interfaces", "alframework.samplers", "alframework.tools"]
# The names tools.tools gave before its helpers moved into modules of their own: (module, name).
TOOLS_TOOLS = [("annealing", "annealing_schedule"), ("config", "build_input_dict"), ("data_shards", "store_current_data"),
               ("geometry", "random_rotation_matrix"), ("label_validation", "system_checker"),
               ("parsl_tasks", "parsl_task_queue"), ("legacy_ani", "build_ANI_info"), ("restarts", "write_status"),
               ("label_sources", "write_label_sources")]


def test_every_old_path_gives_the_new_object_once_warned_and_new_paths_import_quietly():
    """One fresh interpreter: the new modules import first, without a warning; each old path then gives the same
    object as a configuration loads it, warns once naming both paths, and is the new module itself."""
    code = f"""
import importlib, warnings
from alframework.tools.config import load_module_from_string
moved = {MOVED!r}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _, new, _ in moved:
        importlib.import_module(new)
assert not [w for w in caught if issubclass(w.category, FutureWarning)], caught
warned = set()
for old, new, name in moved:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = load_module_from_string(old + "." + name)
    assert found is load_module_from_string(new + "." + name), (old, name)
    if not old.endswith("ase_calculator_interface"):
        assert importlib.import_module(old) is importlib.import_module(new), old
        assert importlib.import_module(old).__spec__.name == new, old
    messages = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    if old not in warned or old.endswith("ase_calculator_interface"):
        assert any(old in m and new in m for m in messages), (old, name, messages)
    warned.add(old)
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("old", RETIRED)
def test_a_retired_development_path_no_longer_imports(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)


def test_a_molecule_pickled_under_molecules_class_still_loads():
    molecule = MoleculesObject(Atoms("H2"), "legacy-id")
    molecule.update_metadata({"stage": "bootstrap"})
    current = pickle.dumps(molecule, protocol=0)
    legacy = current.replace(b"alframework.tools.molecules\n", b"alframework.tools.molecules_class\n")
    assert legacy != current
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        restored = pickle.loads(legacy)
    assert type(restored) is MoleculesObject
    assert restored.get_moleculeid() == "legacy-id"
    assert restored.get_metadata()["stage"] == "bootstrap"
    assert restored.get_atoms().get_chemical_formula() == "H2"


def test_a_name_a_split_module_never_had_is_still_an_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_task'"):
        load_module_from_string("alframework.qm_interfaces.ase_calculator_interface.not_a_task")


@pytest.mark.parametrize("package", PACKAGES)
def test_a_module_that_never_existed_is_still_an_error(package):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{package}.not_a_module")


@pytest.mark.parametrize("module, name", TOOLS_TOOLS)
def test_tools_tools_still_gives_the_helpers_it_defined(module, name):
    old = "alframework.tools.tools"
    canonical = load_module_from_string(f"alframework.tools.{module}.{name}")
    assert load_module_from_config({"helper": f"{old}.{name}"}, "helper") is canonical
    assert pickle.loads(f"c{old}\n{name}\n.".encode()) is canonical


def test_a_checkpoint_naming_the_old_database_class_still_loads():
    """HIPPYNN checkpoints pickle the training database's class by the module it was defined in."""
    pytest.importorskip("hippynn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        loaded = pickle.loads(b"calframework.ml_interfaces.hippynn_interface\nALFPyAniDirectoryDB\n.")
    assert loaded.__name__ == "ALFPyAniDirectoryDB"


def test_configurations_name_the_hippynn_tasks_from_the_package():
    from alframework.ml_interfaces import hippynn
    from alframework.ml_interfaces.hippynn import interface
    for name in hippynn.__all__:
        assert load_module_from_string(f"alframework.ml_interfaces.hippynn.{name}") is getattr(interface, name)
