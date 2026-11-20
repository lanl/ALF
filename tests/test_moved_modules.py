"""Modules that moved still import from their old paths, with a warning naming the new one.

Configurations, scripts and pickled objects may name a path that was on ``development``; each resolves to the
module at its new path. Several tests run in a fresh interpreter, because a module warns only on its first
import, and modules imported by earlier tests would hide it.
"""

import importlib
import pickle
import subprocess
import sys
import warnings

import pytest
from ase import Atoms

from alframework.tools.config import load_module_from_string
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
     "orca_double_calculator_task"),
    ("alframework.qm_interfaces.orca5_interface", "alframework.qm_interfaces.legacy.orca5_interface",
     "orcaGenerator"),
    ("alframework.qm_interfaces.qchem_DFT_interface", "alframework.qm_interfaces.legacy.qchem_DFT_interface",
     "qchem_dft_calculator_task"),
    ("alframework.qm_interfaces.qchem_DFT_interface", "alframework.qm_interfaces.legacy.qchem_DFT_interface",
     "qchemGenerator"),
    ("alframework.qm_interfaces.vaspase_interface", "alframework.qm_interfaces.legacy.vaspase_interface",
     "VASPGenerator"),
    ("alframework.samplers.packed_engine", "alframework.samplers.engines.packed", "PackedState"),
    ("alframework.samplers.torchsim_engine", "alframework.samplers.engines.torchsim", "TorchSimEngine"),
    ("alframework.samplers.alchemi_engine", "alframework.samplers.engines.alchemi", "AlchemiEngine"),
    ("alframework.samplers.geometry", "alframework.tools.geometry", "minimum_distance"),
    ("alframework.tools.excited_state_tools", "alframework.tools.excited_state_schema",
     "derive_state_property_table"),
    ("alframework.tools.failed_tasks", "alframework.tools.task_failure_cleanup", "releases_memory_on_failure"),
    ("alframework.tools.node_setup", "alframework.tools.worker_setup", "worker_setup"),
    ("alframework.tools.labeling", "alframework.tools.labeling_coordinator", "LabelingCoordinator"),
    ("alframework.tools.molecules_class", "alframework.tools.molecules", "MoleculesObject"),
    ("alframework.ml_interfaces.hippynn_interface", "alframework.ml_interfaces.hippynn.interface",
     "train_HIPPYNN_ensemble_task"),
    ("alframework.ml_interfaces.hippynn_interface", "alframework.ml_interfaces.hippynn.interface",
     "load_hippynn_batched_ensemble"),
    ("alframework.ml_interfaces.hippynn_lightning", "alframework.ml_interfaces.hippynn.lightning",
     "train_data_parallel"),
    ("alframework.ml_interfaces._hippynn_compat", "alframework.ml_interfaces.hippynn.compat", "checkpoints"),
]
# Imports PyTorch and HIPPYNN by design, so it is checked only where they are installed.
NEEDS_HIPPYNN = [("alframework.ml_interfaces.hippynn_final", "alframework.ml_interfaces.hippynn.final_evaluation",
                  "FinalEvaluation")]
# Modules that moved whole, rather than being split.
WHOLE = sorted({(old, new) for old, new, _ in MOVED if not old.endswith("ase_calculator_interface")})
PACKAGES = ["alframework.ml_interfaces", "alframework.qm_interfaces", "alframework.samplers", "alframework.tools"]


def run_fresh(code):
    """Run ``code`` in a new interpreter, which fails the test if ``code`` raises."""
    subprocess.run([sys.executable, "-B", "-W", "always::FutureWarning", "-c", code], check=True, timeout=60)


def warning_check(old, new, name):
    """Code that loads ``old.name`` as a configuration would, and checks the object and the warning."""
    return f"""
import warnings
from alframework.tools.config import load_module_from_string
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    moved = load_module_from_string("{old}.{name}")
assert moved is load_module_from_string("{new}.{name}")
assert any(issubclass(w.category, FutureWarning) and "{old}" in str(w.message) and "{new}" in str(w.message)
           for w in caught), caught
"""


@pytest.mark.parametrize("old,new,name", MOVED)
def test_a_configured_old_path_gives_the_same_object_with_a_warning(old, new, name):
    run_fresh(warning_check(old, new, name))


def test_an_old_path_still_warns_when_the_new_module_was_imported_first():
    old, new, name = "alframework.tools.molecules_class", "alframework.tools.molecules", "MoleculesObject"
    run_fresh(f"import {new}\n" + warning_check(old, new, name))


@pytest.mark.parametrize("old,new,name", MOVED)
def test_an_object_pickled_by_its_old_path_still_loads(old, new, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        loaded = pickle.loads(f"c{old}\n{name}\n.".encode())
    assert loaded is load_module_from_string(f"{new}.{name}")


@pytest.mark.parametrize("old,new", WHOLE)
def test_an_old_module_path_is_the_new_module_itself(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        moved = importlib.import_module(old)
    assert moved is importlib.import_module(new)
    # The real module keeps its own name and spec, whichever path imported it first.
    assert moved.__name__ == moved.__spec__.name == new


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


def test_old_paths_import_no_gpu_libraries():
    # The Lightning trainer imports PyTorch Lightning when it is installed, at its old path as at its new one.
    imports = "\n".join(f"import {old}" for old, _ in WHOLE if old != "alframework.ml_interfaces.hippynn_lightning")
    run_fresh(f"""
import sys
{imports}
assert not ({{"torch", "torch_sim", "nvalchemi", "warp", "cupy", "hippynn"}} & sys.modules.keys())
""")


@pytest.mark.parametrize("name", ["generic", "orca", "qchem", "vasp", "workflow"])
def test_ase_modules_import_without_torch_or_hippynn(name):
    run_fresh(f"""
import sys
import alframework.qm_interfaces.ase.{name}
assert "torch" not in sys.modules and "hippynn" not in sys.modules
""")


def test_batched_mlmd_still_gives_the_engine_names_it_used_to_define():
    from alframework.samplers import batched_mlmd
    from alframework.samplers.engines import ase, factory

    for name in ("ASEEngine", "OBSERVED_PROPERTIES", "_PredictedMember", "_StateMeanCalculator"):
        assert getattr(batched_mlmd, name) is getattr(ase, name)
    assert batched_mlmd.make_engine is factory.make_engine


@pytest.mark.parametrize("old,new,name", NEEDS_HIPPYNN)
def test_the_streaming_final_evaluation_keeps_its_old_path(old, new, name):
    pytest.importorskip("hippynn")
    run_fresh(warning_check(old, new, name))


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
