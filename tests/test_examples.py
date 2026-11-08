"""Every tracked example in examples/ loads with the current validators and names tasks that exist.

Submits no jobs and needs no QM programs. NeuroChem examples are imported against a fake backend.
"""

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from alframework.qm_interfaces.gpu4pyscf_interface import validate_gpu4pyscf_config
from alframework.samplers.acquisition import acquisition_settings
from alframework.tools.batching import batching_settings
from alframework.tools.labeling import labeling_batch_settings
from alframework.tools.sampler_coordinator import driver_settings
from tests.helpers.fakes import import_neurochem_with_fake_backend

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "AcAc_excited_pyseqm",
    "CPP_excited_gpu4pyscf",
    "IL",
    "UO2",
    "fine_tuning",
    "mace_fine_tuning",
    "uma_fine_tuning",
    "molten_salt",
    "reactive_sampling",
    "seeded_active_learning",
    "simple_water",
    "simple_water_multi_builder",
)
MASTERS = sorted(p for name in FAMILIES for p in (ROOT / "examples" / name).glob("master*.json"))
TASKS = ("builder_task", "sampler_task", "QM_task", "ML_task")


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", MASTERS, ids=lambda p: str(p.relative_to(ROOT / "examples")))
def test_example_configs_are_valid(path):
    master = read(path)
    configs = {role: read(path.parent / master[role + "_config_path"]) for role in ("builder", "sampler", "QM", "ML")}
    for config in configs.values():
        assert isinstance(config, dict)
    sampler = configs["sampler"]
    acquisition_settings(sampler)
    driver_settings(sampler)
    batching_settings(sampler)
    labeling_batch_settings(configs["QM"])
    for field in TASKS:
        module, name = master[field].rsplit(".", 1)
        source = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            source = ROOT.joinpath(*module.split("."), "__init__.py")  # A package.
        if not source.is_file():
            source = path.parent.joinpath(*module.split(".")).with_suffix(".py")
        assert source.is_file(), master[field]
        tree = ast.parse(source.read_text())
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
            for node in tree.body
        ), master[field]
    assert all(len(spec) == 3 for spec in master["properties_list"].values())


@pytest.mark.parametrize("path", MASTERS, ids=lambda p: str(p.relative_to(ROOT / "examples")))
def test_example_tasks_can_be_imported(path, monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    # Several examples have their own custom_modules.py.
    monkeypatch.delitem(sys.modules, "custom_modules", raising=False)
    for field in TASKS:
        module, name = read(path)[field].rsplit(".", 1)
        if module == "alframework.ml_interfaces.neurochem_interface":
            loaded = import_neurochem_with_fake_backend(monkeypatch)
        else:
            loaded = importlib.import_module(module)
        assert callable(getattr(loaded, name))


@pytest.mark.parametrize("name", ["mlmd_config_torchsim.json", "mlmd_config_alchemi.json"])
def test_packed_water_configs_are_valid(name):
    config = read(ROOT / "examples/simple_water" / name)
    assert driver_settings(config)["strategy"] == "batched"
    batching_settings(config)
    acquisition_settings(config)


@pytest.mark.parametrize("name", ["qm_config_teacher_mace.json", "qm_config_teacher_uma.json"])
def test_teacher_water_configs_are_valid(name, tmp_path):
    from alframework.qm_interfaces.teacher_interface import validate_teacher_config

    qm_config = read(ROOT / "examples/simple_water" / name)
    # The example names a checkpoint you download; stand in a file so the rest can be checked.
    checkpoint = tmp_path / qm_config["teacher"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    master = read(ROOT / "examples/simple_water/master_config_teacher.json")
    validate_teacher_config(qm_config, dict(master, master_directory=str(tmp_path)))
    labeling_batch_settings(qm_config)


@pytest.mark.parametrize("master", ["simple_water/master_config.json", "simple_water/master_config_teacher.json",
                                    "mace_fine_tuning/master_config.json", "uma_fine_tuning/master_config.json"])
def test_cluster_examples_build_non_periodic_structures(master):
    # ORCA ignores the cell, so a teacher or a fine-tuned model must see the same cluster, not a periodic box.
    path = ROOT / "examples" / master
    builder = read(path.parent / read(path)["builder_config_path"])
    assert builder["periodic"] is False


@pytest.mark.parametrize("name", ["QM_config.json", "QM_config_dcm.json"])
def test_cpp_gpu4pyscf_configs_are_valid(name):
    validate_gpu4pyscf_config(read(ROOT / "examples/CPP_excited_gpu4pyscf" / name))


def test_no_old_state_names_remain():
    """State-aware code names states E#/F#; the old sE# names were removed, with no compatibility."""
    import re

    old = re.compile(r"(?<![A-Za-z_])sE[0-9#{]")
    found = [str(path.relative_to(ROOT)) for folder in ("alframework", "examples", "docs/source")
             for path in (ROOT / folder).rglob("*") if path.suffix in {".py", ".json", ".rst", ".md", ".txt"}
             and old.search(path.read_text(errors="ignore"))]
    assert found == []
