"""Every tracked example in examples/ loads with the current validators and names tasks that exist.

Submits no jobs and needs no QM programs. NeuroChem examples are imported against a fake backend.
"""

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from alframework.qm_interfaces.pyscf.gpu4pyscf import validate_gpu4pyscf_config
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
    "periodic_water",
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


@pytest.mark.parametrize("master", ["simple_water/master_config_pyscf.json", "periodic_water/master_config.json"])
def test_pyscf_example_configs_are_valid(master):
    from alframework.qm_interfaces.pyscf import validate_pyscf_config

    path = ROOT / "examples" / master
    config = read(path)
    validate_pyscf_config(read(path.parent / config["QM_config_path"]), config)
    labeling_batch_settings(read(path.parent / config["QM_config_path"]))


def test_periodic_water_builds_periodic_boxes_for_a_periodic_student():
    folder = ROOT / "examples" / "periodic_water"
    from alframework.qm_interfaces.pyscf import validate_pyscf_config

    assert read(folder / "builder_config.json")["periodic"] is True
    assert read(folder / "hippynn_config.json")["cell_key"] == "cell"
    validate_pyscf_config(read(folder / "qm_config_cpu.json"), read(folder / "master_config.json"))


@pytest.mark.parametrize("name", ["QM_config.json", "QM_config_dcm.json"])
def test_cpp_gpu4pyscf_configs_are_valid(name):
    validate_gpu4pyscf_config(read(ROOT / "examples/CPP_excited_gpu4pyscf" / name))


def test_cpp_training_benchmark_runs_and_projects_an_epoch(monkeypatch, tmp_path):
    """The CPP training benchmark, run on CPU for one step of the smallest network."""
    pytest.importorskip("hippynn")
    folder = ROOT / "examples" / "CPP_excited_gpu4pyscf"
    monkeypatch.syspath_prepend(str(folder))
    for name in ("benchmark_training", "prepare"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    benchmark = importlib.import_module("benchmark_training")
    # A CPU-compatible test config must be honored even if a different kernel was
    # previously active. The production example deliberately requires Triton/GPU.
    config = read(folder/'hippynn_config.json')
    config['hippynn_kernel']='pytorch'
    (tmp_path/'hippynn_config.json').write_text(json.dumps(config))
    (tmp_path/'references').mkdir()
    (tmp_path/'references/cpp9_s0.xyz').write_bytes((folder/'references/cpp9_s0.xyz').read_bytes())
    monkeypatch.setattr(benchmark, 'HERE', tmp_path)
    import hippynn
    original_kernel=hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE
    hippynn.custom_kernels.set_custom_kernels('_pytorch_raw')
    records = [benchmark.run_case("hipnn64", 1, mode, "cpu", warmup=0, steps=1, states=4,
                                  hippynn_kernel=None if mode=='train' else 'pytorch') for mode in ("train", "eval")]
    hippynn.custom_kernels.set_custom_kernels(original_kernel)
    assert [record["result"] for record in records] == ["passed", "passed"]
    assert all(record['requested_hippynn_kernel']==record['hippynn_kernel']=='pytorch' for record in records)
    assert records[0]["parameters"] > 0 and records[0]["median_step_s"] > 0
    [row] = benchmark.projections(records, train_rows=100, eval_rows=10, walltime_hours=16)
    expected = (100 * records[0]["median_step_s"] + 10 * records[1]["median_step_s"]) / 60
    assert row["includes_eval"] and row["epoch_minutes"] == pytest.approx(expected)
    for name in ("benchmark_training", "prepare"):
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("spares, expected", [(None, 2), (0, 0), (7, 7)])
def test_cpp_prepare_makes_spare_seeds(tmp_path, monkeypatch, spares, expected):
    from alframework.tools.batching import bootstrap_spare_count

    folder = ROOT / "examples" / "CPP_excited_gpu4pyscf"
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.delitem(sys.modules, "prepare", raising=False)
    prepare = importlib.import_module("prepare").prepare
    run = prepare(tmp_path / "cpp9", count=40, spares=spares)
    master = read(run / "master_config.json")
    assert (master["bootstrap_set"], master["bootstrap_spares"]) == (40, expected)
    assert bootstrap_spare_count(master) == expected
    assert read(run / "seed_config.json")["count"] == 40 + expected, "the seed pool holds the spares"
    assert read(run / "preparation.json")["spares"] == expected
    monkeypatch.delitem(sys.modules, "prepare", raising=False)


@pytest.mark.parametrize('solvent', ['dcm', 'gas'])
def test_cpp_prepare_keeps_solvent_and_vertical_schema_consistent(tmp_path, monkeypatch, solvent):
    folder = ROOT / 'examples' / 'CPP_excited_gpu4pyscf'
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.delitem(sys.modules, 'prepare', raising=False)
    prepare = importlib.import_module('prepare').prepare
    # Exercise the new DCM default as well as switching that template back to gas.
    options = {} if solvent=='dcm' else dict(solvent='gas')
    run = prepare(tmp_path/solvent, count=2, **options)
    assert read(run/'QM_config.json')['solvent']['enabled'] == (solvent=='dcm')
    properties = read(run/'master_config.json')['properties_list']
    vertical = {'vertical_excitation_energies', 'vertical_transition_dipoles', 'vertical_oscillator_strengths'}
    assert set(properties) & vertical == (vertical if solvent=='dcm' else set())
    assert read(run/'preparation.json')['solvent']==solvent
    monkeypatch.delitem(sys.modules, 'prepare', raising=False)


CPP_FOLDER = ROOT / "examples" / "CPP_excited_gpu4pyscf"
CPP_REFERENCES = sorted(CPP_FOLDER.glob("references/cpp*_s0.json"), key=lambda p: int(p.name[3:-8]))
CPP_THEORY = ("xc", "basis", "charge", "multiplicity", "density_fit", "auxbasis", "grids_level",
              "scf_conv_tol", "scf_max_cycle", "nroots", "cphf_conv_tol")


def _cpp_module(monkeypatch, name):
    monkeypatch.syspath_prepend(str(CPP_FOLDER))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_cpp_references_cover_six_through_twelve():
    assert [read(p)["ring_size"] for p in CPP_REFERENCES][:7] == list(range(6, 13))


@pytest.mark.parametrize("record_path", CPP_REFERENCES, ids=lambda p: p.stem)
def test_cpp_bundled_reference_is_consistent(record_path, monkeypatch):
    """Each bundled S0 reference matches its hash, ring size, offset, and the example's theory."""
    import hashlib

    from ase.io import read as read_atoms

    sampling = _cpp_module(monkeypatch, "initial_sampling")
    record, ring = read(record_path), int(record_path.name[3:-8])
    xyz = record_path.with_name(record["reference"])
    assert xyz.name == f"cpp{ring}_s0.xyz" and record["ring_size"] == ring
    assert hashlib.sha256(xyz.read_bytes()).hexdigest() == record["reference_sha256"]
    assert record["success"] and record["optimizer_converged"] and record["selected_state"] == 0
    assert record["energy_offset_eV"] == pytest.approx(
        record["ground_energy_hartree"] * record["hartree_to_ev"], abs=1e-8)
    qm = read(CPP_FOLDER / "QM_config.json")
    assert {key: record["theory"][key] for key in CPP_THEORY} == {key: qm[key] for key in CPP_THEORY}
    atoms = read_atoms(xyz)
    assert len(atoms) == 10 * ring and atoms.get_chemical_formula() == f"C{6 * ring}H{4 * ring}"
    topology = sampling.infer_topology(atoms, sampling.DEFAULT_SETTINGS["screening"])
    assert topology["nring"] == ring
    assert sampling.screen_geometry(atoms.positions, topology, sampling.DEFAULT_SETTINGS["screening"]) is None
    # CAM-B3LYP/6-31G S0 energies grow by about 230.88 Ha per phenylene; a wrong
    # offset copied from another size or theory would break this.
    assert record["ground_energy_hartree"] / ring == pytest.approx(-230.8, abs=0.3)


@pytest.mark.parametrize("record_path", CPP_REFERENCES, ids=lambda p: p.stem)
def test_cpp_prepare_uses_bundled_reference(record_path, tmp_path, monkeypatch):
    import h5py

    record, ring = read(record_path), int(record_path.name[3:-8])
    run = _cpp_module(monkeypatch, "prepare").prepare(
        tmp_path / f"cpp{ring}", ring_size=ring, count=8, spares=2, generate_seeds=True)
    assert read(run / "QM_config.json")["energy_offset_eV"] == record["energy_offset_eV"]
    assert read(run / "hippynn_config.json")["n_atoms"] == 10 * ring
    assert read(run / "preparation.json")["ring_size"] == ring
    assert (run / "reference.xyz").read_bytes() == record_path.with_name(record["reference"]).read_bytes()
    with h5py.File(run / "seeds/seed.h5", "r") as handle:
        assert handle[f"cpp{ring}/coordinates"].shape == (10, 10 * ring, 3)
    for name in ("prepare", "initial_sampling"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_cpp_prepare_rejects_unbundled_size(tmp_path, monkeypatch):
    prepare = _cpp_module(monkeypatch, "prepare").prepare
    with pytest.raises(ValueError, match="No bundled reference for CPP13"):
        prepare(tmp_path / "cpp13", ring_size=13)
    monkeypatch.delitem(sys.modules, "prepare", raising=False)


def test_example_masters_have_valid_bootstrap_spares():
    from alframework.tools.batching import bootstrap_spare_count

    for path in MASTERS:
        bootstrap_spare_count(read(path))


def test_no_old_state_names_remain():
    """State-aware code names states E#/F#; the old sE# names were removed, with no compatibility."""
    import re

    old = re.compile(r"(?<![A-Za-z_])sE[0-9#{]")
    found = [str(path.relative_to(ROOT)) for folder in ("alframework", "examples", "docs/source")
             for path in (ROOT / folder).rglob("*") if path.suffix in {".py", ".json", ".rst", ".md", ".txt"}
             and old.search(path.read_text(errors="ignore"))]
    assert found == []
