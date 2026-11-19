"""Every tracked example in examples/ loads with the current validators and names tasks that exist.

Submits no jobs and needs no QM programs. NeuroChem examples are imported against a fake backend.
"""

import ast
import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from alframework.qm_interfaces.pyscf.gpu4pyscf import validate_gpu4pyscf_config
from alframework.samplers.acquisition import acquisition_settings
from alframework.tools.batching import batching_settings
from alframework.tools.labeling_coordinator import labeling_batch_settings
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
    from alframework.tools.bootstrap import bootstrap_spare_count

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
    from alframework.tools.bootstrap import bootstrap_spare_count

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



def orca_masters():
    return [path for path in MASTERS if read(path)["QM_task"].startswith("alframework.qm_interfaces.ase.")
            and "orca" in read(path)["QM_config_path"].lower()]


@pytest.mark.parametrize("path", orca_masters(), ids=lambda path: str(path.relative_to(ROOT / "examples")))
def test_orca_examples_use_ase_orca_with_ase_units(path):
    master = read(path)
    config = read(path.parent / master["QM_config_path"])
    module, name = master["QM_task"].rsplit(".", 1)
    getattr(importlib.import_module(module), name).func.validate_config(config, master)
    for stage in config.get("stages", [config]):
        assert stage["ASE_calculator"] == "ase.calculators.orca.ORCA"
        assert not {"ncpu", "orca_env_file", "Ediff", "Fdiff"} & set(stage)
    properties = master["properties_list"]
    assert set(properties) <= {"energy", "forces", "dipole"}
    assert all(spec[2] == 1.0 for spec in properties.values()), "ASE gives eV and eV/Angstrom"


@pytest.mark.parametrize("path", orca_masters(), ids=lambda path: str(path.relative_to(ROOT / "examples")))
def test_orca_examples_train_on_stored_datasets_of_clusters(path):
    master = read(path)
    stored = {spec[0] for spec in master["properties_list"].values()}
    training = read(path.parent / master["ML_config_path"])
    for key in ("energy_key", "force_key", "charges_key", "dipole_key", "quadrupole_key"):
        if training.get(key) is not None:
            assert training[key] in stored
    assert training.get("cell_key") is None
    builder = read(path.parent / master["builder_config_path"])
    if "condensed_phase" in master["builder_task"] or "cfg_loader" in master["builder_task"]:
        assert builder["periodic"] is False, "ORCA labels molecules, so the builder makes clusters"


@pytest.mark.parametrize("path", orca_masters(), ids=lambda path: str(path.relative_to(ROOT / "examples")))
def test_orca_examples_label_through_ase_with_a_stand_in_orca(path, tmp_path):
    from alframework.qm_interfaces.ase.generic import label_one
    from alframework.qm_interfaces.ase.workflow import label_workflow
    from tests.helpers.ase_qm import orca_command
    from tests.helpers.labels import unlabeled_h2

    master = read(path)
    config = read(path.parent / master["QM_config_path"])
    for stage in config.get("stages", [config]):
        stage["QM_run_command"] = orca_command(tmp_path)
    label = label_workflow if "stages" in config else label_one
    molecule = label(unlabeled_h2(), config, str(tmp_path / "scratch"), master["properties_list"])
    assert molecule.check_convergence(), molecule.metadata
    assert molecule.metadata["qm_backend"] == "orca"
    assert set(master["properties_list"]) <= set(molecule.get_results())


def test_the_il_example_repeats_its_orca_calculation_and_keeps_the_old_tolerances():
    config = read(ROOT / "examples" / "IL" / "orca_config.json")
    assert config["result_mode"] == "mean_if_agree" and len(config["stages"]) == 2
    # The legacy double calculation compared at 1e-3 Hartree and 1e-4 Hartree/Bohr.
    assert config["agreement"]["energy"] == pytest.approx(1e-3 * 27.211386024367243)
    assert config["agreement"]["forces"] == pytest.approx(1e-4 * 51.422067090480645)
    assert all("%pal nprocs 128 end" in stage["input"]["orcablocks"] for stage in config["stages"])


def test_the_staged_orca_example_reads_the_small_basis_orbitals(tmp_path, monkeypatch, capsys):
    from ase.units import Hartree
    from tests.helpers.ase_qm import executable, orca_output

    command = executable(tmp_path, f"""
import sys
from pathlib import Path

request = Path(sys.argv[1]).read_text()
if Path.cwd().name == "guess":
    assert "def2-SVP" in request and "EnGrad" not in request
    Path("orca.gbw").write_bytes(b"small-basis orbitals")
    print({orca_output(energy="-2.5")!r})
else:
    assert "def2-TZVP" in request and "EnGrad" in request and "MORead" in request
    assert '%moinp "guess.gbw"' in request
    assert Path("guess.gbw").read_bytes() == b"small-basis orbitals"
    Path("orca.engrad").write_text("# The current gradient\\n#\\n" + "0.1\\n" * 6 + "# The atomic numbers\\n")
    print({orca_output(energy="-3.0")!r})
""")
    script = ROOT / "examples" / "staged_ase_qm" / "run_workflow.py"
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(sys, "argv", [str(script), "orca", "--command", command, "--scratch-dir", str(scratch)])
    runpy.run_path(str(script), run_name="__main__")
    assert (scratch / "h2/target/guess.gbw").read_bytes() == b"small-basis orbitals"
    assert f"Final energy (eV): {-3.0 * Hartree}" in capsys.readouterr().out


@pytest.mark.parametrize("repeat", [(1, 1, 1), (2, 2, 2)], ids=["2 atoms", "16 atoms"])
def test_the_staged_vasp_example_labels_iron_cells_of_any_size(tmp_path, monkeypatch, repeat):
    """ASE writes the real VASP input for each stage; only the VASP run itself is a stand-in."""
    import numpy as np
    from ase.build import bulk
    from ase.calculators.vasp import Vasp
    from alframework.qm_interfaces.ase.workflow import label_workflow
    from alframework.tools.molecules import MoleculesObject

    potentials = tmp_path / "potentials" / "potpaw_PBE" / "Fe"
    potentials.mkdir(parents=True)
    (potentials / "POTCAR").write_text("PAW_PBE Fe 06Sep2000\n ZVAL = 8.000\n End of Dataset\n")
    monkeypatch.setenv("VASP_PP_PATH", str(tmp_path / "potentials"))
    atoms = bulk("Fe", "bcc", a=2.86, cubic=True).repeat(repeat)

    def run(self, command=None, out=None, directory=None):
        incar = (Path(self.directory) / "INCAR").read_text().upper()
        if "ISPIN = 1" in incar:
            assert "ICHARG = 2" in incar and "LCHARG = .TRUE." in incar
            (Path(self.directory) / "CHGCAR").write_text("nonmagnetic charge density")
        else:
            assert "ISPIN = 2" in incar and "ICHARG = 1" in incar
            assert f"MAGMOM = {len(atoms)}*2.2" in incar
            assert (Path(self.directory) / "CHGCAR").read_text() == "nonmagnetic charge density"
        return 0, ""

    def read_results(self):
        self.results = {"energy": -1.0, "forces": np.ones((len(self.atoms), 3))}
        self.converged = True

    monkeypatch.setattr(Vasp, "_run", run)
    monkeypatch.setattr(Vasp, "read_results", read_results)
    directory = ROOT / "examples" / "staged_ase_qm"
    master = read(directory / "staged_vasp_master_config_fragment.json")
    config = read(directory / master["QM_config_path"])
    molecule = label_workflow(MoleculesObject(atoms, "fe"), config, str(tmp_path / "scratch"),
                              master["properties_list"])
    assert molecule.check_convergence(), molecule.metadata
    assert [stage["name"] for stage in molecule.metadata["qm_workflow"]["stages"]] == ["charge_density", "magnetic"]


@pytest.mark.parametrize("family", ["fine_tuning", "seeded_active_learning"])
def test_cfg_seed_examples_build_clusters(family):
    from parsl.app.errors import RemoteExceptionWrapper

    from alframework.builders.builders import simple_cfg_loader_task

    folder = ROOT / "examples" / family
    config = read(folder / "builder_config.json")
    config["molecule_library_dir"] = str(folder / config["molecule_library_dir"])
    molecule = simple_cfg_loader_task.func("seed", config, shake=0.0)
    if isinstance(molecule, RemoteExceptionWrapper):
        molecule.reraise()
    assert not molecule.atoms.pbc.any()


@pytest.mark.parametrize("path", MASTERS, ids=lambda path: str(path.relative_to(ROOT / "examples")))
def test_examples_name_current_module_paths(path):
    """Moved modules still import from their old paths, with a FutureWarning; examples use the new ones."""
    from alframework import qm_interfaces, samplers, tools

    moved = tuple(f"{package.__name__}.{name}." for package in (qm_interfaces, samplers, tools)
                  for name in package.MOVED_MODULES)
    for field, value in read(path).items():
        if isinstance(value, str):
            assert not value.startswith(moved), f"{field} names a moved module: {value}"
