"""Foundation-model teacher labeling: batches, failures, provenance, startup checks, and the real loop.

Uses a real TorchSim Lennard-Jones model for batching, which needs torch and TorchSim, and an ASE
calculator or scripted teacher otherwise. Real MACE and UMA aren't installed in CI. They were checked
separately, and the roadmap records how.
"""

import json

import h5py
import numpy as np
import pytest
from ase import Atoms
from parsl import python_app

from alframework.qm_interfaces import teacher_interface as ti
from alframework.tools import tools
from alframework.tools.labeling import LabelingCoordinator
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers import teachers
from tests.helpers.driver import MODEL_PATTERN, run_alf_driver
from tests.helpers.molecules import WATER

LOADER = "tests.helpers.teachers.load_{}_teacher"
PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}


@pytest.fixture(autouse=True)
def fresh_workers():
    ti._TEACHERS.clear()
    teachers.LOADS.clear()
    yield
    ti._TEACHERS.clear()


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "teacher.model"
    path.write_bytes(b"weights")
    return path


def config(checkpoint, kind="scripted", **options):
    return {"teacher": {"loader": LOADER.format(kind), "checkpoint": str(checkpoint), "device": "cpu",
                        "name": f"{kind}-teacher", "options": options}}


def structures(count=6, periodic=(), seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        atoms = Atoms("H" * (2 + index % 3), positions=rng.uniform(0, 3, (2 + index % 3, 3)))
        if index in periodic:
            atoms.set_cell([6.0, 6.0, 6.0])
            atoms.pbc = True
        result.append(MoleculesObject(atoms, f"mol-0000-{index:010d}"))
    return result


def label(molecules, qm_config, **kw):
    return ti.label_with_teacher(molecules, qm_config, PROPERTIES, **kw)


def results(molecules):
    return [(m.get_results()["energy"], m.get_results()["forces"]) for m in molecules]


# Batches.


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_teacher_batches_match_one_at_a_time(checkpoint, width):
    pytest.importorskip("torch_sim")
    qm_config = config(checkpoint, "lennard_jones")
    single = [label([m], qm_config)[0] for m in structures(8, periodic={2, 5})]
    batched = structures(8, periodic={2, 5})
    for start in range(0, len(batched), width):
        label(batched[start:start + width], qm_config)
    for (e1, f1), (e2, f2) in zip(results(single), results(batched)):
        assert e2 == pytest.approx(e1, abs=1e-12)
        np.testing.assert_allclose(f2, f1, atol=1e-12)
    assert all(m.check_convergence() for m in batched)


def test_teacher_returns_labels_on_the_right_molecules_in_input_order(checkpoint):
    pytest.importorskip("torch_sim")
    qm_config = config(checkpoint, "lennard_jones")
    molecules = structures(6, periodic={0, 3})
    order = [4, 0, 5, 2, 1, 3]
    shuffled = [molecules[index] for index in order]
    returned = label(shuffled, qm_config)
    assert [m.get_moleculeid() for m in returned] == [m.get_moleculeid() for m in shuffled]
    alone = {m.get_moleculeid(): label([m], qm_config)[0].get_results()["energy"] for m in structures(6, periodic={0, 3})}
    for molecule in returned:
        assert molecule.get_results()["energy"] == pytest.approx(alone[molecule.get_moleculeid()], abs=1e-12)
        assert molecule.get_results()["forces"].shape == (len(molecule.get_atoms()), 3)


def test_teacher_evaluates_periodic_and_open_structures_separately(checkpoint):
    molecules = structures(6, periodic={1, 4})
    label(molecules, config(checkpoint))
    teacher = next(iter(ti._TEACHERS.values()))
    assert sorted(len(call) for call in teacher.calls) == [2, 4]


def test_ase_teacher_matches_its_calculator(checkpoint):
    from tests.helpers.fakes import PairSpringCalculator

    molecule = structures(1)[0]
    label([molecule], config(checkpoint, "spring"))
    atoms = molecule.get_atoms().copy()
    atoms.calc = PairSpringCalculator(3.5)
    assert molecule.get_results()["energy"] == pytest.approx(atoms.get_potential_energy())
    np.testing.assert_allclose(molecule.get_results()["forces"], atoms.get_forces())


# Failures.


def test_teacher_isolates_a_structure_that_breaks_a_batch(checkpoint):
    molecules = structures(8)
    label(molecules, config(checkpoint, fail_on_size=4))
    bad = [m for m in molecules if len(m.get_atoms()) == 4]
    assert bad and all(not m.check_convergence() for m in bad)
    assert all(m.check_convergence() for m in molecules if len(m.get_atoms()) != 4)
    error = bad[0].get_metadata()["labeling_error"]
    assert "unsupported structure" in error and "batch error" in error


def test_teacher_marks_non_finite_labels_unconverged(checkpoint):
    molecules = label(structures(2), config(checkpoint, energy=float("nan")))
    assert not any(m.check_convergence() for m in molecules)
    assert "non-finite" in molecules[0].get_metadata()["labeling_error"]


def test_teacher_keeps_large_finite_forces(checkpoint):
    molecules = label(structures(2), config(checkpoint, force=5000.0))
    assert all(m.check_convergence() for m in molecules)


def test_teacher_that_cannot_load_fails_the_task(checkpoint):
    with pytest.raises(RuntimeError, match="checkpoint is corrupt"):
        label(structures(1), config(checkpoint, "broken"))


def test_teacher_loads_once_per_worker_and_again_for_new_settings(checkpoint):
    qm_config = config(checkpoint)
    label(structures(2), qm_config)
    label(structures(2), qm_config)
    assert len(teachers.LOADS) == 1
    label(structures(2), config(checkpoint, energy=-2.0))
    assert len(teachers.LOADS) == 2 and len(ti._TEACHERS) == 1, "only one teacher is held per worker"


def test_teacher_rejects_elements_it_has_no_parameters_for(checkpoint):
    molecules = structures(2)
    molecules[1].get_atoms().set_atomic_numbers([8] * len(molecules[1].get_atoms()))
    label(molecules, config(checkpoint, elements=[1]))
    assert molecules[0].check_convergence() and not molecules[1].check_convergence()
    assert "atomic numbers [8]" in molecules[1].get_metadata()["labeling_error"]


@pytest.mark.parametrize("metadata", [{"charge": 1}, {"mult": 3}], ids=["charged", "open_shell"])
def test_teacher_without_charge_and_spin_rejects_other_states(checkpoint, metadata):
    molecule = structures(1)[0]
    molecule.update_metadata(metadata)
    label([molecule], config(checkpoint))
    assert not molecule.check_convergence()
    assert "neutral singlets" in molecule.get_metadata()["labeling_error"]


def test_teacher_with_charge_and_spin_receives_them(checkpoint):
    molecules = structures(2)
    molecules[0].update_metadata({"charge": -1, "mult": 2})
    label(molecules, config(checkpoint, charge_spin=True))
    teacher = next(iter(ti._TEACHERS.values()))
    assert {(info["charge"], info["spin"]) for info in teacher.infos} == {(-1, 2), (0, 1)}
    assert all(m.check_convergence() for m in molecules)


# Provenance.


def test_teacher_is_named_after_its_checkpoint_by_default(checkpoint):
    qm_config = config(checkpoint)
    qm_config["teacher"].pop("name")
    molecule = label(structures(1), qm_config)[0]
    assert molecule.get_metadata()["label_provenance"]["teacher"] == "teacher"


def test_teacher_checks_properties_again_when_labeling(checkpoint):
    """``properties_list`` is re-read every loop, so an edit after startup must still be caught."""
    hartree = dict(PROPERTIES, energy=["energy", "system", 27.211386])
    with pytest.raises(ValueError, match="unit factor"):
        ti.label_with_teacher(structures(1), config(checkpoint), hartree)


def test_teacher_labels_record_the_teacher(checkpoint):
    molecule = label(structures(1), config(checkpoint))[0]
    provenance = molecule.get_metadata()["label_provenance"]
    assert provenance["teacher"] == "scripted-teacher"
    assert provenance["checkpoint_sha256"] == ti.file_sha256(checkpoint)
    assert provenance["units"] == {"energy": "eV", "forces": "eV/Å"}
    assert provenance["loader"] == LOADER.format("scripted")


def test_store_writes_teacher_provenance_onto_the_shard(checkpoint, tmp_path):
    molecules = label(structures(4, periodic=set()), config(checkpoint, fail_on_size=4))
    path = tmp_path / "data-0000.h5"
    tools.store_current_data(str(path), molecules, PROPERTIES)
    (source,) = tools.read_label_provenance(path)
    assert source["checkpoint_sha256"] == ti.file_sha256(checkpoint)
    with h5py.File(path) as handle:
        assert sum(len(group["energy"]) for group in handle.values()) == sum(m.check_convergence() for m in molecules)


def test_shard_without_converged_teacher_labels_records_no_teacher(checkpoint, tmp_path):
    molecules = label(structures(2), config(checkpoint, energy=float("nan")))
    path = tmp_path / "data-0000.h5"
    tools.store_current_data(str(path), molecules, PROPERTIES)
    assert tools.read_label_provenance(path) == []


def test_qm_shards_are_unchanged(tmp_path):
    molecule = structures(1)[0]
    molecule.store_results({"energy": -1.0, "forces": np.zeros((2, 3))})
    molecule.set_converged_flag(True)
    path = tmp_path / "data-0000.h5"
    tools.store_current_data(str(path), [molecule], PROPERTIES)
    with h5py.File(path) as handle:
        assert "label_provenance" not in handle.attrs
    assert tools.read_label_provenance(path) == []


def test_hippynn_reads_a_shard_that_has_provenance(checkpoint, tmp_path):
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB

    tools.store_current_data(str(tmp_path / "data-0000.h5"), label(structures(4), config(checkpoint)), PROPERTIES)
    database = ALFPyAniDirectoryDB(directory=str(tmp_path), inputs=["species", "coordinates"],
                                   targets=["energy", "forces"], quiet=True, seed=0)
    assert len(database.arr_dict["energy"]) == 4


def test_label_sources_record_what_a_model_learned_from(checkpoint, tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    tools.store_current_data(str(h5 / "data-0000.h5"), label(structures(2), config(checkpoint)), PROPERTIES)
    qm = structures(1)[0]
    qm.store_results({"energy": -1.0, "forces": np.zeros((2, 3))})
    qm.set_converged_flag(True)
    tools.store_current_data(str(h5 / "data-0001.h5"), [qm], PROPERTIES)
    tools.write_label_sources(tmp_path / "models/model-0001", h5)
    record = json.loads((tmp_path / "models/model-0001/label_sources.json").read_text())
    assert [teacher["teacher"] for teacher in record["teachers"]] == ["scripted-teacher"]
    assert record["shards_without_label_provenance"] == ["data-0001.h5"] and record["shards"] == 2


def test_label_sources_failure_is_reported_not_raised(tmp_path, capsys):
    (tmp_path / "file").write_text("")
    tools.write_label_sources(tmp_path / "file" / "ensemble", tmp_path)
    assert "Could not record label sources" in capsys.readouterr().out


# Startup checks.


def master(tmp_path, **settings):
    return dict(properties_list=PROPERTIES, master_directory=str(tmp_path), **settings)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda c: c["teacher"].pop("loader"), "teacher.loader"),
        (lambda c: c["teacher"].pop("checkpoint"), "teacher.checkpoint"),
        (lambda c: c["teacher"].update(checkpoint="missing.model"), "checkpoint not found"),
        (lambda c: c["teacher"].update(dtype="float16"), "teacher.dtype"),
        (lambda c: c["teacher"].update(allow_mixed_labels="yes"), "allow_mixed_labels"),
        (lambda c: c.pop("teacher"), "'teacher' block"),
    ],
    ids=["no_loader", "no_checkpoint", "missing_checkpoint", "bad_dtype", "bad_mixed_flag", "no_block"],
)
def test_teacher_config_mistakes_stop_alf_at_startup(checkpoint, tmp_path, change, message):
    qm_config = config(checkpoint)
    change(qm_config)
    with pytest.raises(ValueError, match=message):
        ti.validate_teacher_config(qm_config, master(tmp_path))


@pytest.mark.parametrize(
    "properties,message",
    [
        ({"energy": ["energy", "system", 27.211386]}, "unit factor"),
        ({"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 51.42]}, "unit factor"),
        ({"energy": ["energy", "system", 1.0], "dipole": ["dipole", "system", 1.0]}, "not 'dipole'"),
        ({"energy": ["energy", "atomic", 1.0]}, "must be 'system'"),
        ({"forces": ["forces", "atomic", 1.0]}, "needs an 'energy'"),
    ],
    ids=["hartree_energy", "hartree_forces", "other_property", "wrong_scope", "no_energy"],
)
def test_teacher_refuses_properties_it_cannot_give(checkpoint, tmp_path, properties, message):
    with pytest.raises(ValueError, match=message):
        ti.validate_teacher_config(config(checkpoint), dict(master(tmp_path), properties_list=properties))


def test_uma_teacher_needs_a_task(checkpoint, tmp_path):
    qm_config = config(checkpoint)
    qm_config["teacher"]["loader"] = "alframework.qm_interfaces.teacher_interface.load_uma_teacher"
    with pytest.raises(ValueError, match="teacher.task"):
        ti.validate_teacher_config(qm_config, master(tmp_path))
    qm_config["teacher"]["task"] = "omol"
    ti.validate_teacher_config(qm_config, master(tmp_path))


def test_relative_checkpoint_is_found_from_the_master_directory(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models/teacher.model").write_bytes(b"weights")
    ti.validate_teacher_config(config("models/teacher.model"), master(tmp_path))


def shard(path, molecules):
    tools.store_current_data(str(path), molecules, PROPERTIES)


def test_teacher_refuses_an_h5_dir_with_qm_labels(checkpoint, tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    qm = structures(1)[0]
    qm.store_results({"energy": -1.0, "forces": np.zeros((2, 3))})
    qm.set_converged_flag(True)
    shard(h5 / "data-0000.h5", [qm])
    with pytest.raises(ValueError, match="no teacher recorded"):
        ti.validate_teacher_config(config(checkpoint), master(tmp_path, h5_dir=str(h5)))
    allowed = config(checkpoint)
    allowed["teacher"]["allow_mixed_labels"] = True
    ti.validate_teacher_config(allowed, master(tmp_path, h5_dir=str(h5)))


def test_teacher_refuses_labels_from_another_teacher(checkpoint, tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    other = tmp_path / "other.model"
    other.write_bytes(b"other weights")
    shard(h5 / "data-0000.h5", label(structures(1), config(other)))
    with pytest.raises(ValueError, match="another teacher"):
        ti.validate_teacher_config(config(checkpoint), master(tmp_path, h5_dir=str(h5)))


def test_teacher_accepts_its_own_earlier_labels(checkpoint, tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    shard(h5 / "data-0000.h5", label(structures(1), config(checkpoint)))
    ti.validate_teacher_config(config(checkpoint), master(tmp_path, h5_dir=str(h5)))


@pytest.mark.parametrize("task,batching", [(ti.teacher_batch_task, {"mode": "native", "batch_size": 4}), (ti.teacher_task, None)],
                         ids=["teacher_batch_task", "teacher_task"])
def test_labeling_coordinator_runs_the_teacher_check_at_startup(checkpoint, tmp_path, task, batching):
    bad = config(checkpoint)
    bad["teacher"]["checkpoint"] = str(tmp_path / "missing.model")
    with pytest.raises(ValueError, match="checkpoint not found"):
        LabelingCoordinator(task, "teacher", dict(bad, **({"batching": batching} if batching else {})), master(tmp_path))
    assert LabelingCoordinator(task, "teacher", config(checkpoint), master(tmp_path)) is not None


def test_labeling_coordinator_ignores_labelers_without_a_check():
    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        return molecule_object

    assert LabelingCoordinator(qm, "qm", {}, {}) is not None


# The real coordinator.


@pytest.mark.parametrize(
    "task,batching",
    [(ti.teacher_batch_task, {"mode": "native", "batch_size": 3, "max_wait_seconds": 0.05}),
     (ti.teacher_task, {"mode": "sequential", "batch_size": 3, "max_wait_seconds": 0.05}),
     (ti.teacher_task, None)],
    ids=["native", "sequential", "unbatched"],
)
def test_teacher_labels_a_full_training_cycle(tmp_path, monkeypatch, task, batching):
    checkpoint = tmp_path / "teacher.model"
    checkpoint.write_bytes(b"weights")
    trained = []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id, h5_dir):
        trained.append(current_training_id)
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        tools.write_label_sources(tmp_path / MODEL_PATTERN.format(current_training_id), h5_dir)
        return [True] * 4, current_training_id

    qm_config = config(checkpoint, "spring")
    if batching:
        qm_config["batching"] = batching
    run = run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=sampler, qm=task, ml=ml, qm_config=qm_config,
                         bootstrap_set=4, parallel_samplers=2, target_queued_QM=3, save_h5_threshold=3,
                         maximum_completed_QM=8)
    assert run.status["current_model_id"] >= 1 and run.status["lifetime_failed_QM_tasks"] == 0
    shards = sorted((tmp_path / "h5").glob("data-*.h5"))
    assert len(shards) >= 2
    for path in shards:
        (source,) = tools.read_label_provenance(path)
        assert source["teacher"] == "spring-teacher" and source["adapter"] == "AseTeacher"
    record = json.loads((tmp_path / MODEL_PATTERN.format(1) / "label_sources.json").read_text())
    assert [t["teacher"] for t in record["teachers"]] == ["spring-teacher"]


def test_teacher_is_used_from_one_thread_at_a_time(checkpoint):
    """Thread-based executors share a worker's teacher, and models such as MACE keep state between calls."""
    from concurrent.futures import ThreadPoolExecutor

    qm_config = config(checkpoint, "stateful")
    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(lambda seed: label(structures(3, seed=seed), qm_config), range(12)))
    errors = {m.get_metadata().get("labeling_error") for batch in batches for m in batch} - {None}
    assert not errors and all(m.check_convergence() for batch in batches for m in batch)
