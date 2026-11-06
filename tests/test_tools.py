"""alframework.tools.tools helpers, HDF5 storage, and the Parsl task queue."""
import json
import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import (
    annealing_schedule,
    compute_empirical_formula,
    load_config_file,
    random_rotation_matrix,
    store_current_data,
    system_checker,
)
from alframework.tools.tools import is_final_task, parsl_task_queue
from tests.helpers.fakes import FakeTask


@pytest.mark.parametrize(
    "t, amp, expected",
    [(0.0, 0.0, 100.0), (5.0, 0.0, 200.0), (10.0, 0.0, 300.0), (1.0, 50.0, 170.0)],
    ids=["start", "midpoint", "end", "with_oscillation"],
)
def test_annealing_schedule(t, amp, expected):
    assert annealing_schedule(t=t, tmax=10.0, amp=amp, per=2.0, srt=100.0, end=300.0) == pytest.approx(expected)


def test_compute_empirical_formula():
    assert compute_empirical_formula(["H", "O", "H", "C"]) == "C01_H02_O01"


def test_random_rotation_matrix():
    rotation = random_rotation_matrix(randnums=np.array([0.25, 0.5, 0.75]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(abs(np.linalg.det(rotation)), 1.0)


def test_system_checker_accepts_valid_system_and_rejects_bad_data():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    results = {"energy": np.array([-1.0])}
    for identifier in ({"moleculeid": "h2"}, "h2"):
        for sequence_type in (list, tuple):
            assert system_checker(sequence_type((identifier, atoms, results)))
    molecule_object = MoleculesObject(atoms.copy(), "h2")
    molecule_object.store_results(results)
    assert system_checker(molecule_object)
    assert not system_checker([{"moleculeid": "bad"}, atoms, {"forces": np.array([np.nan])}], kill_on_fail=False)
    molecule_object.store_results({"forces": np.array([np.nan])})
    invalid_systems = [
        molecule_object,
        [{}, atoms, {}],
        [{"moleculeid": 1}, atoms, {}],
        ["h2", None, {}],
        ["h2", atoms, []],
    ]
    for system in invalid_systems:
        assert not system_checker(system, kill_on_fail=False, print_error=False)
        with pytest.raises(RuntimeError):
            system_checker(system, print_error=False)


def test_load_config_file_paths(tmp_path):
    config_path = tmp_path / "master.json"
    config_path.write_text(
        json.dumps(
            {
                "master_directory": "pwd",
                "scratch_dir": "scratch",
                "model_path": "models/model-{:04d}",
                "absolute_dir": "/already/absolute",
            }
        )
    )

    config = load_config_file(str(config_path))

    assert config["master_directory"].endswith("/")
    assert config["scratch_dir"].endswith("/scratch")
    assert config["model_path"].endswith("/models/model-{:04d}")
    assert config["model_dir"].endswith("/models/")
    assert config["absolute_dir"] == "/already/absolute"


def test_store_current_data_writes_converged_sorted_h5(tmp_path):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    molecule = MoleculesObject(atoms, "water-0")
    molecule.store_results({"energy": -1.5, "forces": np.arange(9).reshape(3, 3)})
    molecule.set_converged_flag(True)

    unconverged = MoleculesObject(atoms.copy(), "skip-me")
    unconverged.store_results({"energy": 0.0, "forces": np.zeros((3, 3))})
    unconverged.set_converged_flag(False)

    h5_path = tmp_path / "data.h5"
    store_current_data(
        str(h5_path),
        [molecule, unconverged],
        {"energy": ["energy", "system", 2.0], "forces": ["forces", "atomic", 0.5]},
    )

    with h5py.File(h5_path, "r") as h5:
        assert list(h5.keys()) == ["H02_O01"]
        group = h5["H02_O01"]
        assert [item.decode("utf-8") for item in group["species"][()]] == ["H", "H", "O"]
        assert [item.decode("utf-8") for item in group["_id"][()]] == ["water-0"]
        np.testing.assert_allclose(group["energy"][()], [-3.0])
        assert group["forces"].shape == (1, 3, 3)
        assert group["cell"].shape == (1, 3, 3)


def test_store_current_data_omits_cell_for_nonperiodic_structures(tmp_path):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], cell=[20.0] * 3)
    molecule = MoleculesObject(atoms, "sample")
    molecule.store_results({"energy": -1.0, "forces": np.arange(9.0).reshape(3, 3)})
    molecule.set_converged_flag(True)

    h5_path = tmp_path / "data.h5"
    store_current_data(str(h5_path), [molecule],
                       {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]})

    with h5py.File(h5_path, "r") as h5:
        group = h5["H02_O01"]
        assert "cell" not in group
        assert group["coordinates"].shape == (1, 3, 3)


def test_parsl_task_queue_counts_statuses():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("exec_done", done=True))
    queue.add_task(FakeTask("exec_done", done=False))
    queue.add_task(FakeTask("running", done=False, running=True))
    queue.add_task(FakeTask("failed", done=True))

    assert queue.get_number() == 4
    assert queue.get_completed_number() == 3
    assert queue.get_running_number() == 1
    assert queue.get_exec_done_number() == 1
    assert queue.get_failed_number() == 1
    assert queue.get_queued_number() == 0
    assert queue.get_task_status() == ["exec_done", "exec_done", "running", "failed"]

    for status in ("exec_done", "memo_done", "failed", "dep_fail"):
        assert is_final_task(FakeTask(status, done=False))
    for status in ("pending", "launched", "running", "joining", "running_ended", "fail_retryable", "unknown"):
        assert not is_final_task(FakeTask(status, done=False))


def test_parsl_task_queue_collects_results_and_removes_finished_tasks():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("running", result="keep", done=False, running=True))
    queue.add_task(FakeTask("exec_done", result="done", done=True))
    queue.add_task(FakeTask("memo_done", result="memoized", done=True))
    queue.add_task(FakeTask("failed", exception=RuntimeError("task failed")))
    queue.add_task(FakeTask("dep_fail", result="drop", done=True))
    intermediate_statuses = ["pending", "launched", "joining", "running_ended", "fail_retryable"]
    for status in intermediate_statuses:
        queue.add_task(FakeTask(status, result="keep", done=False))

    assert queue.get_failed_number() == 2

    results, failed = queue.get_task_results()

    assert sorted(results) == ["done", "memoized"]
    assert failed == 2
    assert queue.get_number() == 1 + len(intermediate_statuses)
    assert queue.get_task_status() == ["running"] + intermediate_statuses
