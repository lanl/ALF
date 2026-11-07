"""Restarting after a crash: whatever a crash leaves on disk must let the same command continue.

Status and HDF5 shards are written to a temporary name and renamed when complete, so a crash leaves
either the old file or the new one, never half of one.
"""

import ast
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from alframework.tools import files, pyanitools, tools
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import find_empty_directory, store_current_data, write_label_sources, write_status
from tests.helpers.driver import run_alf_driver, stored_ids
from tests.helpers.molecules import WATER

PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}
MAIN = Path(__file__).resolve().parents[1] / "alframework" / "__main__.py"


def labeled(name="water-0", energy=-1.0):
    molecule = MoleculesObject(WATER.copy(), name)
    molecule.store_results({"energy": energy, "forces": np.zeros((len(WATER), 3))})
    molecule.set_converged_flag(True)
    return molecule


def crash_inside_store_data(monkeypatch):
    """Fail after the first group is written, as a crash partway through a store would."""
    original = pyanitools.datapacker.store_data

    def store_then_crash(self, store_loc, **kwargs):
        original(self, store_loc, **kwargs)
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(pyanitools.datapacker, "store_data", store_then_crash)


def test_write_status_replaces_the_file(tmp_path):
    path = tmp_path / "status.json"
    write_status(str(path), {"current_h5_id": 1})
    write_status(str(path), {"current_h5_id": 2})
    assert json.loads(path.read_text()) == {"current_h5_id": 2}
    assert os.listdir(tmp_path) == ["status.json"]


def test_write_status_keeps_the_old_status_when_a_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    write_status(str(path), {"current_h5_id": 1})

    def fail_after_writing(descriptor):
        raise OSError("node lost")

    # The temporary file is written in full, then the flush to disk fails.
    monkeypatch.setattr(files.os, "fsync", fail_after_writing)
    with pytest.raises(OSError, match="node lost"):
        write_status(str(path), {"current_h5_id": 2})
    assert json.loads(path.read_text()) == {"current_h5_id": 1}
    assert os.listdir(tmp_path) == ["status.json"]


def test_coordinator_writes_status_only_through_write_status():
    """A direct ``open(..., "w")`` on the status file would bring back truncated status after a crash."""
    direct = [node.lineno for node in ast.walk(ast.parse(MAIN.read_text()))
              if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"
              and "status_path" in ast.unparse(node.args[0])
              and "w" in ast.unparse(node.args[1:2] or [ast.Constant("r")])]
    assert direct == []
    # Startup, two in --test_ml, each bootstrap loop, after bootstrap training starts, each loop, and the budget stop.
    assert MAIN.read_text().count("write_status(master_config['status_path'], status)") == 7


def test_store_current_data_leaves_nothing_when_a_store_fails(tmp_path, monkeypatch):
    crash_inside_store_data(monkeypatch)
    with pytest.raises(OSError, match="disk quota"):
        store_current_data(str(tmp_path / "data-0000.h5"), [labeled()], PROPERTIES)
    assert os.listdir(tmp_path) == []


def test_store_current_data_refuses_to_replace_a_shard(tmp_path):
    path = tmp_path / "data-0000.h5"
    store_current_data(str(path), [labeled(energy=-1.0)], PROPERTIES)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="ALF never replaces a shard"):
        store_current_data(str(path), [labeled(energy=-2.0)], PROPERTIES)
    assert path.read_bytes() == before
    assert os.listdir(tmp_path) == ["data-0000.h5"]


def test_store_current_data_overwrites_a_partial_shard_left_by_a_crash(tmp_path):
    path = tmp_path / "data-0000.h5"
    Path(str(path) + ".partial").write_bytes(b"half a shard")
    assert store_current_data(str(path), [labeled()], PROPERTIES) == 1
    assert os.listdir(tmp_path) == ["data-0000.h5"]
    with h5py.File(path) as handle:
        np.testing.assert_allclose(handle["H02_O01"]["energy"][()], [-1.0])


def test_readers_ignore_a_partial_shard(tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    store_current_data(str(h5 / "data-0000.h5"), [labeled()], PROPERTIES)
    (h5 / "data-0001.h5.partial").write_bytes(b"half a shard")

    assert find_empty_directory(str(h5 / "data-{:04d}.h5")) == 1
    write_label_sources(str(tmp_path / "model"), str(h5))
    assert json.loads((tmp_path / "model" / "label_sources.json").read_text()) == {
        "teachers": [], "shards_without_label_provenance": ["data-0000.h5"], "shards": 1}

    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB, _h5_shards

    assert _h5_shards(str(h5)) == [str(h5 / "data-0000.h5")]
    database = ALFPyAniDirectoryDB(directory=str(h5), inputs=["species", "coordinates"],
                                   targets=["energy", "forces"], quiet=True, seed=0)
    assert len(database.arr_dict["energy"]) == 1


def coordinator_tasks():
    from parsl import python_app

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        return [True, True], current_training_id

    return dict(builder=builder, sampler=sampler, qm=qm, ml=ml, bootstrap_set=2, parallel_samplers=2,
                target_queued_QM=2, save_h5_threshold=3)


def test_coordinator_restarts_after_a_crash_during_the_first_store(tmp_path, monkeypatch):
    with monkeypatch.context() as crash:
        crash_inside_store_data(crash)
        with pytest.raises(OSError, match="disk quota"):
            run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["current_h5_id"] == 0 and status["current_model_id"] == -1
    assert os.listdir(tmp_path / "h5") == []

    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    assert run.status["current_model_id"] >= 1
    assert stored_ids(tmp_path / "h5")
    assert not list((tmp_path / "h5").glob("*.partial"))


def test_coordinator_starts_with_a_partial_shard_left_by_a_crash(tmp_path, monkeypatch):
    (tmp_path / "h5").mkdir()
    (tmp_path / "h5" / "data-0000.h5.partial").write_bytes(b"half a shard")
    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    assert run.status["current_model_id"] >= 1
    assert sorted(os.listdir(tmp_path / "h5"))[0] == "data-0000.h5"
    assert not list((tmp_path / "h5").glob("*.partial"))


# --- The coordinator lock ---------------------------------------------------------------------------

HOLDER = """
import sys, time
from alframework.tools.files import hold_lock
hold_lock(sys.argv[1])
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
"""


def start_holder(lock, seconds=60):
    import subprocess
    import sys

    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock), str(seconds)], stdout=subprocess.PIPE,
                              text=True, cwd=Path(__file__).resolve().parents[1])
    assert holder.stdout.readline().strip() == "locked"
    return holder


def lock_is_free(lock):
    """Try the lock from a fresh process, which holds it only while checking."""
    import subprocess
    import sys

    check = subprocess.run([sys.executable, "-c", HOLDER, str(lock), "0"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1])
    return check.returncode == 0


def test_lock_refuses_a_second_holder_and_names_the_first(tmp_path):
    lock = tmp_path / "status.json.lock"
    holder = start_holder(lock)
    try:
        assert not lock_is_free(lock)
        with pytest.raises(RuntimeError, match=f"locked by host .*, process {holder.pid}"):
            files.hold_lock(lock)
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.parametrize("ending", ["exits", "is_killed"])
def test_lock_is_released_when_its_holder_ends(tmp_path, ending):
    import signal

    lock = tmp_path / "status.json.lock"
    holder = start_holder(lock, seconds=0.2 if ending == "exits" else 60)
    if ending == "is_killed":
        os.kill(holder.pid, signal.SIGKILL)
    holder.wait()
    assert lock_is_free(lock)


def test_coordinator_refuses_to_start_while_another_holds_the_run(tmp_path, monkeypatch):
    holder = start_holder(tmp_path / "status.json.lock")
    try:
        with pytest.raises(RuntimeError, match="Another ALF coordinator is using this run"):
            run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    finally:
        holder.kill()
        holder.wait()
    assert not (tmp_path / "runinfo").exists(), "a refused coordinator must not start Parsl"
    assert not (tmp_path / "status.json").exists()


# --- Held labels ----------------------------------------------------------------------------------

def test_held_labels_survive_a_new_process(tmp_path):
    from alframework.tools.restarts import HeldLabels

    held = HeldLabels(tmp_path / "held")
    held.hold([labeled("a"), labeled("b")], failed=3)
    held.hold([], failed=1)
    held.hold([labeled("c")])
    assert (held.successful(), held.finished()) == (3, 7)

    again = HeldLabels(tmp_path / "held")  # As after a restart: failures aren't carried over.
    assert (again.successful(), again.finished()) == (3, 3)
    assert again.molecule_ids() == {"a", "b", "c"}
    results, pieces = again.take()
    assert [result.get_moleculeid() for result in results] == ["a", "b", "c"]
    again.discard(pieces[:1])
    assert again.molecule_ids() == {"c"} and again.finished() == 1
    again.hold([labeled("d")])
    assert again.pieces() == ["held-00000001.pkl", "held-00000002.pkl"], "numbering continues after a discard"


def test_held_labels_directory_defaults_beside_h5_dir_and_never_inside_it(tmp_path):
    from alframework.tools.restarts import held_labels_directory

    assert held_labels_directory({"h5_dir": str(tmp_path / "h5store_teacher")}) == tmp_path / "h5store_teacher_held"
    assert held_labels_directory({"h5_dir": str(tmp_path / "h5"), "held_labels_dir": str(tmp_path / "x")}) \
        == tmp_path / "x"
    for inside in ("h5", "h5/held"):
        with pytest.raises(ValueError, match="must be outside h5_dir"):
            held_labels_directory({"h5_dir": str(tmp_path / "h5"), "held_labels_dir": str(tmp_path / inside)})


def test_next_bootstrap_ids_skips_held_ids_and_reaches_the_end():
    from alframework.tools.batching import bootstrap_molecule_ids, next_bootstrap_ids

    skip = {f"mol-boot-{index:010d}" for index in (0, 1, 3, 4, 5)}
    assert next_bootstrap_ids(0, 6, 2, True, skip) == (["mol-boot-0000000002"], 6)
    assert next_bootstrap_ids(6, 6, 2, True, skip) == ([], 6)
    assert next_bootstrap_ids(0, 6, 2, False, skip) == (["mol-boot-0000000002", "mol-boot-0000000006"], 7)
    assert bootstrap_molecule_ids(4, 6, 4, True) == ["mol-boot-0000000004", "mol-boot-0000000005"]


def test_ase_task_moves_aside_a_scratch_directory_left_by_an_earlier_attempt(tmp_path):
    from alframework.qm_interfaces.ase_calculator_interface import fresh_scratch_directory

    directory = tmp_path / "mol-boot-0000000003"
    directory.mkdir()
    (directory / "orca.out").write_text("from the crashed run")
    (tmp_path / "mol-boot-0000000003.previous-0").mkdir()
    fresh_scratch_directory(str(directory))
    assert list(directory.iterdir()) == []
    assert (tmp_path / "mol-boot-0000000003.previous-1" / "orca.out").read_text() == "from the crashed run"


# --- Crashing and restarting the real coordinator ----------------------------------------------------

def restart_tasks(tmp_path, built, release_training=None, qm_seconds=0.0, training_seconds=0.0):
    """Stand-ins that record every built ID. Training writes its model directory, and training 1 can wait."""
    import threading
    import time

    from parsl import python_app

    if release_training is None:  # Training waits only for tests that release it.
        release_training = threading.Event()
        release_training.set()

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        built.append(moleculeid)
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        time.sleep(qm_seconds)
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        (tmp_path / "models" / f"model-{current_training_id:04d}").mkdir(parents=True)
        if current_training_id == 1:
            release_training.wait(30)
        if current_training_id > 0:
            time.sleep(training_seconds)
        return [True, True], current_training_id

    return dict(builder=builder, sampler=sampler, qm=qm, ml=ml)


LOOP = dict(bootstrap_set=6, parallel_samplers=2, target_queued_QM=2, save_h5_threshold=3)


def held_ids(tmp_path):
    from alframework.tools.restarts import HeldLabels

    return HeldLabels(tmp_path / "h5_held").molecule_ids()


def assert_stored_once(tmp_path):
    ids = stored_ids(tmp_path / "h5")
    assert len(ids) == len(set(ids)), "a label was stored twice"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["lifetime_stored_labels"] == len(ids)
    return set(ids)


def test_restart_during_a_bootstrap_keeps_its_finished_labels(tmp_path, monkeypatch):
    built = []
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP,
                   stop_when=lambda status: len(held_ids(tmp_path)) >= 3)
    kept = held_ids(tmp_path)
    assert len(kept) >= 3 and not list((tmp_path / "h5").glob("*.h5"))

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP)
    with h5py.File(tmp_path / "h5" / "data-0000.h5") as bootstrap:
        bootstrap_ids = {name for group in bootstrap.values() for name in group["_id"].asstr()[()]}
    assert kept <= bootstrap_ids, "labels finished before the crash must be in the bootstrap"
    assert_stored_once(tmp_path)


def test_restarted_finite_bootstrap_rebuilds_exactly_what_it_lacks(tmp_path, monkeypatch, capsys):
    built = []
    finite = dict(LOOP, finite_bootstrap=True)
    # Stop once every ID has been handed out, while some are still being labeled. Before restarts
    # were reconciled, this stopped the next run with "Finite bootstrap exhausted".
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built, qm_seconds=0.05), **finite,
                   stop_when=lambda status: status["current_molecule_id"] >= 6 and len(held_ids(tmp_path)) < 6)
    kept = held_ids(tmp_path)
    assert 0 < len(kept) < 6

    first_run = len(built)
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **finite)
    assert "finite bootstrap" in capsys.readouterr().out
    expected = {f"mol-boot-{index:010d}" for index in range(6)}
    with h5py.File(tmp_path / "h5" / "data-0000.h5") as bootstrap:
        bootstrap_ids = sorted(name for group in bootstrap.values() for name in group["_id"].asstr()[()])
    assert bootstrap_ids == sorted(expected), "the bootstrap must hold exactly the IDs a run without a crash would"
    rebuilt = [name for name in built[first_run:] if name.startswith("mol-boot-")]
    assert set(rebuilt) == expected - kept


def test_labels_finished_during_training_survive_a_crash(tmp_path, monkeypatch):
    import threading

    built, release = [], threading.Event()

    def crash_while_training_with_labels_held(status):
        if status["current_training_id"] >= 2 and status["current_model_id"] == 0 and held_ids(tmp_path):
            release.set()  # Let the stand-in training thread end, as the real process would be gone.
            return True
        return False

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built, release), **LOOP,
                   stop_when=crash_while_training_with_labels_held)
    kept = held_ids(tmp_path)
    assert kept and not kept & set(stored_ids(tmp_path / "h5"))

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built, release), **LOOP,
                   stop_when=lambda status: status["current_h5_id"] >= 3)
    assert kept <= assert_stored_once(tmp_path)


def crash_saving_status(monkeypatch, when, settle=0.0):
    """Fail the first status write for which ``when(status)`` holds, as if the node died just before it.

    ``settle`` lets tasks already submitted run first, as they would on their own workers.
    """
    import time

    original = tools.write_status

    def write_or_crash(path, status):
        if when(status):
            time.sleep(settle)
            raise OSError("node lost")
        original(path, status)

    monkeypatch.setattr(tools, "write_status", write_or_crash)


def test_restart_after_a_crash_between_a_store_and_saving_status(tmp_path, monkeypatch, capsys):
    built = []
    with monkeypatch.context() as crash:
        crash_saving_status(crash, lambda status: status["current_h5_id"] >= 2)
        with pytest.raises(OSError, match="node lost"):
            run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["current_h5_id"] == 1 and (tmp_path / "h5" / "data-0001.h5").exists()
    assert (tmp_path / "models" / f"model-{status['current_training_id']:04d}").exists(), \
        "the training submitted with that store started under an ID status doesn't know about"

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP,
                   stop_when=lambda status: status["current_h5_id"] >= 3)
    output = capsys.readouterr().out
    assert "data-0001.h5 was stored before the crash" in output
    assert "already started" in output
    assert_stored_once(tmp_path)


def test_restart_after_a_crash_between_a_store_and_clearing_its_held_labels(tmp_path, monkeypatch):
    from alframework.tools.restarts import HeldLabels

    built = []
    with monkeypatch.context() as crash:
        def crash_on_second_store(self, names):
            if (tmp_path / "h5" / "data-0001.h5").exists():
                raise OSError("node lost")
            original(self, names)

        original = HeldLabels.discard
        crash.setattr(HeldLabels, "discard", crash_on_second_store)
        with pytest.raises(OSError, match="node lost"):
            run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP)
    assert held_ids(tmp_path), "the stored labels are still held"

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **LOOP,
                   stop_when=lambda status: status["current_h5_id"] >= 3)
    assert_stored_once(tmp_path)


# Several builders per loop, so one unsaved loop can use several IDs.
WIDE = dict(LOOP, parallel_samplers=4, target_queued_QM=4, save_h5_threshold=6)


def test_restart_never_reuses_a_molecule_id(tmp_path, monkeypatch):
    built = []
    with monkeypatch.context() as crash:
        # Status is saved once per loop, so IDs handed out in the loop that crashes are never saved.
        # Crash in a later loop with model 0, so the restart continues with the same ID prefix.
        last_saved = {}

        def loop_used_two_unsaved_ids(status):
            if last_saved.get("current_model_id") == 0 and status["current_model_id"] == 0 \
                    and status["current_molecule_id"] - last_saved["current_molecule_id"] >= 2:
                return True
            last_saved.update(status)
            return False

        crash_saving_status(crash, loop_used_two_unsaved_ids, settle=0.3)
        with pytest.raises(OSError, match="node lost"):
            run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **WIDE)
    saved = json.loads((tmp_path / "status.json").read_text())
    assert saved["current_model_id"] == 0
    unsaved = {f"mol-0000-{index:010d}" for index in range(saved["current_molecule_id"], saved["current_molecule_id"] + 2)}
    assert unsaved <= set(built), "the crash must come after builders used IDs that status doesn't know about"

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built), **WIDE,
                   stop_when=lambda status: status["current_h5_id"] >= 3)
    assert len(built) == len(set(built)), "a molecule ID was used twice"
    assert_stored_once(tmp_path)


def test_budget_stop_stores_the_labels_it_holds(tmp_path, monkeypatch, capsys):
    from alframework.tools.restarts import STORED_AT_BUDGET_STOP

    built = []
    # The last training runs while the labels it didn't include finish.
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, built, qm_seconds=0.2, training_seconds=1.0),
                   **dict(LOOP, maximum_stored_labels=8), stop_when=lambda status: False)
    assert not held_ids(tmp_path)
    ids = assert_stored_once(tmp_path)
    shards = sorted((tmp_path / "h5").glob("data-*.h5"))
    with h5py.File(shards[-1]) as handle:
        assert handle.attrs.get(STORED_AT_BUDGET_STOP) == "true"
        at_stop = {name for group in handle.values() for name in group["_id"].asstr()[()]}
    assert at_stop and len(shards) == 3, "bootstrap, one training shard, then the labels held at the stop"
    stopped = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Stopping:")]
    assert f"including {len(at_stop)} finished labels stored at the stop without training" in stopped[0]
    assert len(ids) > 8


def test_failed_labels_count_toward_the_next_store_as_before(tmp_path, monkeypatch):
    """Failed QM work counted toward save_h5_threshold while it waited in the queue, and still does."""
    from parsl import python_app

    built = []
    tasks = restart_tasks(tmp_path, built)

    @python_app(executors=["alf_QM_executor"])
    def fails_in_active_learning(molecule_object):
        if not molecule_object.get_moleculeid().startswith("mol-boot-"):
            raise RuntimeError("SCF did not converge")
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    # Counting only converged labels, the second store would never come.
    run_alf_driver(tmp_path, monkeypatch, **dict(tasks, qm=fails_in_active_learning), **LOOP, timeout=20,
                   stop_when=lambda status: status["current_h5_id"] >= 2)
    with h5py.File(tmp_path / "h5" / "data-0001.h5") as handle:
        second = [name for group in handle.values() for name in group["_id"].asstr()[()]]
    assert len(second) <= LOOP["save_h5_threshold"], "failures, not labels, filled the threshold"
    assert all(name.startswith("mol-boot-") for name in second), "only bootstrap labels that finished late"
