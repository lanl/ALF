"""Multi-node ensemble training: one app per member, retries, GPU choice, and the coordinator's wait.

Uses real Parsl with thread executors and the real coordinator, with trainers faked, so it runs
without Torch or HIPPYNN. GPU choice uses a fake Torch. The one real training test needs Torch and
HIPPYNN and skips without them. Nothing here shows two real nodes; that needs a real allocation.
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from parsl import python_app

from alframework.ml_interfaces import hippynn_interface as ml
from alframework.tools import ensemble_training
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.driver import MODEL_PATTERN, loaded_parsl, run_alf_driver
from tests.helpers.molecules import WATER

PROPERTIES = {
    "sE0": ["sE0", "system", 1.0],
    "F0": ["F0", "atomic", 1.0],
    "sE1": ["sE1", "system", 1.0],
    "F1": ["F1", "atomic", 1.0],
}
EXCITED_CONFIG = {
    "cell_key": None,
    "train_forces": True,
    "network_choice": 0,
    "valid_size": 0.2,
    "test_size": 0.2,
    "device_string": "cpu",
    "exports": {"pdf": False, "png": False, "csv": False},
}


def member_and_training(arg_dict):
    directory = Path(arg_dict["model_dir"])
    return int(directory.name.split("-")[1]), int(directory.parent.name.split("-")[1])


def fake_members(monkeypatch, behavior=lambda member, attempt, training: True):
    """Replace the child process that trains a member, and record each call."""
    calls = []

    def child(function, arg_dict, attempt):
        member, training = member_and_training(arg_dict)
        calls.append((member, attempt))
        Path(arg_dict["model_dir"]).mkdir(parents=True)
        return behavior(member, attempt, training)

    monkeypatch.setattr(ensemble_training, "run_member_in_child", child)
    return calls


@pytest.fixture(params=["ground_state", "excited_state"])
def multinode_task(request, tmp_path):
    """Each multi-node task with arguments for three members, and one shard to find."""
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    (h5_dir / "data-0000.h5").touch()
    arguments = dict(h5_dir=str(h5_dir), model_path=str(tmp_path / "models/model-{:04d}"),
                     current_training_id=3, gpus_per_node=4)
    if request.param == "ground_state":
        return lambda **settings: ml.train_HIPPYNN_ensemble_multinode_task(
            ML_config={"n_models": 3, "device_string": "from_multiprocessing"}, **arguments, **settings)
    return lambda **settings: ml.train_excited_state_HIPPYNN_ensemble_multinode_task(
        ML_config={**EXCITED_CONFIG, "n_models": 3}, properties_list=PROPERTIES, **arguments, **settings)


def test_multinode_training_runs_on_the_coordinator_and_members_on_the_ml_executor():
    for task in (ml.train_HIPPYNN_ensemble_multinode_task, ml.train_excited_state_HIPPYNN_ensemble_multinode_task):
        assert task.executors == ["_parsl_internal"]
    for member_task in (ml._ground_state_member_task, ml._excited_state_member_task):
        assert member_task.executors == ["alf_ML_executor"]


def test_multinode_training_returns_members_in_order_after_the_last_one_finishes(tmp_path, monkeypatch, multinode_task):
    release = threading.Event()
    finished = []

    def behavior(member, attempt, training):
        if member == 0:
            assert release.wait(30)  # Member 0 finishes last.
        finished.append(member)
        return member != 1

    fake_members(monkeypatch, behavior)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        future = multinode_task(member_retries=0)
        while len(finished) < 2:
            threading.Event().wait(0.01)
        assert not future.done(), "the task must wait for member 0"
        release.set()
        assert future.result(timeout=30) == ([True, False, True], 3)
    assert finished[-1] == 0


def test_multinode_training_retries_only_the_failed_member(tmp_path, monkeypatch, multinode_task, capfd):
    def behavior(member, attempt, training):
        if member == 1 and attempt == 1:
            raise RuntimeError("lost worker")
        return True

    calls = fake_members(monkeypatch, behavior)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        assert multinode_task().result(timeout=30) == ([True, True, True], 3)
    assert sorted(calls) == [(0, 1), (1, 1), (1, 2), (2, 1)]
    ensemble = tmp_path / "models/model-0003"
    assert (ensemble / "failed-model-01-attempt-1").is_dir() and (ensemble / "model-01").is_dir()
    output = capfd.readouterr().out
    assert "member 1 attempt 1 failed: RuntimeError: lost worker" in output
    assert re.search(r"Training 0003: \d/3 members finished, 1 retrying, 0 failed", output)
    assert output.rstrip().endswith("Training 0003: 3/3 members finished, 0 retrying, 0 failed")


def test_multinode_training_gives_up_after_the_retry_limit(tmp_path, monkeypatch, multinode_task):
    calls = fake_members(monkeypatch, lambda member, attempt, training: member != 2)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        assert multinode_task(member_retries=2).result(timeout=30) == ([True, True, False], 3)
    assert sorted(calls) == [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3)]
    ensemble = tmp_path / "models/model-0003"
    assert sorted(path.name for path in ensemble.glob("failed-*")) == [
        "failed-model-02-attempt-1", "failed-model-02-attempt-2"]
    assert (ensemble / "model-02").is_dir(), "the last attempt stays for inspection"


def test_multinode_training_with_no_retries_runs_each_member_once(tmp_path, monkeypatch, multinode_task):
    calls = fake_members(monkeypatch, lambda member, attempt, training: member != 2)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        assert multinode_task(member_retries=0).result(timeout=30) == ([True, True, False], 3)
    assert sorted(calls) == [(0, 1), (1, 1), (2, 1)]
    assert not list((tmp_path / "models/model-0003").glob("failed-*"))


@pytest.mark.parametrize("retries", [-1, 1.5, True, "2"], ids=["negative", "float", "boolean", "string"])
def test_multinode_training_rejects_invalid_member_retries(tmp_path, monkeypatch, multinode_task, retries):
    calls = fake_members(monkeypatch)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        with pytest.raises(ValueError, match="member_retries"):
            multinode_task(member_retries=retries).result(timeout=30)
    assert not calls


def test_multinode_excited_training_rejects_a_bad_configuration_once(tmp_path, monkeypatch):
    calls = fake_members(monkeypatch)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        future = ml.train_excited_state_HIPPYNN_ensemble_multinode_task(
            ML_config={**EXCITED_CONFIG, "n_models": 3}, h5_dir=str(tmp_path), current_training_id=0,
            model_path=str(tmp_path / "model-{:04d}"), gpus_per_node=4, properties_list={})
        with pytest.raises(ValueError):
            future.result(timeout=30)
    assert not calls


@pytest.mark.parametrize("unsafe", [{"remove_existing": True}, {"h5_test_dir": "test"}], ids=["remove_existing", "h5_test_dir"])
def test_multinode_excited_training_rejects_unsafe_options(tmp_path, monkeypatch, unsafe):
    calls = fake_members(monkeypatch)
    (tmp_path / "data.h5").touch()
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        future = ml.train_excited_state_HIPPYNN_ensemble_multinode_task(
            ML_config={**EXCITED_CONFIG, "n_models": 3}, h5_dir=str(tmp_path), current_training_id=0,
            model_path=str(tmp_path / "model-{:04d}"), gpus_per_node=4, properties_list=PROPERTIES, **unsafe)
        with pytest.raises(ValueError, match=next(iter(unsafe))):
            future.result(timeout=30)
    assert not calls


def test_multinode_excited_training_refuses_a_nonempty_ensemble(tmp_path, monkeypatch):
    calls = fake_members(monkeypatch)
    (tmp_path / "data.h5").touch()
    (tmp_path / "model-0000").mkdir()
    (tmp_path / "model-0000/old").touch()
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        future = ml.train_excited_state_HIPPYNN_ensemble_multinode_task(
            ML_config={**EXCITED_CONFIG, "n_models": 3}, h5_dir=str(tmp_path), current_training_id=0,
            model_path=str(tmp_path / "model-{:04d}"), gpus_per_node=4, properties_list=PROPERTIES)
        with pytest.raises(FileExistsError):
            future.result(timeout=30)
    assert not calls


def test_multinode_member_waits_for_the_shards_the_coordinator_listed(tmp_path, monkeypatch, multinode_task):
    fake_members(monkeypatch)
    seen = []
    monkeypatch.setattr(ensemble_training, "wait_for_shards", seen.append)
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        multinode_task().result(timeout=30)
    assert seen == [[str(tmp_path / "h5/data-0000.h5")]] * 3


# Train_members, with plain futures.


def finished_future(value=None, exception=None):
    future = Future()
    future.set_exception(exception) if exception else future.set_result(value)
    return future


def test_train_members_keeps_going_when_a_failed_member_cannot_be_archived(tmp_path, monkeypatch, capsys):
    def no_archive(*args):
        raise OSError("read-only")

    monkeypatch.setattr(ensemble_training, "archive_failed_member", no_archive)
    outcome = ensemble_training.train_members(lambda member, attempt: finished_future(member == 0), 2, tmp_path, 1)
    assert outcome == [True, False]
    assert "could not retry member 1: read-only" in capsys.readouterr().out


def test_train_members_keeps_going_when_a_retry_cannot_be_submitted(tmp_path, capsys):
    def submit(member, attempt):
        if attempt > 1:
            raise RuntimeError("no executor")
        return finished_future(member == 0)

    assert ensemble_training.train_members(submit, 2, tmp_path, 1) == [True, False]
    assert "could not retry member 1: no executor" in capsys.readouterr().out


def test_archived_member_is_invisible_to_ensemble_loaders(tmp_path, monkeypatch):
    for name in ("model-00", "model-01"):
        (tmp_path / name).mkdir()
    ensemble_training.archive_failed_member(tmp_path, 1, 1)
    (tmp_path / "model-01").mkdir()
    ensemble_training.archive_failed_member(tmp_path, 1, 2)
    (tmp_path / "model-01").mkdir()
    (tmp_path / "placement-model-01-attempt-1.json").write_text("{}")
    monkeypatch.setattr(ml, "HIPNN_ASE_calculator", lambda directory, device: Path(directory).name)
    assert ml.HIPNN_ASE_load_ensemble(str(tmp_path), device="cpu") == ["model-00", "model-01"]
    assert (tmp_path / "failed-model-01-attempt-2").is_dir()


def test_archiving_a_member_that_never_started_is_harmless(tmp_path):
    ensemble_training.archive_failed_member(tmp_path, 0, 1)
    assert not list(tmp_path.iterdir())


# GPU choice.


@pytest.fixture
def four_gpus(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    return fake_torch


def test_member_device_gives_each_worker_on_a_node_its_own_gpu(four_gpus, monkeypatch):
    devices = []
    for rank in range(4):
        monkeypatch.setenv("PARSL_WORKER_RANK", str(rank))
        devices.append(ensemble_training.member_device("from_multiprocessing", 4))
    assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_member_device_does_not_depend_on_the_member_number(four_gpus, monkeypatch, tmp_path):
    """Members 0 and 4 can land on one node, and must not share a GPU."""
    seen = {}

    def train(**arguments):
        seen[Path(arguments["model_dir"]).name] = arguments["device_string"]
        Path(arguments["model_dir"]).mkdir(parents=True)
        (Path(arguments["model_dir"]) / "training_log.txt").write_text("Training complete\n")

    monkeypatch.setattr(ml, "train_HIPNN_model", train)
    for member, rank in ((0, 0), (4, 1)):
        monkeypatch.setenv("PARSL_WORKER_RANK", str(rank))
        assert ml._train_ground_state_member(
            {"model_dir": str(tmp_path / f"model-{member:02d}"), "from_multiprocessing_nGPU": 4}, 1)
    assert seen == {"model-00": "cuda:0", "model-04": "cuda:1"}


def test_member_device_respects_the_gpus_per_node_cap(four_gpus, monkeypatch):
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    assert ensemble_training.member_device("cuda", 2) == "cuda:1"
    assert ensemble_training.member_device("cuda", 0) == "cuda:3"


def test_member_device_leaves_a_worker_mask_alone(four_gpus, monkeypatch):
    """With ``available_accelerators``, each worker sees one GPU as its device 0."""
    four_gpus.cuda.device_count = lambda: 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    assert ensemble_training.member_device("from_multiprocessing", 4) == "cuda:0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"


def test_member_device_allows_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert ensemble_training.member_device("CPU", 4) == "cpu"


@pytest.mark.parametrize("device", ["0", "cuda:1", "2,3"], ids=["mask", "fixed_gpu", "mask_list"])
def test_member_device_rejects_a_fixed_gpu(device, four_gpus):
    with pytest.raises(ValueError, match="device_string"):
        ensemble_training.member_device(device, 4)


def test_ground_state_trainer_accepts_a_device_already_chosen(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert ml._ensemble_member_device("cuda:3", 4) == "cuda:3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"


def test_excited_member_trains_on_the_device_its_worker_chose(four_gpus, monkeypatch, tmp_path):
    monkeypatch.setenv("PARSL_WORKER_RANK", "2")
    seen = {}

    def worker(arguments):
        seen.update(arguments["ML_config"])
        return {"ok": True, "payload": {"model_dir": arguments["model_dir"], "model_id": 5}}

    monkeypatch.setattr(ml, "_excited_state_training_worker", worker)
    monkeypatch.setattr(ml, "_excited_state_member_complete", lambda result: True)
    monkeypatch.setattr(ml, "_validate_completed_member", lambda *args: None)
    arguments = {"model_id": 5, "model_dir": str(tmp_path / "model-05"), "properties_list": PROPERTIES,
                 "from_multiprocessing_nGPU": 4,
                 "ML_config": {**EXCITED_CONFIG, "n_models": 8, "device_string": "from_multiprocessing"}}
    assert ml._train_excited_state_member(arguments, 1) is True
    assert seen["device_string"] == "cuda:2"
    assert arguments["ML_config"]["device_string"] == "from_multiprocessing", "the caller's config must not change"


def test_ground_state_member_without_a_complete_log_counts_as_failed(four_gpus, monkeypatch, tmp_path):
    def train(**arguments):
        Path(arguments["model_dir"]).mkdir(parents=True)
        (Path(arguments["model_dir"]) / "training_log.txt").write_text("Stopped before convergence\n")

    monkeypatch.setattr(ml, "train_HIPNN_model", train)
    model_dir = tmp_path / "model-00"
    assert ml._train_ground_state_member({"model_dir": str(model_dir), "from_multiprocessing_nGPU": 4}, 1) is False
    assert json.loads((tmp_path / "placement-model-00-attempt-1.json").read_text())["completed"] is False


def test_excited_member_whose_training_failed_counts_as_failed(four_gpus, monkeypatch, tmp_path):
    def worker(arguments):
        return {"ok": False, "payload": {"model_dir": arguments["model_dir"], "model_id": 0, "error": "bad member"}}

    monkeypatch.setattr(ml, "_excited_state_training_worker", worker)
    model_dir = tmp_path / "model-00"
    arguments = {"model_id": 0, "model_dir": str(model_dir), "properties_list": PROPERTIES,
                 "from_multiprocessing_nGPU": 4, "ML_config": {**EXCITED_CONFIG, "n_models": 2}}
    assert ml._train_excited_state_member(arguments, 1) is False
    assert json.loads((model_dir / "training_error.json").read_text())["error"] == "bad member"


def test_excited_member_that_fails_its_reload_counts_as_failed(four_gpus, monkeypatch, tmp_path):
    def bad_reload(*args):
        raise RuntimeError("no such output")

    monkeypatch.setattr(ml, "_excited_state_training_worker",
                        lambda arguments: {"ok": True, "payload": {"model_dir": arguments["model_dir"], "model_id": 0}})
    monkeypatch.setattr(ml, "_excited_state_member_complete", lambda result: True)
    monkeypatch.setattr(ml, "_validate_completed_member", bad_reload)
    model_dir = tmp_path / "model-00"
    model_dir.mkdir()
    arguments = {"model_id": 0, "model_dir": str(model_dir), "properties_list": PROPERTIES,
                 "from_multiprocessing_nGPU": 4, "ML_config": {**EXCITED_CONFIG, "n_models": 2}}
    assert ml._train_excited_state_member(arguments, 1) is False
    assert json.loads((model_dir / "training_error.json").read_text())["error"] == "no such output"


# Capacity.


def executor(workers, nodes, blocks):
    return SimpleNamespace(workers_per_node=workers, provider=SimpleNamespace(nodes_per_block=nodes, max_blocks=blocks))


@pytest.mark.parametrize("nodes,blocks", [(1, 10), (10, 1), (2, 5)], ids=["one_node_blocks", "one_large_block", "mixed"])
def test_check_capacity_is_silent_when_every_member_has_a_slot(nodes, blocks, capsys):
    ensemble_training.check_capacity(40, 4, "from_multiprocessing", executor(4, nodes, blocks))
    assert capsys.readouterr().out == ""


def test_check_capacity_warns_when_members_must_wait(capsys):
    ensemble_training.check_capacity(8, 4, "from_multiprocessing", executor(1, 1, 2))
    assert "2 slots for 8 ensemble members" in capsys.readouterr().out


def test_check_capacity_refuses_members_sharing_a_gpu():
    with pytest.raises(ValueError, match="share a GPU"):
        ensemble_training.check_capacity(8, 4, "from_multiprocessing", executor(8, 1, 1))


@pytest.mark.parametrize("gpus,device", [(0, "from_multiprocessing"), (4, "cpu")], ids=["no_gpu_cap", "cpu"])
def test_check_capacity_allows_more_workers_when_no_gpu_is_shared(gpus, device, capsys):
    ensemble_training.check_capacity(8, gpus, device, executor(8, 1, 1))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "unknown",
    [None, SimpleNamespace(), executor(float("inf"), 1, 1)],
    ids=["no_executor", "thread_executor", "unlimited_workers"],
)
def test_check_capacity_skips_executors_it_cannot_measure(unknown, capsys):
    ensemble_training.check_capacity(8, 4, "from_multiprocessing", unknown)
    assert capsys.readouterr().out == ""


# Shards, placement, and the child process.


def test_wait_for_shards_waits_for_a_directory_listing_to_catch_up(tmp_path, monkeypatch):
    shard = tmp_path / "data-0001.h5"
    shard.touch()
    real_listdir, listings = os.listdir, []

    def stale_listdir(path):
        listings.append(path)
        return [] if len(listings) < 3 else real_listdir(path)

    monkeypatch.setattr(ensemble_training.os, "listdir", stale_listdir)
    ensemble_training.wait_for_shards([str(shard)], timeout=30, poll=0.001)
    assert len(listings) == 3


def test_wait_for_shards_raises_after_its_timeout(tmp_path):
    with pytest.raises(FileNotFoundError, match="1 of 1 HDF5 shards"):
        ensemble_training.wait_for_shards([str(tmp_path / "data-0001.h5")], timeout=0, poll=0.001)


def test_member_placement_records_a_finished_member(tmp_path, monkeypatch):
    monkeypatch.setenv("PARSL_WORKER_RANK", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    path = tmp_path / "placement-model-03-attempt-2.json"
    with ensemble_training.member_placement(tmp_path / "model-03", 2, "cuda:1") as record:
        assert json.loads(path.read_text())["finished"] is None
        record["completed"] = True
    written = json.loads(path.read_text())
    assert (written["member"], written["attempt"], written["device"]) == ("model-03", 2, "cuda:1")
    assert (written["worker_rank"], written["slurm_job"]) == ("2", "77")
    assert written["visibility"]["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert written["completed"] is True and written["finished"] >= written["started"]
    assert not (tmp_path / "model-03").exists(), "trainers refuse a nonempty member directory"


def test_member_placement_records_a_member_that_raised(tmp_path):
    with pytest.raises(RuntimeError):
        with ensemble_training.member_placement(tmp_path / "model-00", 1, "cpu"):
            raise RuntimeError("out of memory")
    written = json.loads((tmp_path / "placement-model-00-attempt-1.json").read_text())
    assert written["completed"] is False and written["finished"] is not None


def test_run_member_in_child_raises_when_the_child_dies():
    with pytest.raises(BrokenProcessPool):
        ensemble_training.run_member_in_child(os._exit, 1)


# The real coordinator.


def coordinator_tasks(tmp_path):
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

    return dict(builder=builder, sampler=sampler, qm=qm, ml=ml.train_HIPPYNN_ensemble_multinode_task,
                ml_config={"n_models": 3, "device_string": "from_multiprocessing"}, gpus_per_node=4,
                bootstrap_set=2, parallel_samplers=2, target_queued_QM=2, save_h5_threshold=3,
                maximum_completed_QM=8)


def test_coordinator_promotes_only_after_a_retried_member_finishes(tmp_path, monkeypatch):
    release, polls = threading.Event(), []

    def behavior(member, attempt, training):
        if training == 1 and member == 0:
            if attempt == 1:
                raise RuntimeError("lost worker")
            assert release.wait(60)
        return True

    calls = fake_members(monkeypatch, behavior)

    def stop_when(status):
        polls.append((status["current_model_id"], status["current_h5_id"], release.is_set()))
        if status["current_training_id"] == 2 and sum(1 for poll in polls if poll[1] == 2) >= 5:
            release.set()
        return status["current_model_id"] >= 1

    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks(tmp_path), stop_when=stop_when)

    assert run.status["current_model_id"] == 1
    assert all(released for model, _, released in polls if model >= 1), "promoted before the last member finished"
    assert {h5 for model, h5, released in polls if model == 0 and not released and h5 >= 2} == {2}, \
        "no store may happen while members train"
    assert calls.count((0, 2)) == 1 and (1, 2) not in calls and (2, 2) not in calls
    assert (tmp_path / MODEL_PATTERN.format(1) / "failed-model-00-attempt-1").is_dir()


def test_coordinator_keeps_running_when_a_member_runs_out_of_retries(tmp_path, monkeypatch):
    fake_members(monkeypatch, lambda member, attempt, training: not (training == 1 and member == 2))
    models = []

    def stop_when(status):
        models.append(status["current_model_id"])
        return status["current_model_id"] >= 2

    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks(tmp_path), stop_when=stop_when, member_retries=0)

    assert 1 not in models, "an incomplete ensemble must not be promoted"
    assert run.status["current_model_id"] == 2 and run.status["lifetime_failed_ML_tasks"] == 0


def test_coordinator_waits_for_a_retried_member_during_the_bootstrap(tmp_path, monkeypatch):
    calls = fake_members(monkeypatch, lambda member, attempt, training: not (training == 0 and member == 1 and attempt == 1))
    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks(tmp_path),
                         stop_when=lambda status: status["current_model_id"] >= 0)
    assert run.status["current_model_id"] == 0
    assert (1, 2) in calls


# Real training.


def test_multinode_excited_training_trains_members_that_reload(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("hippynn")
    data = tmp_path / "h5"
    data.mkdir()
    coordinates = np.repeat(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), 12, axis=0)
    coordinates[:, 1, 0] += np.linspace(0.0, 0.1, 12)
    energies = 0.01 * (coordinates**2).sum(axis=(1, 2))
    with h5py.File(data / "data-0000.h5", "w") as handle:
        group = handle.create_group("water")
        group["species"] = np.array(["O", "H", "H"], dtype="S")
        group["coordinates"] = coordinates
        group["_id"] = np.array([f"frame-{i}" for i in range(12)], dtype="S")
        group["topology_atom_ids"] = np.tile(np.arange(3), (12, 1))
        for state in range(2):
            group[f"sE{state}"] = energies + 0.1 * state
            group[f"F{state}"] = -0.02 * coordinates
    config = {**EXCITED_CONFIG, "n_models": 2, "max_epochs": 2, "batch_size": 4, "eval_batch_size": 4,
              "termination_patience": 10, "torch_threads": 1, "fraction_train_eval": 1.0,
              "network_params": dict(possible_species=[0, 1, 8], n_features=4, n_sensitivities=4,
                                     n_interaction_layers=1, n_atom_layers=1, dist_soft_min=0.4,
                                     dist_soft_max=2.0, dist_hard_max=3.0)}
    model_path = str(tmp_path / "models/model-{:04d}")
    with loaded_parsl(tmp_path / "runinfo", "alf_ML_executor"):
        outcome = ml.train_excited_state_HIPPYNN_ensemble_multinode_task(
            ML_config=config, h5_dir=str(data), model_path=model_path, current_training_id=0,
            gpus_per_node=0, properties_list=PROPERTIES, member_retries=0).result(timeout=300)
    assert outcome == ([True, True], 0)

    ensemble = tmp_path / "models/model-0000"
    for member in ("model-00", "model-01"):
        assert (ensemble / member / "best_model.pt").is_file()
        placement = json.loads((ensemble / f"placement-{member}-attempt-1.json").read_text())
        assert placement["device"] == "cpu" and placement["completed"] is True
    (ensemble / "failed-model-00-attempt-1").mkdir()
    evaluator = ml.load_hippynn_excited_state_ensemble(str(ensemble), "cpu", PROPERTIES)
    assert evaluator.n_models == 2
