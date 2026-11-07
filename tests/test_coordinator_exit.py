"""How the coordinator ends: every exit must stop Parsl's interchange and release its allocations.

Runs ``python -m alframework`` as a subprocess, because Python's exit handling can only be seen from
outside the process. The tasks are small stand-ins. Tests that kill training use real Parsl workers on
this machine, so they need ``pgrep``, and they can't show that Slurm releases a real allocation.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import parsl
import pytest
from parsl import python_app

from alframework.tools import tools as tool_module
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import release_allocations_at_exit, stop_if_executor_failed
from tests.helpers.driver import run_alf_driver
from tests.helpers.molecules import WATER

REPOSITORY = Path(__file__).resolve().parents[1]
CLEANED_UP = "Terminated executors"
NOT_CLEANED_UP = "Python is exiting with a DFK still running"

HARNESS = """
import os
from parsl import join_app, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from alframework.tools import ensemble_training

root = os.environ['HARNESS_ROOT']
threads = [ThreadPoolExecutor(label=label, max_threads=1) for label in ('alf_builder_executor', 'alf_sampler_executor', 'alf_QM_executor')]
local = Config(run_dir=root + '/runinfo', executors=threads + [ThreadPoolExecutor(label='alf_ML_executor', max_threads=1)])
real_workers = Config(run_dir=root + '/runinfo', retries=0, strategy_period=1, executors=threads + [HighThroughputExecutor(
    label='alf_ML_executor', address='127.0.0.1', max_workers_per_node=2, cores_per_worker=1, heartbeat_period=2,
    heartbeat_threshold=10, provider=LocalProvider(init_blocks=1, min_blocks=0, max_blocks=1))])

@python_app(executors=['alf_builder_executor'])
def builder(moleculeid): raise RuntimeError('unused')

@python_app(executors=['alf_sampler_executor'])
def sampler(molecule_object): raise RuntimeError('unused')

@python_app(executors=['alf_QM_executor'])
def qm(molecule_object): raise RuntimeError('unused')

@python_app(executors=['alf_ML_executor'])
def trains(current_training_id): return [True, True], current_training_id

@python_app(executors=['alf_ML_executor'])
def fails_to_train(current_training_id): return [True, False], current_training_id

@python_app(executors=['alf_ML_executor'])
def crashes(current_training_id): raise RuntimeError('mock training error')

@python_app(executors=['alf_ML_executor'])
def slow_member(root):
    import pathlib, time
    pathlib.Path(root, 'member_started').touch()
    time.sleep(300)
    return True

@join_app
def slow_multinode(current_training_id):
    return ensemble_training.train_ensemble_across_nodes(
        lambda member, attempt: slow_member(root), 2, root, current_training_id, 0, 'cpu', 0, None)
"""


def start_coordinator(tmp_path, ml_task, parsl_config="local", arguments=(), trained_labels=True):
    """Start a coordinator. ``trained_labels`` skips the bootstrap by pretending data is already stored."""
    (tmp_path / "exit_harness.py").write_text(HARNESS)
    master = {
        "master_directory": str(tmp_path),
        "builder_task": "exit_harness.builder",
        "sampler_task": "exit_harness.sampler",
        "QM_task": "exit_harness.qm",
        "ML_task": f"exit_harness.{ml_task}",
        "parsl_configuration": f"exit_harness.{parsl_config}",
        "builder_config_path": "builder.json",
        "sampler_config_path": "sampler.json",
        "QM_config_path": "qm.json",
        "ML_config_path": "ml.json",
        "status_path": "status.json",
        "model_path": "models/model-{:04d}",
        "h5_path": "h5/data-{:04d}.h5",
        "properties_list": {"energy": ["energy", "system", 1.0]},
        "minimum_QM": 0,
    }
    (tmp_path / "master.json").write_text(json.dumps(master))
    for name in ("builder", "sampler", "qm", "ml"):
        (tmp_path / f"{name}.json").write_text("{}")
    if trained_labels:
        status = {"current_training_id": 0, "current_model_id": -1, "current_h5_id": 1, "current_molecule_id": 0,
                  "lifetime_failed_builder_tasks": 0, "lifetime_failed_sampler_tasks": 0,
                  "lifetime_failed_ML_tasks": 0, "lifetime_failed_QM_tasks": 0}
        (tmp_path / "status.json").write_text(json.dumps(status))
    environment = dict(os.environ, HARNESS_ROOT=str(tmp_path), PYTHONPATH=f"{tmp_path}{os.pathsep}{REPOSITORY}",
                       PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    return subprocess.Popen(
        [sys.executable, "-m", "alframework", "--master", str(tmp_path / "master.json"), *arguments],
        cwd=tmp_path, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def parsl_log(tmp_path):
    return next(tmp_path.glob("runinfo/*/parsl.log")).read_text()


def assert_cleaned_up(tmp_path):
    log = parsl_log(tmp_path)
    assert CLEANED_UP in log
    assert NOT_CLEANED_UP not in log


@pytest.mark.parametrize(
    "ml_task,arguments,expected_code,expected_output",
    [
        ("trains", ["--test_ml"], 0, "New Model: 0000"),
        ("fails_to_train", [], 0, "Bootstrap network failed to train"),
        ("crashes", [], 1, "mock training error"),
    ],
    ids=["test_ml", "failed_bootstrap_training", "uncaught_error"],
)
def test_coordinator_exit_cleans_up_parsl(tmp_path, ml_task, arguments, expected_code, expected_output):
    coordinator = start_coordinator(tmp_path, ml_task, arguments=arguments)
    output, _ = coordinator.communicate(timeout=120)
    assert coordinator.returncode == expected_code, output
    assert expected_output in output
    assert_cleaned_up(tmp_path)


def _process_alive(pid):
    state = subprocess.run(["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _workers_for(tmp_path):
    rows = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
    return [row.split()[0] for row in rows if str(tmp_path) in row and "process_worker_pool" in row]


@pytest.fixture
def real_coordinator(tmp_path):
    """Yield a function that starts a coordinator on real workers, and always stop what it started."""
    if shutil.which("pgrep") is None:
        pytest.skip("needs pgrep")
    started = SimpleNamespace(coordinator=None, children=[])

    def start(ml_task):
        started.coordinator = start_coordinator(tmp_path, ml_task, "real_workers")
        deadline = time.monotonic() + 120
        while not (tmp_path / "member_started").exists():
            assert started.coordinator.poll() is None, started.coordinator.communicate()[0]
            assert time.monotonic() < deadline, "training never started"
            time.sleep(0.2)
        started.children = subprocess.run(["pgrep", "-P", str(started.coordinator.pid)],
                                          capture_output=True, text=True).stdout.split()
        assert started.children, "the interchange should be a child of the coordinator"
        return started

    yield start
    if started.coordinator is not None and started.coordinator.poll() is None:
        started.coordinator.kill()
    for pid in [*started.children, *_workers_for(tmp_path)]:
        subprocess.run(["kill", "-9", pid], capture_output=True)


def test_coordinator_sigterm_during_multinode_training_releases_everything(tmp_path, real_coordinator):
    """Slurm sends SIGTERM for ``scancel`` and at walltime. Training would otherwise last 300 s."""
    started = real_coordinator("slow_multinode")
    started.coordinator.send_signal(signal.SIGTERM)
    started.coordinator.communicate(timeout=60)
    assert started.coordinator.returncode == 128 + signal.SIGTERM
    assert_cleaned_up(tmp_path)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (
        any(_process_alive(pid) for pid in started.children) or _workers_for(tmp_path)
    ):
        time.sleep(0.5)
    assert not any(_process_alive(pid) for pid in started.children), "the interchange outlived the coordinator"
    assert not _workers_for(tmp_path), "the worker pool outlived the coordinator"


# The handler itself.


class FakeDataFlowKernel:
    def __init__(self, cleanup_called=False, error=None):
        self.cleanup_called, self.error, self.calls = cleanup_called, error, 0

    def cleanup(self):
        self.calls += 1
        if self.error:
            raise self.error


@pytest.fixture
def registered_handler(monkeypatch):
    handlers = []
    monkeypatch.setattr("atexit.register", handlers.append)
    original = signal.getsignal(signal.SIGTERM)
    yield handlers
    signal.signal(signal.SIGTERM, original)


def test_release_allocations_runs_cleanup_at_exit(registered_handler):
    dfk = FakeDataFlowKernel()
    release_allocations_at_exit(dfk)
    registered_handler[0]()
    assert dfk.calls == 1


def test_release_allocations_does_not_repeat_a_finished_cleanup(registered_handler):
    """``parsl.dfk().cleanup()`` raises if it runs twice, and a budget stop has already called it."""
    dfk = FakeDataFlowKernel(cleanup_called=True)
    release_allocations_at_exit(dfk)
    registered_handler[0]()
    assert dfk.calls == 0


def test_release_allocations_reports_a_failed_cleanup_without_raising(registered_handler, capsys):
    release_allocations_at_exit(FakeDataFlowKernel(error=RuntimeError("scheduler is down")))
    registered_handler[0]()
    assert "Could not release Parsl allocations: RuntimeError: scheduler is down" in capsys.readouterr().out


def test_release_allocations_turns_sigterm_into_a_normal_exit(registered_handler):
    release_allocations_at_exit(FakeDataFlowKernel())
    with pytest.raises(SystemExit) as stopped:
        signal.raise_signal(signal.SIGTERM)
    assert stopped.value.code == 128 + signal.SIGTERM


# A dead executor.


def dead(reason="block failed"):
    return SimpleNamespace(bad_state_is_set=True, executor_exception=RuntimeError(reason))


def test_stop_if_executor_failed_names_the_executor_and_its_cause():
    executors = {"alf_sampler_executor": SimpleNamespace(bad_state_is_set=False), "alf_ML_executor": dead("node failure")}
    with pytest.raises(RuntimeError, match="alf_ML_executor.*node failure"):
        stop_if_executor_failed(SimpleNamespace(executors=executors))


@pytest.mark.parametrize(
    "executors",
    [
        {"alf_ML_executor": SimpleNamespace(bad_state_is_set=False)},
        {"_parsl_internal": SimpleNamespace()},
        {"alf_QM_executor": SimpleNamespace(bad_state_is_set=False), "alf_QM_standby_executor": dead()},
    ],
    ids=["healthy", "thread_executor_has_no_state", "dead_standby"],
)
def test_stop_if_executor_failed_lets_a_working_run_continue(executors):
    stop_if_executor_failed(SimpleNamespace(executors=executors))


def coordinator_tasks():
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
                target_queued_QM=2, save_h5_threshold=3, maximum_completed_QM=8)


@pytest.mark.parametrize(
    "ready,trained",
    [(lambda status: True, False), (lambda status: status["current_model_id"] >= 0, True)],
    ids=["bootstrap_loop", "active_learning_loop"],
)
def test_coordinator_stops_with_status_saved_when_an_executor_dies(tmp_path, monkeypatch, ready, trained):
    real_check, saved = stop_if_executor_failed, []

    def check_after_comparing_status(dfk):
        status_on_disk = json.loads((tmp_path / "status.json").read_text())
        saved.append(status_on_disk == sys._getframe(1).f_globals["status"])
        real_check(dfk)

    monkeypatch.setattr(tool_module, "stop_if_executor_failed", check_after_comparing_status)

    def kill_ml_executor_when_ready(status):
        if ready(status):
            executor = parsl.dfk().executors["alf_ML_executor"]
            executor.bad_state_is_set, executor.executor_exception = True, RuntimeError("all blocks failed")
        return False

    with pytest.raises(RuntimeError, match="alf_ML_executor.*all blocks failed"):
        run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks(), stop_when=kill_ml_executor_when_ready)
    assert saved and all(saved), "status must be saved before stopping"
    final = json.loads((tmp_path / "status.json").read_text())
    assert (final["current_model_id"] >= 0) == trained, "the check must fire in the loop where the executor died"
