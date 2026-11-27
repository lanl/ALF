"""Run ALF's real coordinator in-process, with local stand-ins for its four tasks.

The coordinator (``python -m alframework``) is a script with an endless loop, so
these helpers run it with ``runpy`` and stop it from inside its own ``time.sleep``.
"""

import json
import runpy
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import parsl
from parsl.config import Config
from parsl.errors import NoDataFlowKernelError
from parsl.executors import ThreadPoolExecutor

from alframework.tools import config as workflow_config
from alframework.tools.molecules import MoleculesObject
from tests.helpers.molecules import WATER

EXECUTORS = ("alf_builder_executor", "alf_sampler_executor", "alf_QM_executor", "alf_ML_executor")
MODEL_PATTERN = "models/model-{:04d}"
PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}


def stored_ids(h5_dir):
    """Return the IDs of every structure saved in a directory of HDF5 shards."""
    ids = []
    for path in sorted(Path(h5_dir).glob("data-*.h5")):
        with h5py.File(path) as handle:
            for group in handle.values():
                ids.extend(value.decode() for value in group["_id"][()])
    return ids


def local_parsl_config(run_dir, executors=EXECUTORS):
    return Config(
        executors=[ThreadPoolExecutor(label=label, max_threads=2) for label in executors],
        strategy="none",
        run_dir=str(run_dir),
    )


@contextmanager
def loaded_parsl(run_dir, *executors):
    """Load Parsl with local thread executors, and always unload it afterward."""
    dfk = parsl.load(local_parsl_config(run_dir, executors))
    try:
        yield dfk
    finally:
        dfk.cleanup()
        parsl.clear()


class _DriverStopped(Exception):
    """Raised from the driver's sleep, because its loop has no other way to end."""


def retrained_once(status):
    return status["current_model_id"] >= 1


def run_alf_driver(
    tmp_path,
    monkeypatch,
    *,
    builder,
    sampler,
    qm,
    ml,
    builder_config=None,
    sampler_config=None,
    qm_config=None,
    ml_config=None,
    executors=EXECUTORS,
    stop_when=retrained_once,
    timeout=120,
    arguments=(),
    **master_settings,
):
    """Run the coordinator in ``tmp_path`` until ``stop_when(status)`` holds.

    ``builder``, ``sampler``, ``qm`` and ``ml`` are Parsl apps, and ``master_settings``
    are added to the master configuration. Models and HDF5 shards go to
    ``tmp_path / MODEL_PATTERN`` and ``tmp_path / "h5"``. Returns a namespace with the
    final ``status``, the driver's requested ``sleeps``, and ``sampler_queue_sizes``.
    Errors raised by the driver propagate.
    """
    monkeypatch.chdir(tmp_path)
    modules = {
        "mock.config": local_parsl_config(tmp_path / "runinfo", executors),
        "mock.builder": builder,
        "mock.sampler": sampler,
        "mock.qm": qm,
        "mock.ml": ml,
    }
    monkeypatch.setattr(workflow_config, "load_module_from_string", modules.__getitem__)

    master = dict(
        master_directory="pwd",
        builder_task="mock.builder",
        sampler_task="mock.sampler",
        QM_task="mock.qm",
        ML_task="mock.ml",
        parsl_configuration="mock.config",
        builder_config_path="builder.json",
        sampler_config_path="sampler.json",
        QM_config_path="qm.json",
        ML_config_path="ml.json",
        status_path="status.json",
        model_path=str(tmp_path / MODEL_PATTERN),
        h5_path=str(tmp_path / "h5/data-{:04d}.h5"),
        properties_list=PROPERTIES,
        minimum_QM=0,
        **master_settings,
    )
    configs = {"master": master, "builder": builder_config or {}, "sampler": sampler_config or {},
               "qm": qm_config or {}, "ml": ml_config or {}}
    for name, values in configs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(values))
    monkeypatch.setattr(sys, "argv", ["alframework", "--master", str(tmp_path / "master.json"), *arguments])

    run = SimpleNamespace(status={}, sleeps=[], sampler_queue_sizes=[], coordinator_cleaned_up=False)
    original_sleep = time.sleep
    deadline = time.monotonic() + timeout

    def driver_sleep(seconds):
        frame = sys._getframe(1)
        if frame.f_globals.get("__name__") != "__main__":
            return original_sleep(seconds)  # Parsl's own threads keep their normal timing.
        run.sleeps.append(seconds)
        run.status = dict(frame.f_globals["status"])
        if "sampler_task_queue" in frame.f_globals:
            run.sampler_queue_sizes.append(frame.f_globals["sampler_task_queue"].get_number())
        if stop_when(run.status):
            raise _DriverStopped
        assert time.monotonic() < deadline, f"coordinator did not finish within {timeout} s"
        original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(time, "sleep", driver_sleep)
    sigterm_handler = signal.getsignal(signal.SIGTERM)  # The coordinator sets its own.
    try:
        runpy.run_module("alframework", run_name="__main__")
    except _DriverStopped:
        pass
    finally:
        signal.signal(signal.SIGTERM, sigterm_handler)
        try:
            parsl.dfk().cleanup()
        except NoDataFlowKernelError:
            run.coordinator_cleaned_up = True  # It finished on its own.
        finally:
            parsl.clear()
    return run


# A short loop for coordinator tests, and stand-in tasks for it.
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
