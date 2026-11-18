"""A failed task frees what its frames held before the worker takes its next task."""

import gc
import traceback
import weakref

import pytest

from alframework.samplers import batched_mlmd
from alframework.samplers.batched_mlmd import ASEEngine
from alframework.tools.task_failure_cleanup import release_failed_task, releases_memory_on_failure
from alframework.tools.molecules import MoleculesObject
from tests.helpers.fakes import spring_ensemble as fake_ensemble
from tests.helpers.molecules import SPRING_WATER


class Big:
    """Stands in for a failed evaluation's tensors."""


@pytest.fixture
def no_cycle_collector():
    """Without automatic collection, only reference counting frees objects, as between two collector runs."""
    enabled = gc.isenabled()
    gc.disable()
    yield
    if enabled:
        gc.enable()


def _evaluate(found):
    tensors = Big()
    found.append(weakref.ref(tensors))
    try:
        raise MemoryError("out of memory while evaluating")
    except MemoryError as first:
        held = first  # A frame that refers to its own exception: a reference cycle.
        raise RuntimeError("the step failed") from held


def _task(found):
    _evaluate(found)


def test_an_unreleased_failure_keeps_its_frames_alive(no_cycle_collector):
    found = []
    with pytest.raises(RuntimeError) as caught:
        _task(found)
    assert found[0]() is not None, "the traceback still holds the failed evaluation's locals"
    del caught


def test_a_released_failure_frees_its_frames_and_still_prints(no_cycle_collector):
    found = []
    with pytest.raises(RuntimeError) as caught:
        releases_memory_on_failure(_task)(found)
    assert found[0]() is None
    printed = "".join(traceback.format_exception(caught.value))
    assert "_evaluate" in printed and "out of memory while evaluating" in printed and "the step failed" in printed


def test_release_follows_the_whole_chain(no_cycle_collector):
    found = []
    try:
        _task(found)
    except RuntimeError as error:
        release_failed_task(error)
    assert found[0]() is None


class BrokenStepEngine(ASEEngine):
    """A batch step that fails for the whole task, as a GPU running out of memory does."""

    def step(self, indices, n):
        self.tensors = Big()
        raise MemoryError("out of memory in the batch step")


def test_a_failed_sampler_task_frees_its_engine(no_cycle_collector, monkeypatch):
    engines = []

    def make_engine(name, *args, device=None, **options):
        engine = BrokenStepEngine(*args, **options)
        engines.append(weakref.ref(engine))
        return engine

    monkeypatch.setattr(batched_mlmd, "sampling_device", lambda config, gpus_per_node=None: "cpu")
    monkeypatch.setattr(batched_mlmd, "worker_ensemble", lambda config, directory, device: fake_ensemble())
    monkeypatch.setattr(batched_mlmd, "make_engine", make_engine)
    config = dict(dt=0.5, maxt=0.05, Ncheck=5, distcut=0.5, min_time=0.0, Fscut=100.0, Escut=0.012, random_seed=5,
                  srt_temp=[300.0, 300.0], end_temp=[300.0, 300.0], amp_temp=[0.0, 0.0], per_temp=[0.01, 0.01],
                  amp_dens=None, per_dens=None, end_dens=None)
    molecules = [MoleculesObject(SPRING_WATER.copy(), f"mol-0000-{index:010d}") for index in range(2)]
    # As on a worker: Parsl's wrapper returns the failure, to be raised again on the coordinator.
    returned = batched_mlmd.mlmd_trajectories_task.func(molecules, config, "models-{}", 0)
    with pytest.raises(MemoryError, match="batch step"):
        returned.reraise()
    [engine] = engines
    assert engine() is None, "the next task would start with this task's engine still allocated"


# Every task that uses a GPU in the worker process. Members train in child processes, so they are not here.
GPU_TASKS = [
    ("alframework.samplers.batched_mlmd", "mlmd_trajectories_task"),
    ("alframework.samplers.mlmd_sampling", "simple_mlmd_sampling_task"),
    ("alframework.samplers.udd_sampling", "simple_udd_sampling_task"),
    ("alframework.samplers.reactive_sampler", "reactive_sampling"),
    ("alframework.qm_interfaces.teacher_interface", "teacher_task"),
    ("alframework.qm_interfaces.teacher_interface", "teacher_batch_task"),
    ("alframework.qm_interfaces.pyseqm_interface", "pyseqm_task"),
    ("alframework.qm_interfaces.pyseqm_interface", "pyseqm_excited_state_task"),
    ("alframework.qm_interfaces.pyseqm_interface", "pyseqm_excited_state_batch_task"),
    ("alframework.qm_interfaces.pyscf", "pyscf_task"),
    ("alframework.qm_interfaces.pyscf.gpu4pyscf", "gpu4pyscf_excited_state_task"),
]
RELEASING = releases_memory_on_failure(lambda: None).__code__


@pytest.mark.parametrize("module, name", GPU_TASKS)
def test_every_in_process_gpu_task_releases_on_failure(module, name):
    import importlib
    try:
        task = getattr(importlib.import_module(module), name)
    except ImportError as missing:
        pytest.skip(f"{module} needs {missing.name}")
    function, chain = task.func, []
    while function is not None:
        chain.append(function.__code__)
        function = getattr(function, "__wrapped__", None)
    assert RELEASING in chain


def test_release_empties_the_cupy_pools(monkeypatch):
    import sys
    from types import SimpleNamespace
    freed = []
    pool = lambda name: SimpleNamespace(free_all_blocks=lambda: freed.append(name))  # noqa: E731
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(get_default_memory_pool=lambda: pool("device"),
                                                             get_default_pinned_memory_pool=lambda: pool("pinned")))
    release_failed_task(RuntimeError("SCF failed"))
    assert freed == ["device", "pinned"]
