"""The per-task timeline: one JSON line per task when ALF_TASK_TIMELINE_DIR is set, nothing otherwise."""

import inspect
import json
import os

import pytest
from parsl import python_app

from alframework.tools.task_timeline import ENV, read_timeline, stage_spans, timed
from alframework.tools.tools import build_input_dict


@timed("sampler")
def _sample(molecules, scale=1):
    return [m * scale for m in molecules]


@timed("QM")
def _broken(molecule):
    raise ValueError("no SCF convergence")


def test_off_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert _sample([1, 2], scale=3) == [3, 6]
    assert list(tmp_path.iterdir()) == []


def test_one_record_per_task(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "timeline"))
    monkeypatch.setenv("SLURM_JOB_ID", "5919222")
    monkeypatch.setenv("PARSL_WORKER_BLOCK_ID", "0")
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    assert _sample([1, 2, 3]) == [1, 2, 3]
    assert _sample([4]) == [4]
    [path] = (tmp_path / "timeline").iterdir()
    assert path.name == f"{os.uname().nodename}-{os.getpid()}.jsonl", "one file per worker process"
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["stage"] == "sampler" and first["task"] == "_sample" and first["items"] == 3
    assert second["items"] == 1 and first["outcome"] == "ok"
    assert first["end"] >= first["start"] and first["seconds"] == pytest.approx(first["end"] - first["start"])
    assert (first["slurm_job"], first["parsl_block"], first["parsl_rank"]) == ("5919222", "0", "3")


def test_a_failed_task_is_recorded_and_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(ValueError, match="SCF"):
        _broken("mol-1")
    [record] = read_timeline(str(tmp_path))
    assert record["outcome"] == "ValueError" and record["stage"] == "QM"


def test_an_unwritable_timeline_never_fails_the_task(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv(ENV, str(blocker / "timeline"))
    assert _sample([1]) == [1]


def test_stage_spans(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    _sample([1, 2])
    _sample([3])
    with pytest.raises(ValueError):
        _broken("m")
    spans = stage_spans(read_timeline(str(tmp_path)))
    assert spans["sampler"]["tasks"] == 2 and spans["sampler"]["items"] == 3 and spans["sampler"]["failed"] == 0
    assert spans["QM"]["failed"] == 1
    assert spans["sampler"]["wall_seconds"] >= 0 and spans["sampler"]["task_seconds"] >= 0


def test_alf_tasks_keep_their_signatures_for_build_input_dict():
    from alframework.ml_interfaces.hippynn_interface import train_HIPPYNN_ensemble_task
    from alframework.samplers.batched_mlmd import mlmd_trajectories_task

    task = mlmd_trajectories_task.func
    assert list(inspect.signature(task).parameters)[:3] == ["molecule_objects", "sampler_config", "model_path"]
    inputs = build_input_dict(task, [{"sampler_config": {"a": 1}, "model_path": "m", "current_model_id": 0}])
    assert inputs == {"sampler_config": {"a": 1}, "model_path": "m", "current_model_id": 0}
    assert "gpus_per_node" in inspect.signature(train_HIPPYNN_ensemble_task.func).parameters


@python_app(executors=["alf_sampler_executor"])
@timed("sampler")
def _app(molecules):
    return len(molecules)


def test_through_parsl(tmp_path, monkeypatch):
    from tests.helpers.driver import loaded_parsl

    monkeypatch.setenv(ENV, str(tmp_path / "timeline"))
    with loaded_parsl(tmp_path / "runinfo", "alf_sampler_executor"):
        assert [_app([1, 2]).result(timeout=60), _app([3]).result(timeout=60)] == [2, 1]
    records = read_timeline(str(tmp_path / "timeline"))
    assert [r["items"] for r in records] == [2, 1] and {r["task"] for r in records} == {"_app"}


@timed("sampler")
def _outer(molecules):
    return _sample(molecules, scale=2)


@timed("builder")
def _builder(moleculeid=None, moleculeids=None, builder_config=None):
    return list(moleculeids or [moleculeid])


def test_a_task_calling_another_task_is_recorded_once(tmp_path, monkeypatch):
    # As UDD sampling calls MLMD sampling's function.
    monkeypatch.setenv(ENV, str(tmp_path))
    assert _outer([1, 2]) == [2, 4]
    [record] = read_timeline(str(tmp_path))
    assert record["task"] == "_outer" and record["items"] == 2
    _sample([3])
    assert [r["task"] for r in read_timeline(str(tmp_path))] == ["_outer", "_sample"], "recording resumes after"


def test_builders_called_by_keyword_count_their_molecule_ids(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    _builder(moleculeid="mol-0", moleculeids=["mol-0", "mol-1", "mol-2"], builder_config={})
    _builder(moleculeid="mol-3", builder_config={})
    assert [r["items"] for r in read_timeline(str(tmp_path))] == [3, 1]


def test_a_partial_line_does_not_hide_the_other_records(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path))
    _sample([1])
    [path] = tmp_path.iterdir()
    with open(path, "a") as handle:
        handle.write('{"stage": "sampler", "start": 1')  # a worker killed mid-write
    (tmp_path / "other.jsonl").write_text("[1, 2]\n")
    assert [r["task"] for r in read_timeline(str(tmp_path))] == ["_sample"]


TIMED_MODULES = [
    "alframework.builders.builders", "alframework.builders.rdkit_molecule", "alframework.builders.external_geometry",
    "alframework.builders.rdkit_condensed_phase", "alframework.builders.defect_crystal_builder",
    "alframework.builders.reactive_builder", "alframework.samplers.batched_mlmd", "alframework.samplers.mlmd_sampling",
    "alframework.samplers.udd_sampling", "alframework.samplers.reactive_sampler",
    "alframework.qm_interfaces.teacher_interface", "alframework.qm_interfaces.ase_calculator_interface",
    "alframework.qm_interfaces.orca5_interface", "alframework.qm_interfaces.qchem_DFT_interface",
    "alframework.ml_interfaces.hippynn_interface",
]


def _timed_apps():
    import importlib
    from parsl.app.python import PythonApp
    apps = []
    for name in TIMED_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        apps += [(f"{name}.{key}", value) for key, value in vars(module).items()
                 if isinstance(value, PythonApp) and hasattr(value.func, "__wrapped__")]
    return apps


def test_timed_tasks_serialize_for_remote_workers():
    # A worker gets each task through Parsl's serializer, which copies the timed wrapper by value; anything
    # unpicklable it refers to (a thread-local, a lock) fails every task on a real executor.
    from parsl.serialize import pack_apply_message
    apps = _timed_apps()
    assert len(apps) >= 10, [name for name, _ in apps]
    failed = []
    for name, app in apps:
        try:
            pack_apply_message(app.func, (), {})
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failed, failed


def test_a_deserialized_task_records_once(tmp_path, monkeypatch):
    from parsl.serialize import pack_apply_message, unpack_apply_message
    monkeypatch.setenv(ENV, str(tmp_path))
    function, args, kwargs = unpack_apply_message(pack_apply_message(_outer, ([1, 2],), {}))
    assert function(*args, **kwargs) == [2, 4]
    assert [r["task"] for r in read_timeline(str(tmp_path))] == ["_outer"]
