"""Batched labeling through the real ALF coordinator with local mock tasks."""
import h5py
import numpy as np
import pytest
from ase import Atoms
from parsl import python_app

from alframework.tools import labeling as labeling_module
from alframework.tools import tools as tool_module
from alframework.tools.batching import BatchBuffer, BatchedTaskQueue
from alframework.tools.labeling import LabelingCoordinator, labeling_batch_settings
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.driver import MODEL_PATTERN, loaded_parsl, run_alf_driver, stored_ids
from tests.helpers.fakes import FakeTask
from tests.helpers.molecules import WATER

FAILING_ID = "mol-boot-0000000001"
NATIVE_CALLS = []


def label(molecule_object):
    molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
    molecule_object.set_converged_flag(True)
    return molecule_object


@python_app(executors=["alf_QM_executor"])
def single_qm(molecule_object, QM_config):
    assert isinstance(molecule_object, MoleculesObject)
    if molecule_object.get_moleculeid() == FAILING_ID:
        raise RuntimeError("mock SCF failure")
    return label(molecule_object)


@python_app(executors=["alf_QM_executor"])
def native_qm(molecule_objects, QM_config):
    NATIVE_CALLS.append([mol.get_moleculeid() for mol in molecule_objects])
    labeled = [label(mol) for mol in molecule_objects]
    for mol in labeled:
        if mol.get_moleculeid() == FAILING_ID:
            mol.set_converged_flag(False)
    return labeled


@pytest.mark.parametrize("mode,batch_size", [(None, None), ("sequential", 2), ("sequential", 5),
                                             ("native", 2), ("native", 5)])
def test_coordinator_labels_bootstraps_stores_and_retrains(tmp_path, monkeypatch, mode, batch_size):
    # Waiting structures count as queued QM, so with target_queued_QM=2 a batch of 5
    # cannot fill during bootstrap; bootstrap can only finish through timed-out partial batches.
    training_calls, sampler_calls, packed_calls = [], [], []
    native_calls = NATIVE_CALLS
    native_calls.clear()

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object, current_model_id):
        sampler_calls.append(molecule_object.get_moleculeid())
        molecule_object.update_metadata({"sampled_with": current_model_id})
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        training_calls.append((current_training_id, stored_ids(tmp_path / "h5")))
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True] * 4, current_training_id

    original_pack = labeling_module.label_sequentially

    def recording_pack(molecule_objects, label_one, task_inputs):
        packed_calls.append([mol.get_moleculeid() for mol in molecule_objects])
        return original_pack(molecule_objects, label_one, task_inputs)

    monkeypatch.setattr(labeling_module, "label_sequentially", recording_pack)

    qm_config = {}
    if mode is not None:
        qm_config["batching"] = {"mode": mode, "batch_size": batch_size, "max_wait_seconds": 0.05}

    run = run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=sampler, ml=ml,
                         qm=native_qm if mode == "native" else single_qm, qm_config=qm_config, bootstrap_set=4,
                         parallel_samplers=2, target_queued_QM=2, save_h5_threshold=3, maximum_completed_QM=8)

    bootstrap_ids = training_calls[0][1]
    assert training_calls[0][0] == 0 and len(bootstrap_ids) >= 3
    assert all(value.startswith("mol-boot-") for value in bootstrap_ids)
    assert len(bootstrap_ids) == len(set(bootstrap_ids))
    retrained_ids = training_calls[1][1]
    assert training_calls[1][0] == 1 and any(value.startswith("mol-0000-") for value in retrained_ids)
    assert set(value for value in retrained_ids if value.startswith("mol-0000-")) <= set(sampler_calls)
    assert (tmp_path / "h5/data-0001.h5").exists()

    stored = stored_ids(tmp_path / "h5")
    if mode is None:
        # The single-structure labeler fails one whole task, as before.
        assert run.status["lifetime_failed_QM_tasks"] >= 1
        assert not packed_calls and not native_calls
        assert any(seconds == 60 for seconds in run.sleeps)
    else:
        # The failed member is recorded but does not discard its batch-mates' labels.
        assert run.status["lifetime_failed_QM_tasks"] == 0
        batches = packed_calls if mode == "sequential" else native_calls
        assert not (packed_calls and native_calls)
        assert batches and all(1 <= len(batch) <= batch_size for batch in batches)
        if batch_size == 5:
            assert all(len(batch) < 5 for batch in batches if batch[0].startswith("mol-boot-"))
            # The waiting partial batch shortens the coordinator's sleep.
            assert any(seconds < 60 for seconds in run.sleeps)
        labeled = [value for batch in batches for value in batch]
        assert len(labeled) == len(set(labeled))
        failed_batch = next(batch for batch in batches if FAILING_ID in batch)
        assert set(failed_batch) - {FAILING_ID} <= set(stored)
    assert FAILING_ID not in stored
    with h5py.File(tmp_path / "h5/data-0000.h5") as handle:
        group = next(iter(handle.values()))
        assert group["species"].asstr()[()].tolist() == ["H", "H", "O"]


def test_sequential_packing_keeps_successful_labels():
    molecules = [MoleculesObject(WATER.copy(), name) for name in ("mol-boot-0000000000", FAILING_ID, "c")]
    labeled = labeling_module.label_sequentially(molecules, single_qm.func, {"QM_config": {}})
    assert [mol.get_moleculeid() for mol in labeled] == ["mol-boot-0000000000", FAILING_ID, "c"]
    assert [mol.check_convergence() for mol in labeled] == [True, False, True]
    assert "mock SCF failure" in labeled[1].get_metadata()["labeling_error"]
    assert labeled[0].get_results()["energy"] == -1.0


def test_labeling_settings_require_explicit_mode():
    assert labeling_batch_settings({}) is None
    assert labeling_batch_settings({"batching": {"mode": "native", "batch_size": 2}})["mode"] == "native"
    for bad, reason in (({"batch_size": 2}, "mode"), ({"mode": "packed", "batch_size": 2}, "mode"),
                        ({"mode": "native"}, "batch_size"), ([], "dictionary")):
        with pytest.raises(ValueError, match=reason):
            labeling_batch_settings({"batching": bad})


def test_coordinator_rejects_labeler_signature_mismatch():
    with pytest.raises(ValueError, match="molecule_objects"):
        LabelingCoordinator(single_qm, "mock.qm", {"batching": {"mode": "native", "batch_size": 2}})
    with pytest.raises(ValueError, match="molecule_object"):
        LabelingCoordinator(native_qm, "mock.qm", {"batching": {"mode": "sequential", "batch_size": 2}})


def test_full_only_requires_queue_target_that_can_fill_a_batch():
    settings = {"batching": {"mode": "native", "batch_size": 4, "partial_policy": "full_only"}}
    with pytest.raises(ValueError, match="target_queued_QM"):
        LabelingCoordinator(native_qm, "mock.qm", settings, {"target_queued_QM": 3})
    assert LabelingCoordinator(native_qm, "mock.qm", settings, {"target_queued_QM": 4}).batched


def test_unbatched_coordinator_keeps_ordinary_queue():
    coordinator = LabelingCoordinator(single_qm, "mock.qm", {})
    assert type(coordinator.queue) is tool_module.parsl_task_queue
    assert coordinator.poll_delay(60) == 60
    assert coordinator.submit_ready([]) == 0


def test_batched_queue_counts_structures_and_flattens_results():
    buffer = BatchBuffer(4)
    queue = BatchedTaskQueue(buffer)
    first, second = MoleculesObject(WATER.copy(), "a"), MoleculesObject(WATER.copy(), "b")
    queue.add_task(FakeTask("exec_done", result=[first, [second]]), width=2)
    queue.add_task(FakeTask("failed", exception=RuntimeError("worker lost")), width=3)
    queue.add_task(FakeTask("running", running=True, done=False), width=4)
    buffer.add(MoleculesObject(WATER.copy(), "waiting"))

    assert queue.get_number() == 10 and queue.get_task_number() == 3
    assert queue.get_completed_number() == 5
    assert queue.get_exec_done_number() == 2
    assert queue.get_failed_number() == 3
    assert queue.get_running_number() == 4
    assert queue.get_queued_number() == 1

    results, failed = queue.get_task_results()
    assert [mol.get_moleculeid() for mol in results] == ["a", "b"]
    assert failed == 1 and queue.failed_structures == 3
    assert queue.get_number() == 5 and queue.get_task_number() == 1


@pytest.mark.parametrize('width', [1, 2])
def test_real_label_coordinator_submits_timeout_tails_without_flush(tmp_path, width):
    @python_app(executors=['tail'])
    def label(molecule_object):
        return molecule_object

    with loaded_parsl(tmp_path / 'parsl', 'tail') as dfk:
        coordinator = LabelingCoordinator(label, 'test.label', dict(batching=dict(
            mode='sequential', batch_size=4, partial_policy='timeout', max_wait_seconds=0)))
        for i in range(width):
            coordinator.submit(MoleculesObject(Atoms('H'), str(i)), [])
        assert coordinator.submit_ready([]) == 1
        for task in list(dfk.tasks.values()):
            task['app_fu'].result()
        results, failures = coordinator.queue.get_task_results()
        assert failures == 0 and len(results) == width
