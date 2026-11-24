"""Opt-in batched reference labeling for the ALF coordinator.

Labeling stays behind ALF's ``QM_task`` import string. Without a ``batching``
block in the QM configuration, :class:`LabelingCoordinator` submits one
ordinary ``QM_task`` per structure, exactly as before. With one, structures
are grouped by :class:`alframework.tools.batching.BatchBuffer` and submitted in
one of two explicitly configured modes:

``native``
    The labeler evaluates several structures in one call. Its task accepts
    ``molecule_objects`` (a list) and returns the labeled ``MoleculesObject``
    instances, marking unsuccessful members unconverged.

``sequential``
    Job packing for an existing single-structure labeler (``molecule_object``).
    One Parsl task on the labeler's executors calls it once per member. A
    member that raises is marked unconverged and recorded in its metadata;
    the other members' labels are kept.

Example ``QM_config`` entry::

    "batching": {"mode": "sequential", "batch_size": 8,
                 "partial_policy": "timeout", "max_wait_seconds": 5.0}

Batching settings are read once at startup. Waiting structures are held in
memory and are not restored after a restart.
"""

import inspect
import logging
from collections import Counter

from parsl import python_app

from alframework.tools.batching import BatchedSubmitter, StartupSettings, batching_settings, flatten_molecule_output
from alframework.tools.capabilities import validate_task_config
from alframework.tools.config import build_input_dict
from alframework.tools.sampled_frames import SampledFrame, label_sampled_frames

logger = logging.getLogger(__name__)

LABELING_MODES = ("native", "sequential")


def labeling_mode_for(qm_task):
    """The labeling mode a QM task's signature calls for: ``native`` if it takes ``molecule_objects``,
    ``sequential`` if it takes ``molecule_object``, else ``None``."""
    parameters = inspect.signature(getattr(qm_task, "func", qm_task)).parameters
    if "molecule_objects" in parameters:
        return "native"
    if "molecule_object" in parameters:
        return "sequential"
    return None


def labeling_batch_settings(QM_config, qm_task=None):
    """Return validated labeling batch settings, or ``None`` when not selected.

    Without ``batching.mode``, the mode comes from ``qm_task``'s signature (:func:`labeling_mode_for`), and
    ``inferred_mode`` is true; without a task either, the mode is required.
    """
    block = (QM_config or {}).get("batching")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("QM batching must be a dictionary.")
    block = dict(block)
    mode = block.pop("mode", None)
    inferred = mode is None and qm_task is not None
    if inferred:
        mode = labeling_mode_for(qm_task)
    if mode not in LABELING_MODES:
        raise ValueError("QM batching.mode must be 'native' (the labeler accepts molecule_objects) "
                         "or 'sequential' (an existing single-structure labeler is called per member); "
                         f"received {mode!r}.")
    settings = batching_settings({"batching": block})
    settings["mode"] = mode
    settings["inferred_mode"] = inferred
    return settings


def label_sequentially(molecule_objects, label_one, task_inputs):
    """Label each member with a single-structure labeler function, keeping successes."""
    from parsl.app.errors import RemoteExceptionWrapper

    labeled = []
    for molecule_object in molecule_objects:
        try:
            output = label_one(molecule_object=molecule_object, **task_inputs)
            # A python_app's func returns, rather than raises, its wrapped exception.
            if isinstance(output, RemoteExceptionWrapper):
                output.reraise()
            labeled.extend(flatten_molecule_output(output))
        except Exception as exc:
            logger.warning("Labeling failed for %s", molecule_object.get_moleculeid(), exc_info=True)
            molecule_object.set_converged_flag(False)
            molecule_object.update_metadata({"labeling_error": "{}: {}".format(type(exc).__name__, exc)})
            labeled.append(molecule_object)
    return labeled


class LabelingCoordinator(BatchedSubmitter):
    """Submit reference-labeling work for the ALF coordinator.

    Args:
        qm_task: The loaded ``QM_task`` Parsl app.
        qm_task_path (str): Its import string (``master_config['QM_task']``).
        QM_config (dict): QM configuration; batching is selected by its
            ``batching`` block.
        master_config (dict): Used to check that ``full_only`` batches can fill.

    ``queue`` replaces ``QM_task_queue``. It is an ordinary
    ``ParslTaskQueue`` when batching is not selected and a
    :class:`BatchedTaskQueue` (counting structures) otherwise.
    """

    def __init__(self, qm_task, qm_task_path, QM_config, master_config=None, sampled_frame_handoff=False,
                 label_handoff=None):
        settings = labeling_batch_settings(QM_config, qm_task)
        self._reference_task = None
        # data_by_reference labels: write_label_part's keywords, for the QM workers.
        self._label_parts = label_handoff
        for handoff, name in ((sampled_frame_handoff, 'sampler_frames'), (label_handoff, 'labels')):
            if handoff and (settings is None or settings['mode'] != 'native'):
                raise ValueError(f'data_by_reference {name} requires native QM batching.')
        if sampled_frame_handoff or label_handoff:
            self._reference_task = python_app(executors=qm_task.executors)(label_sampled_frames)
        if settings is not None:
            mode = settings["mode"]
            required = "molecule_objects" if mode == "native" else "molecule_object"
            if required not in inspect.signature(qm_task.func).parameters:
                raise ValueError("QM batching.mode '{}' requires QM_task {} to accept '{}'.".format(
                    mode, qm_task_path, required))
            target = (master_config or {}).get("target_queued_QM")
            if settings["partial_policy"] == "full_only" and target is not None \
                    and target < settings["batch_size"]:
                # Waiting structures count as queued, so builders would stop before a batch fills.
                raise ValueError("QM batching with partial_policy 'full_only' requires "
                                 "target_queued_QM >= batching.batch_size.")
            if mode == "sequential":
                self._packed_task = python_app(executors=qm_task.executors)(label_sequentially)
        # A labeler can check its own configuration, so that a mistake stops ALF at launch.
        validate_task_config(qm_task, QM_config, master_config)
        super().__init__(qm_task, settings)
        if label_handoff:
            from alframework.tools.label_parts import LabelReceipt
            from alframework.tools.molecules import MoleculesObject
            self.queue.output_types = (MoleculesObject, LabelReceipt)
        self._startup_settings = StartupSettings(QM_config, ("batching",), "QM")
        self.qm_task = qm_task
        self.qm_task_path = qm_task_path

    def describe(self):
        if not self.batched:
            return "one structure per QM task"
        how = " (mode from the QM task)" if self.settings.get("inferred_mode") else ""
        return "{mode} batches of up to {batch_size} structures ({partial_policy}, {max_wait_seconds:g} s)".format(
            **self.settings) + how

    def take_finished_with_references(self, max_items):
        """Count completed references by file, including inputs omitted from a partial output."""
        succeeded, failed = Counter(), Counter()

        def record(inputs, success):
            (succeeded if success else failed).update(frame.path for frame in inputs if isinstance(frame, SampledFrame))

        results, failed_tasks = self.queue.get_task_results(max_items=max_items, on_finished_inputs=record)
        return results, failed_tasks, succeeded, failed

    def _group_task(self, group, input_dicts):
        if self.settings["mode"] == "native":
            if self._reference_task is not None and (
                    self._label_parts is not None or any(isinstance(frame, SampledFrame) for frame in group)):
                task_input = build_input_dict(self.qm_task.func, [{"molecule_objects": group}, *input_dicts],
                                              raise_on_fail=True)
                task_input.pop('molecule_objects')
                return self._reference_task(group, self.qm_task_path, task_input, self._label_parts)
            return super()._group_task(group, input_dicts)
        task_input = build_input_dict(self.qm_task.func, [{"molecule_object": None}, *input_dicts],
                                      raise_on_fail=True)
        task_input.pop("molecule_object")
        return self._packed_task(group, self.qm_task.func, task_input)
