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

from parsl import python_app

from alframework.tools.batching import BatchBuffer, BatchedTaskQueue, batching_settings
from alframework.tools.batching import flatten_molecule_output, poll_delay
from alframework.tools.tools import build_input_dict, parsl_task_queue

logger = logging.getLogger(__name__)

LABELING_MODES = ("native", "sequential")


def labeling_batch_settings(QM_config):
    """Return validated labeling batch settings, or ``None`` when not selected."""
    block = (QM_config or {}).get("batching")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("QM batching must be a dictionary.")
    block = dict(block)
    mode = block.pop("mode", None)
    if mode not in LABELING_MODES:
        raise ValueError("QM batching.mode must be 'native' (the labeler accepts molecule_objects) "
                         "or 'sequential' (an existing single-structure labeler is called per member); "
                         f"received {mode!r}.")
    settings = batching_settings({"batching": block})
    settings["mode"] = mode
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


class LabelingCoordinator:
    """Submit reference-labeling work for the ALF coordinator.

    Args:
        qm_task: The loaded ``QM_task`` Parsl app.
        qm_task_path (str): Its import string (``master_config['QM_task']``).
        QM_config (dict): QM configuration; batching is selected by its
            ``batching`` block.
        master_config (dict): Used to check that ``full_only`` batches can fill.

    ``queue`` replaces ``QM_task_queue``. It is an ordinary
    ``parsl_task_queue`` when batching is not selected and a
    :class:`BatchedTaskQueue` (counting structures) otherwise.
    """

    def __init__(self, qm_task, qm_task_path, QM_config, master_config=None):
        self.qm_task = qm_task
        self.qm_task_path = qm_task_path
        self.settings = labeling_batch_settings(QM_config)
        self.buffer = None
        if self.settings is None:
            self.queue = parsl_task_queue()
            return

        mode = self.settings["mode"]
        required = "molecule_objects" if mode == "native" else "molecule_object"
        if required not in inspect.signature(qm_task.func).parameters:
            raise ValueError("QM batching.mode '{}' requires QM_task {} to accept '{}'.".format(
                mode, qm_task_path, required))
        target = (master_config or {}).get("target_queued_QM")
        if self.settings["partial_policy"] == "full_only" and target is not None \
                and target < self.settings["batch_size"]:
            # Waiting structures count as queued, so builders would stop before a batch fills.
            raise ValueError("QM batching with partial_policy 'full_only' requires "
                             "target_queued_QM >= batching.batch_size.")
        self.buffer = BatchBuffer(self.settings["batch_size"], self.settings["partial_policy"],
                                  self.settings["max_wait_seconds"])
        self.queue = BatchedTaskQueue(self.buffer)
        if mode == "sequential":
            self._packed_task = python_app(executors=list(qm_task.executors))(label_sequentially)

    @property
    def batched(self):
        return self.settings is not None

    def describe(self):
        if not self.batched:
            return "one structure per QM task"
        return "{mode} batches of up to {batch_size} structures ({partial_policy}, {max_wait_seconds:g} s)".format(
            **self.settings)

    def submit(self, molecule_object, input_dicts):
        """Submit (or buffer) one structure.

        ``input_dicts`` is the usual lookup list for ``build_input_dict``, e.g.
        ``[{"QM_config": QM_config}, *all_configs, status]``.
        """
        if not self.batched:
            task_input = build_input_dict(self.qm_task.func, [{"molecule_object": molecule_object}, *input_dicts],
                                          raise_on_fail=True)
            self.queue.add_task(self.qm_task(**task_input))
        else:
            self.buffer.add(molecule_object)

    def submit_ready(self, input_dicts, flush=False):
        """Submit every ready batch; call once per coordinator iteration."""
        if not self.batched:
            return 0
        groups = self.buffer.pop_ready(flush=flush)
        for group in groups:
            if self.settings["mode"] == "native":
                task_input = build_input_dict(self.qm_task.func, [{"molecule_objects": group}, *input_dicts],
                                              raise_on_fail=True)
                task = self.qm_task(**task_input)
            else:
                task_input = build_input_dict(self.qm_task.func, [{"molecule_object": None}, *input_dicts],
                                              raise_on_fail=True)
                task_input.pop("molecule_object")
                task = self._packed_task(group, self.qm_task.func, task_input)
            self.queue.add_task(task, width=len(group))
        return len(groups)

    def poll_delay(self, default):
        """Coordinator sleep, shortened so that partial batches are submitted on time."""
        return poll_delay(default, self.buffer)
