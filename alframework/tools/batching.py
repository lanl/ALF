"""Shared helpers for opt-in batched task submission.

Nothing in this module changes ALF's ordinary one-molecule-per-task execution.
A coordinator uses these helpers only when a task's configuration contains a
``batching`` block. The helpers are independent of any sampler, labeler, or
model so that dynamics batches, labeling batches, and builder output widths
can be chosen separately.

Waiting batches are held in memory only; unfinished buffers are not restored
after a process restart.
"""

import copy
import logging
import time
from collections import OrderedDict

from alframework.tools.config import build_input_dict
from alframework.tools.molecules import MoleculesObject
from alframework.tools.parsl_tasks import (
    FAILED_TASK_STATUSES,
    SUCCESSFUL_TASK_STATUSES,
    is_final_task,
    ParslTaskQueue,
)
from alframework.tools.sampled_frames import SampledFrame

logger = logging.getLogger(__name__)

PARTIAL_POLICIES = ("timeout", "full_only")
DEFAULT_MAX_WAIT_SECONDS = 5.0


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return int(value)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")
    return int(value)


def batching_settings(config):
    """Return validated batching settings from a task configuration.

    Args:
        config (dict): Sampler or QM configuration. Batching is selected only
            when it contains a ``batching`` dictionary.

    Returns:
        (dict or None): ``None`` when batching is not selected. Otherwise a
        dictionary with ``batch_size``, ``partial_policy``, and
        ``max_wait_seconds``.
    """
    block = (config or {}).get("batching")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("batching must be a dictionary.")
    unknown = set(block) - {"batch_size", "partial_policy", "max_wait_seconds"}
    if unknown:
        raise ValueError(f"Unknown batching settings: {sorted(unknown)}.")
    if "batch_size" not in block:
        raise ValueError("batching.batch_size is required.")
    policy = block.get("partial_policy", "timeout")
    if policy not in PARTIAL_POLICIES:
        raise ValueError(f"batching.partial_policy must be one of {PARTIAL_POLICIES}.")
    wait = float(block.get("max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS))
    if not wait >= 0:
        raise ValueError("batching.max_wait_seconds must be nonnegative.")
    return {
        "batch_size": _positive_integer(block["batch_size"], "batching.batch_size"),
        "partial_policy": policy,
        "max_wait_seconds": wait,
    }


def flatten_molecule_output(output, types=(MoleculesObject,)):
    """Normalize single, nested-list, or empty task output into a flat list.

    Order and ``MoleculesObject`` identity are preserved, so splitting or
    combining builder output does not change molecule IDs.
    """
    if output is None:
        return []
    if isinstance(output, types):
        return [output]
    if isinstance(output, (list, tuple)):
        flattened = []
        for item in output:
            flattened.extend(flatten_molecule_output(item, types))
        return flattened
    raise TypeError("Task output must be a MoleculesObject, a nested list/tuple of "
                    f"MoleculesObject instances, or None; got {type(output).__name__}.")


def default_batch_key(molecule):
    """Group molecules by periodicity.

    Mixed sizes, compositions, and atom orders may share a batch. Adapters with
    stricter requirements (for example a selected excited state or a common
    cell) supply their own key function to :class:`BatchBuffer`.
    """
    if isinstance(molecule, SampledFrame):
        if molecule.pbc is None:
            raise ValueError(f"Cannot batch rejected sampled frame {molecule.get_moleculeid()!r}.")
        return ("pbc", molecule.pbc)
    atoms = molecule.get_atoms()
    if atoms is None:
        raise ValueError(f"Cannot batch {molecule.get_moleculeid()!r}: atoms are None.")
    return ("pbc", tuple(bool(value) for value in atoms.get_pbc()))


class BatchBuffer:
    """In-memory waiting buffer that groups compatible molecules into batches.

    Full batches are released immediately. With the ``timeout`` policy, a
    partial batch is released once its oldest member has waited
    ``max_wait_seconds``; :meth:`pop_ready` should be polled every coordinator
    iteration, including iterations with no new input.
    """

    def __init__(self, batch_size, partial_policy="timeout",
                 max_wait_seconds=DEFAULT_MAX_WAIT_SECONDS, key_function=default_batch_key):
        self.batch_size = _positive_integer(batch_size, "batch_size")
        if partial_policy not in PARTIAL_POLICIES:
            raise ValueError(f"partial_policy must be one of {PARTIAL_POLICIES}.")
        self.partial_policy = partial_policy
        self.max_wait_seconds = float(max_wait_seconds)
        self.key_function = key_function
        # Insertion-ordered buckets give deterministic release order.
        self._buckets = OrderedDict()

    @classmethod
    def from_config(cls, config, key_function=default_batch_key):
        """Build a buffer from a configuration's ``batching`` block, or return ``None``."""
        settings = batching_settings(config)
        if settings is None:
            return None
        return cls(key_function=key_function, **settings)

    def __len__(self):
        return sum(len(bucket) for bucket in self._buckets.values())

    def add(self, molecule, now=None):
        if not isinstance(molecule, (MoleculesObject, SampledFrame)):
            raise TypeError("BatchBuffer accepts MoleculesObject instances and sampled frame references.")
        key = self.key_function(molecule)
        timestamp = time.time() if now is None else float(now)
        self._buckets.setdefault(key, []).append((timestamp, molecule))

    def extend(self, molecules, now=None):
        for molecule in molecules:
            self.add(molecule, now=now)

    def pop_ready(self, now=None, flush=False):
        """Remove and return ready batches as lists of ``MoleculesObject``.

        Args:
            now (float): Current time; defaults to ``time.time()``.
            flush (bool): Release every waiting partial batch regardless of
                policy or age, e.g. when no more input will arrive.
        """
        current = time.time() if now is None else float(now)
        ready = []
        for key in list(self._buckets):
            bucket = self._buckets[key]
            while len(bucket) >= self.batch_size:
                ready.append([molecule for _, molecule in bucket[:self.batch_size]])
                del bucket[:self.batch_size]
            expired = (self.partial_policy == "timeout"
                       and bucket and current - bucket[0][0] >= self.max_wait_seconds)
            if bucket and (flush or expired):
                ready.append([molecule for _, molecule in bucket])
                bucket.clear()
            if not bucket:
                del self._buckets[key]
        return ready

    def seconds_until_ready(self, now=None):
        """Seconds until the next partial batch times out, or ``None`` if none will."""
        if self.partial_policy != "timeout" or not self._buckets:
            return None
        current = time.time() if now is None else float(now)
        oldest = min(bucket[0][0] for bucket in self._buckets.values())
        return max(0.0, self.max_wait_seconds - (current - oldest))

    def status(self, now=None):
        """Return a JSON-safe summary of waiting molecules."""
        current = time.time() if now is None else float(now)
        buckets = [{"key": repr(key), "count": len(bucket),
                    "oldest_age_seconds": current - bucket[0][0]}
                   for key, bucket in self._buckets.items()]
        return {"batch_size": self.batch_size, "partial_policy": self.partial_policy,
                "waiting": len(self), "buckets": buckets}


class TaskWidths:
    """Record how many input molecules each submitted task represents.

    ALF's capacity limits such as ``parallel_samplers`` count structures, while
    task queues count tasks. Recording the actual width at submission keeps
    accounting correct for partial batches and changed batch sizes.
    """

    def __init__(self):
        self._widths = {}

    def record(self, task, width):
        self._widths[id(task)] = _nonnegative_integer(width, "task width")
        return task

    def width(self, task):
        # Tasks submitted without batching represent one structure.
        return self._widths.get(id(task), 1)

    def total(self, tasks):
        return sum(self.width(task) for task in tasks)

    def forget(self, task):
        self._widths.pop(id(task), None)

    def prune(self, tasks):
        """Forget tasks no longer present in ``tasks`` (e.g. after harvesting)."""
        live = {id(task) for task in tasks}
        self._widths = {key: value for key, value in self._widths.items() if key in live}


class BatchedTaskQueue(ParslTaskQueue):
    """Task queue whose counts are in input structures rather than tasks.

    This keeps ALF's structure-based limits (``bootstrap_set``,
    ``target_queued_QM``, ``retraining.after``, ...) meaningful when one task
    holds several structures. Molecules waiting in ``buffer`` count as queued.
    :meth:`get_task_results` flattens batch results into one list of
    ``MoleculesObject`` and still reports the number of failed *tasks*;
    :attr:`failed_structures` accumulates the structures those tasks held.
    """

    def __init__(self, buffer=None):
        super().__init__()
        self.buffer = buffer
        self.widths = TaskWidths()
        self.failed_structures = 0
        # Inputs of tasks that failed as a whole, for callers that track members.
        self.failed_inputs = []
        self._inputs = {}
        # What a finished task may return; data_by_reference labels add label receipts.
        self.output_types = (MoleculesObject,)

    def _weight(self, task):
        return self.widths.width(task)

    def add_task(self, task, width=1, inputs=None):
        self.widths.record(task, width)
        if inputs is not None:
            self._inputs[id(task)] = list(inputs)
        super().add_task(task)

    def get_waiting_number(self):
        return 0 if self.buffer is None else len(self.buffer)

    def get_task_number(self):
        return len(self.task_list)

    def get_number(self):
        return self._total + self.get_waiting_number()

    def get_completed_number(self):
        return int(self._count('final', is_final_task))

    def get_running_number(self):
        return super().get_running_number()

    def get_exec_done_number(self):
        return int(self._count('exec_done', lambda task: task.task_status() == 'exec_done' and task.done()))

    def get_failed_number(self):
        return int(self._count('failed', lambda task: task.task_status() in FAILED_TASK_STATUSES))

    def get_task_results(self, max_items=None, *, on_finished_inputs=None):
        return self._harvest(self._take_finished(max_items=max_items), self.output_types,
                             on_finished_inputs=on_finished_inputs)

    def _harvest(self, taken, output_types=(MoleculesObject,), *, on_finished_inputs=None):
        results_list = []
        failed_number = 0
        self.last_failed = []
        for task, status in taken:
            if status in SUCCESSFUL_TASK_STATUSES:
                try:
                    results = flatten_molecule_output(task.result(), output_types)
                except Exception:
                    logger.error("Task marked as done but raised or returned invalid output", exc_info=True)
                else:
                    results_list.extend(results)
                    if on_finished_inputs is not None:
                        on_finished_inputs(self._inputs.get(id(task), ()), True)
                    self._forget(task)
                    continue
            else:
                try:
                    task.result()
                    logger.warning("Task marked as %s returned a result; discarding it", status)
                except Exception:
                    logger.warning("Task failed", exc_info=True)
            failed_number += 1
            self.last_failed.append(task)
            self.failed_structures += self.widths.width(task)
            inputs = self._inputs.get(id(task), ())
            if on_finished_inputs is None:
                self.failed_inputs.extend(inputs)
            else:
                on_finished_inputs(inputs, False)
            self._forget(task)
        return results_list, failed_number

    def _forget(self, task):
        """Drop a harvested task's width and inputs."""
        self.widths.forget(task)
        self._inputs.pop(id(task), None)

    def print_status(self):
        print('Total Structures: {:d} ({:d} tasks, {:d} waiting for a batch)'.format(
            self.get_number(), self.get_task_number(), self.get_waiting_number()))
        print('Finished Structures: {:d}'.format(self.get_completed_number()))
        print('Finished Successfully Structures: {:d}'.format(self.get_exec_done_number()))
        print('Finished Failed Structures: {:d}'.format(self.get_failed_number()))


class SampledFrameQueue(BatchedTaskQueue):
    """Release finished Parsl futures while keeping their receipts in the sampler-wave counts."""

    def __init__(self, buffer):
        super().__init__(buffer)
        self._clear_receipts()

    def _clear_receipts(self):
        self._receipts = []
        self._received_tasks = self._received_structures = 0
        self._received_failed_tasks = self._received_failed_structures = 0
        self._received_exec_done = 0

    def collect(self):
        taken = self._take_finished()
        if not taken:
            return
        width = sum(self.widths.width(task) for task, _ in taken)
        self._received_exec_done += sum(self.widths.width(task) for task, status in taken if status == 'exec_done')
        prior_failures = self.failed_structures
        receipts, failed = self._harvest(taken, (SampledFrame,))
        self._receipts.extend(receipts)
        self._received_tasks += len(taken)
        self._received_structures += width
        self._received_failed_tasks += failed
        self._received_failed_structures += self.failed_structures - prior_failures

    def get_number(self):
        return super().get_number() + self._received_structures

    def get_task_number(self):
        return super().get_task_number() + self._received_tasks

    def get_completed_number(self):
        return super().get_completed_number() + self._received_structures

    def get_exec_done_number(self):
        return super().get_exec_done_number() + self._received_exec_done

    def get_failed_number(self):
        return super().get_failed_number() + self._received_failed_structures

    def get_task_results(self):
        self.collect()
        receipts = self._receipts
        # A later batch may have finished first. Keep the same handoff order as inline results.
        receipts.sort(key=lambda frame: frame.batch)
        failures = self._received_failed_tasks
        self._clear_receipts()
        return receipts, failures


def poll_delay(default, *buffers, now=None):
    """Cap the coordinator sleep so that partial-batch timeouts are honored."""
    delay = float(default)
    for buffer in buffers:
        if buffer is None:
            continue
        remaining = buffer.seconds_until_ready(now=now)
        if remaining is not None:
            delay = min(delay, remaining)
    return delay


_ABSENT = object()


def check_full_only(settings, limit, limit_name, stage):
    """Refuse ``full_only`` batches that could never fill: structures waiting in a batch count toward ``limit``
    (``parallel_samplers`` or ``target_queued_QM``), so builders would stop before a batch is full. Used at startup
    and on each reload, since the limits can be edited during a run."""
    if settings and settings.get("partial_policy") == "full_only" and limit is not None \
            and limit < settings["batch_size"]:
        raise ValueError(f"{stage} batching with partial_policy 'full_only' requires "
                         f"{limit_name} >= batching.batch_size.")


class StartupSettings:
    """Hold the settings a coordinator read at startup, so that a reload can't change their meaning.

    ALF reloads its configuration files every loop, but a coordinator builds its batching and
    selection once. Letting tasks see newer values would mix two meanings within one run.
    """

    def __init__(self, config, keys, label):
        self._values = {key: copy.deepcopy(config[key]) for key in keys if key in config}
        self._keys = tuple(keys)
        self._label = label
        self._warned = set()

    def restore(self, reloaded):
        """Return ``reloaded`` with the startup values, warning once per changed setting."""
        restored = dict(reloaded)
        for key in self._keys:
            if reloaded.get(key, _ABSENT) == self._values.get(key, _ABSENT):
                self._warned.discard(key)
                continue
            if key not in self._warned:
                print(f"Ignoring the reloaded {self._label} setting '{key}': it is fixed at startup. "
                      "Restart ALF to change it.")
                self._warned.add(key)
            if key in self._values:
                restored[key] = copy.deepcopy(self._values[key])
            else:
                del restored[key]
        return restored


class BatchedSubmitter:
    """Submit structures to a task one at a time or in compatible batches.

    With ``settings`` of ``None``, :meth:`submit` creates one ordinary task per
    structure (``molecule_object=...``) on a plain ``ParslTaskQueue``, exactly
    as ALF always has. Otherwise structures wait in a :class:`BatchBuffer` and
    :meth:`submit_ready` turns ready groups into tasks via :meth:`_group_task`,
    on a :class:`BatchedTaskQueue` that counts structures.

    Args:
        task: The loaded Parsl app.
        settings (dict or None): ``batch_size``, ``partial_policy``, and
            ``max_wait_seconds``, as returned by :func:`batching_settings`.
    """

    def __init__(self, task, settings, key_function=default_batch_key):
        self.task = task
        self.settings = settings
        self.buffer = None
        self._failed_structures_read = 0
        self._startup_settings = None
        if settings is None:
            self.queue = ParslTaskQueue()
        else:
            self.buffer = BatchBuffer(settings["batch_size"], settings["partial_policy"],
                                      settings["max_wait_seconds"], key_function=key_function)
            self.queue = BatchedTaskQueue(self.buffer)

    @property
    def batched(self):
        return self.settings is not None

    def describe(self):
        if not self.batched:
            return "one structure per task"
        return "batches of up to {batch_size} structures ({partial_policy}, {max_wait_seconds:g} s)".format(
            **self.settings)

    def keep_startup_settings(self, reloaded):
        """Return a reloaded task configuration with this coordinator's fixed settings restored."""
        return reloaded if self._startup_settings is None else self._startup_settings.restore(reloaded)

    def take_failed_structures(self, failed_tasks):
        """Return the structures lost since the last call, given the ``failed_tasks`` just harvested.

        A plain queue's tasks hold one structure each, so ``failed_tasks`` is already that count.
        """
        if not self.batched:
            return failed_tasks
        lost = self.queue.failed_structures - self._failed_structures_read
        self._failed_structures_read = self.queue.failed_structures
        return lost

    def submit(self, molecule_object, input_dicts):
        """Submit (or buffer) one structure.

        ``input_dicts`` is the usual lookup list for ``build_input_dict``, e.g.
        ``[{"QM_config": QM_config}, *all_configs, status]``.
        """
        if not self.batched:
            task_input = build_input_dict(self.task.func, [{"molecule_object": molecule_object}, *input_dicts],
                                          raise_on_fail=True)
            self.queue.add_task(self.task(**task_input))
        else:
            self.buffer.add(molecule_object)

    def submit_ready(self, input_dicts, flush=False):
        """Submit every ready batch; call once per coordinator iteration."""
        if not self.batched:
            return 0
        groups = self.buffer.pop_ready(flush=flush)
        for group in groups:
            self.queue.add_task(self._group_task(group, input_dicts), width=len(group), inputs=group)
        return len(groups)

    def _group_task(self, group, input_dicts):
        """Launch one task for ``group``; by default the task accepts ``molecule_objects``."""
        task_input = build_input_dict(self.task.func, [{"molecule_objects": group}, *input_dicts],
                                      raise_on_fail=True)
        return self.task(**task_input)

    def poll_delay(self, default):
        """Coordinator sleep, shortened so that partial batches are submitted on time."""
        return poll_delay(default, self.buffer)
