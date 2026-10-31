"""Shared helpers for opt-in batched task submission.

Nothing in this module changes ALF's ordinary one-molecule-per-task execution.
A coordinator uses these helpers only when a task's configuration contains a
``batching`` block. The helpers are independent of any sampler, labeler, or
model so that dynamics batches, labeling batches, and builder output widths
can be chosen separately.

Waiting batches are held in memory only; unfinished buffers are not restored
after a process restart.
"""

import time
from collections import OrderedDict

from alframework.tools.molecules_class import MoleculesObject

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


def flatten_molecule_output(output):
    """Normalize single, nested-list, or empty task output into a flat list.

    Order and ``MoleculesObject`` identity are preserved, so splitting or
    combining builder output does not change molecule IDs.
    """
    if output is None:
        return []
    if isinstance(output, MoleculesObject):
        return [output]
    if isinstance(output, (list, tuple)):
        flattened = []
        for item in output:
            flattened.extend(flatten_molecule_output(item))
        return flattened
    raise TypeError("Task output must be a MoleculesObject, a nested list/tuple of "
                    f"MoleculesObject instances, or None; got {type(output).__name__}.")


def default_batch_key(molecule):
    """Group molecules by periodicity.

    Mixed sizes, compositions, and atom orders may share a batch. Adapters with
    stricter requirements (for example a selected excited state or a common
    cell) supply their own key function to :class:`BatchBuffer`.
    """
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
        if not isinstance(molecule, MoleculesObject):
            raise TypeError("BatchBuffer accepts only MoleculesObject instances.")
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

    def prune(self, tasks):
        """Forget tasks no longer present in ``tasks`` (e.g. after harvesting)."""
        live = {id(task) for task in tasks}
        self._widths = {key: value for key, value in self._widths.items() if key in live}


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
