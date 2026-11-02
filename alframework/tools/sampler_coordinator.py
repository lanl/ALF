"""Sampler task submission for the ALF coordinator.

Existing single-structure sampler tasks (``molecule_object``) are submitted
one task per structure exactly as before. Multi-trajectory sampler tasks
(``molecule_objects``, e.g. ``batched_mlmd.mlmd_trajectories_task``) select
their grouping with the sampler configuration's ``driver`` block:

``"driver": {"strategy": "single"}`` (default)
    One trajectory per task, submitted immediately.

``"driver": {"strategy": "batched"}``
    Trajectories wait in a buffer configured by the ``batching`` block and
    are submitted in groups of up to ``batch_size``.

In both cases the sampler queue counts structures, so ``parallel_samplers``
keeps its per-structure meaning. Settings are read once at startup; waiting
structures are held in memory only.

With ``acquisition.selection.mode`` ``cohort_top_k``, every structure is first
admitted to a :class:`~alframework.tools.cohorts.CohortPool` and sampled with
its cohort's pinned model; :meth:`SamplerCoordinator.harvest` releases each
cohort's selected frames for labeling once the whole cohort has finished.
"""

import inspect

from alframework.samplers.acquisition import cohort_selection
from alframework.tools.batching import BatchedSubmitter, batching_settings, default_batch_key
from alframework.tools.cohorts import CohortPool

STRATEGIES = ("single", "batched")
ENGINES = ("ase",)
PLANNED_ENGINES = {"torchsim": "Increment 5", "alchemi": "Increment 6"}


def driver_settings(sampler_config):
    """Return validated ``driver`` settings for a multi-trajectory sampler."""
    block = sampler_config.get("driver") or {}
    if not isinstance(block, dict):
        raise ValueError("driver must be a dictionary.")
    unknown = set(block) - {"strategy", "engine"}
    if unknown:
        raise ValueError(f"Unknown driver settings: {sorted(unknown)}.")
    strategy = block.get("strategy", "single")
    engine = block.get("engine", "ase")
    if strategy not in STRATEGIES:
        raise ValueError(f"driver.strategy must be one of {STRATEGIES}; received {strategy!r}.")
    if engine in PLANNED_ENGINES:
        raise NotImplementedError(f"driver.engine {engine!r} is not available yet ({PLANNED_ENGINES[engine]}).")
    if engine not in ENGINES:
        raise ValueError(f"driver.engine must be one of {ENGINES}; received {engine!r}.")
    batching = batching_settings(sampler_config)
    if strategy == "batched" and batching is None:
        raise ValueError("driver.strategy 'batched' requires a sampler 'batching' block.")
    if strategy == "single" and batching is not None:
        raise ValueError("A sampler 'batching' block requires driver.strategy 'batched'.")
    return {"strategy": strategy, "engine": engine, "batching": batching}


class SamplerCoordinator(BatchedSubmitter):
    """Submit sampler work; ``queue`` replaces ``sampler_task_queue``.

    Args:
        sampler_task: The loaded ``sampler_task`` Parsl app.
        sampler_task_path (str): Its import string.
        sampler_config (dict): Sampler configuration.
    """

    def __init__(self, sampler_task, sampler_task_path, sampler_config):
        parameters = inspect.signature(sampler_task.func).parameters
        self.multi_trajectory = "molecule_objects" in parameters
        self.pool = None
        if not self.multi_trajectory:
            if "driver" in sampler_config or "batching" in sampler_config or \
                    "selection" in (sampler_config.get("acquisition") or {}):
                raise ValueError("Sampler driver, batching, and acquisition.selection settings require a "
                                 "sampler_task that accepts molecule_objects; {} accepts a single "
                                 "molecule_object.".format(sampler_task_path))
            super().__init__(sampler_task, None)
            self.driver = None
            return
        self.driver = driver_settings(sampler_config)
        settings = self.driver["batching"] or {"batch_size": 1, "partial_policy": "timeout",
                                               "max_wait_seconds": 0.0}
        selection = cohort_selection(sampler_config)
        if selection is not None:
            self.pool = CohortPool(selection)
        super().__init__(sampler_task, settings, key_function=_model_batch_key)

    def submit(self, molecule_object, input_dicts):
        """Submit (or buffer) one structure; in cohort mode, admit it to a cohort first."""
        if self.pool is not None:
            model_id = _lookup("current_model_id", input_dicts)
            molecule_object.update_metadata({"sampling_model_id": self.pool.admit(molecule_object, model_id)})
        elif self.multi_trajectory:
            molecule_object.update_metadata({"sampling_model_id": _lookup("current_model_id", input_dicts)})
        super().submit(molecule_object, input_dicts)

    def _group_task(self, group, input_dicts):
        # Every member of a group shares one model generation (see _model_batch_key).
        model_id = group[0].get_metadata()["sampling_model_id"]
        return super()._group_task(group, [{"current_model_id": model_id}, *input_dicts])

    def harvest(self, sampler_results):
        """Return the frames to label from harvested sampler results.

        Without cohorts these are the results whose ``atoms`` is set, as before.
        With cohorts, results (and members of failed tasks) are recorded in the
        pool and the selected frames of every completed cohort are returned.
        """
        failed = list(getattr(self.queue, "failed_inputs", []))
        if failed:
            self.queue.failed_inputs.clear()
        if self.pool is None:
            return [structure for structure in sampler_results if structure.get_atoms() is not None]
        for structure in sampler_results:
            self.pool.record(structure)
        self.pool.record_failed(failed)
        return self.pool.ready()

    def status(self):
        return None if self.pool is None else self.pool.status()

    def describe(self):
        if not self.multi_trajectory:
            return "one structure per sampler task"
        if self.driver["strategy"] == "single":
            return "one trajectory per task ({} engine)".format(self.driver["engine"])
        return "{} engine, ".format(self.driver["engine"]) + super().describe()


def _lookup(name, input_dicts):
    for values in input_dicts:
        if name in values:
            return values[name]
    raise ValueError(f"Sampler submission requires {name!r}.")


def _model_batch_key(molecule):
    """Group by PBC and pinned model generation, so a batch never mixes models."""
    return default_batch_key(molecule) + (molecule.get_metadata().get("sampling_model_id"),)
