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
"""

import inspect

from alframework.tools.batching import BatchedSubmitter, batching_settings

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
        if not self.multi_trajectory:
            for key in ("driver", "batching"):
                if key in sampler_config:
                    raise ValueError("Sampler '{}' configuration requires a sampler_task that accepts "
                                     "molecule_objects; {} accepts a single molecule_object.".format(
                                         key, sampler_task_path))
            super().__init__(sampler_task, None)
            self.driver = None
            return
        self.driver = driver_settings(sampler_config)
        settings = self.driver["batching"] or {"batch_size": 1, "partial_policy": "timeout",
                                               "max_wait_seconds": 0.0}
        super().__init__(sampler_task, settings)

    def describe(self):
        if not self.multi_trajectory:
            return "one structure per sampler task"
        if self.driver["strategy"] == "single":
            return "one trajectory per task ({} engine)".format(self.driver["engine"])
        return "{} engine, ".format(self.driver["engine"]) + super().describe()
