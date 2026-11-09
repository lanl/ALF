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
from alframework.tools.batching import BatchedSubmitter, StartupSettings, batching_settings, default_batch_key
from alframework.tools.cohorts import CohortPool
from alframework.ml_interfaces.ensemble_inference import ADJACENT_GAPS
from alframework.tools.gap_seeking import gap_seeking_settings
from alframework.tools.sampling_windows import validated_windows, window_number, windowed_config
from alframework.tools.trajectory_summary import TrajectorySummary

STRATEGIES = ("single", "batched")
ENGINES = ("ase", "torchsim", "alchemi")
# Read once at startup. Cutoffs set what a cohort's scores mean, so they stay fixed only in cohort mode.
FIXED_SETTINGS = ("driver", "batching", "acquisition", "model_mode", "state_selection", "gap_uncertainty",
                  "gap_seeking", "sampling_windows")
FIXED_COHORT_SETTINGS = ("Escut", "Fscut", "gap_std_cut")


def state_settings(config):
    """Validate opt-in molecular state selection without importing model libraries."""
    mode = config.get('model_mode', 'ground_state')
    if mode not in ('ground_state', 'excited_state'):
        raise ValueError('model_mode must be ground_state or excited_state')
    seeking = gap_seeking_settings(config)
    if mode == 'ground_state':
        if 'state_selection' in config or 'gap_uncertainty' in config:
            raise ValueError('State selection/gap uncertainty requires model_mode excited_state')
        return None
    policy = config.get('state_selection')
    if policy is not None:
        if not isinstance(policy, dict) or policy.get('mode') not in ('fixed', 'batch_cycle'):
            raise ValueError('state_selection requires fixed or batch_cycle mode')
        states = [policy.get('state')] if policy['mode'] == 'fixed' else policy.get('states')
        if not isinstance(states, (list, tuple)) or not states:
            raise ValueError('State selection requires a nonempty state list')
        for state in states:
            _state_index(state)
        if len(set(states)) != len(states):
            raise ValueError('State cycling cannot contain duplicate states')
        unknown = set(policy) - {'mode', 'state', 'states', 'block_size'}
        if unknown:
            raise ValueError(f'Unknown state_selection settings: {sorted(unknown)}')
        if 'block_size' in policy:
            block = policy['block_size']
            if policy['mode'] != 'batch_cycle':
                raise ValueError('state_selection.block_size requires batch_cycle mode')
            if isinstance(block, bool) or not isinstance(block, int) or block < 1:
                raise ValueError('state_selection.block_size must be a positive integer')
    pair = None
    if 'gap_uncertainty' in config:
        gap = config['gap_uncertainty']
        if isinstance(gap, dict) and 'pairs' in gap:
            # The largest spread among S0-S1, S1-S2, ... on each frame, under one gap_std_cut.
            if 'pair' in gap or gap['pairs'] != ADJACENT_GAPS:
                raise ValueError('gap_uncertainty takes either a pair or "pairs": "adjacent"')
            return dict(policy=policy, gap_pair=ADJACENT_GAPS, gap_seeking=seeking)
        pair = gap.get('pair') if isinstance(gap, dict) else None
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('gap_uncertainty requires a pair of ordered states')
        for state in pair:
            _state_index(state)
        if pair[0] >= pair[1]:
            raise ValueError('Gap pair must be lower state followed by upper state')
    return dict(policy=policy, gap_pair=pair, gap_seeking=seeking)


def _state_index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Selected state must be a nonnegative integer')
    return value


def resolve_selected_state(molecule, settings, ordinal=None):
    metadata = molecule.get_metadata()
    policy = settings['policy']
    if 'selected_state' in metadata:
        state, source = _state_index(metadata['selected_state']), metadata.get('selected_state_source', 'metadata')
    elif policy is None:
        raise ValueError('Excited-state sampling requires selected_state metadata or state_selection')
    elif policy['mode'] == 'fixed':
        state, source = policy['state'], 'state_selection.fixed'
    else:
        if ordinal is None:
            raise ValueError('Unresolved state cycling requires coordinator admission')
        # Each state holds for block_size admissions in a row, so a refill of that many fills one batch.
        block = policy.get('block_size', 1)
        state, source = policy['states'][(ordinal // block) % len(policy['states'])], 'state_selection.batch_cycle'
    molecule.update_metadata(dict(selected_state=state, selected_state_source=source))
    if ordinal is not None:
        molecule.update_metadata(dict(sampling_admission_ordinal=ordinal))
    return state


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
    if engine not in ENGINES:
        raise ValueError(f"driver.engine must be one of {ENGINES}; received {engine!r}.")
    states = state_settings(sampler_config)
    if sampler_config.get("batched_calculator") is not None and engine not in {"torchsim", "alchemi"} and states is None:
        raise ValueError("batched_calculator requires driver.engine 'torchsim' or 'alchemi'; "
                         "the ASE engine uses ase_calculator.")
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
        master_config (dict): Used to check that ``full_only`` batches can fill, and for
            ``trajectory_summary_path``, which is read once, here.
    """

    def __init__(self, sampler_task, sampler_task_path, sampler_config, master_config=None):
        parameters = inspect.signature(sampler_task.func).parameters
        self._states = state_settings(sampler_config)
        self._admission_ordinal = 0
        self.multi_trajectory = "molecule_objects" in parameters
        self.pool = None
        summary_path = (master_config or {}).get("trajectory_summary_path")
        self._summary = None if summary_path is None else TrajectorySummary(summary_path)
        self._windows = validated_windows(sampler_config)
        self._window = None
        if not self.multi_trajectory:
            if self._windows is not None:
                raise ValueError("sampling_windows requires a sampler_task that accepts molecule_objects.")
            if self._states is not None:
                raise ValueError('Excited-state sampling requires a multi-trajectory sampler task')
            if "driver" in sampler_config or "batching" in sampler_config or \
                    "selection" in (sampler_config.get("acquisition") or {}):
                raise ValueError("Sampler driver, batching, and acquisition.selection settings require a "
                                 "sampler_task that accepts molecule_objects; {} accepts a single "
                                 "molecule_object.".format(sampler_task_path))
            super().__init__(sampler_task, None)
            self.driver = None
            return
        for number in range(len(self._windows or ())):
            # A window that would fail later, such as gap-seeking with UDD, should stop the run now.
            driver_settings(windowed_config(sampler_config, self._windows, number))
        self.driver = driver_settings(sampler_config)
        settings = self.driver["batching"] or {"batch_size": 1, "partial_policy": "timeout",
                                               "max_wait_seconds": 0.0}
        capacity = (master_config or {}).get("parallel_samplers")
        if settings["partial_policy"] == "full_only" and capacity is not None and capacity < settings["batch_size"]:
            # Waiting structures count toward parallel_samplers, so builders would stop before a batch fills.
            raise ValueError("Sampler batching with partial_policy 'full_only' requires "
                             "parallel_samplers >= batching.batch_size.")
        selection = cohort_selection(sampler_config)
        if selection is not None:
            self.pool = CohortPool(selection)
        super().__init__(sampler_task, settings, key_function=lambda molecule:
                         _model_batch_key(molecule, excited=self._states is not None))
        fixed = FIXED_SETTINGS + (FIXED_COHORT_SETTINGS if self.pool is not None else ())
        self._startup_settings = StartupSettings(sampler_config, fixed, "sampler")

    def apply_window(self, sampler_config, stored_labels):
        """Return the configuration new sampler tasks should use at this count of stored labels."""
        if self._windows is None:
            return sampler_config
        number = window_number(self._windows, stored_labels)
        if number != self._window:
            print(f"Sampling window {number} of {len(self._windows) - 1} applies at {stored_labels} stored labels.")
            self._window = number
        return windowed_config(sampler_config, self._windows, number)

    def submit(self, molecule_object, input_dicts):
        """Submit (or buffer) one structure; in cohort mode, admit it to a cohort first."""
        if self._states is not None:
            resolve_selected_state(molecule_object, self._states, self._admission_ordinal)
            self._admission_ordinal += 1
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
        if self._summary is not None:
            self._summary.append(sampler_results, failed)
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


def _model_batch_key(molecule, excited=False):
    """Group by PBC and pinned model generation, so a batch never mixes models."""
    key = default_batch_key(molecule) + (molecule.get_metadata().get("sampling_model_id"),)
    return key + (molecule.get_metadata().get('selected_state'),) if excited else key
