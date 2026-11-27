"""Resolve and validate a workflow before Parsl or the coordinator starts it."""

import inspect
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

from alframework.samplers.acquisition import SINGLE_MEMBER, uses_spread
from alframework.tools import config, memory
from alframework.tools.bootstrap import bootstrap_spare_count
from alframework.tools.builder_outputs import builder_width, builds_batches
from alframework.tools.capabilities import require, validate_task_config
from alframework.tools.capacity import capacity_report, check_member_gpus, task_slots
from alframework.tools.ensemble_training import validated_member_nodes
from alframework.tools.h5_parallel import read_budget_setting, read_workers_setting
from alframework.tools.label_hdf5 import label_units_setting
from alframework.tools.label_parts import incoming_directory
from alframework.tools.labeling_coordinator import LabelingCoordinator
from alframework.tools.loop_timing import coordinator_loop_timer
from alframework.tools.restarts import (
    check_label_grouping_for_ml, held_labels_directory, label_h5_options, label_storage_batch_size,
    status_on_change_setting,
)
from alframework.tools.retraining import retraining_settings
from alframework.tools.sampled_frames import data_by_reference_settings
from alframework.tools.sampler_coordinator import SamplerCoordinator
from alframework.tools.scaling import apply_scaling, describe as describe_scaling, large_run_warning
from alframework.tools.serialization import data_serializer
from alframework.tools.sizing import apply_sizing, derive_sizing, describe as describe_sizing, sizing_block


def bind_task(path, executors):
    """Load the task at ``path`` and check that the Parsl executors it, and the members a training join submits,
    run on exist. A standby executor (``alf_<role>_standby_executor``) that is configured joins its role's."""
    task = config.load_module_from_string(path)
    own_labels = () if task.executors == 'all' else task.executors
    member_labels = getattr(task.func, 'member_executors', ())
    missing = (set(own_labels) | set(member_labels)) - executors.keys() - {'_parsl_internal'}
    if missing:
        raise ValueError(f'{path} requires unconfigured Parsl executors {sorted(missing)}.')
    if task.executors == 'all':
        return task
    standby = (label.replace('_executor', '_standby_executor') for label in task.executors)
    labels = list(dict.fromkeys([*task.executors, *(label for label in standby if label in executors)]))
    if labels != list(task.executors):
        task = copy(task)
        task.executors = labels
    return task


def _labels(task, executors):
    return list(executors) if task.executors == 'all' else task.executors


def _check_members(ml, ml_config, master, executors, member_labels):
    """Make the check a training join makes when it starts its members, so a mistake stops the run at startup.

    A join task's ``member_layout`` gives the device, GPUs per member and nodes per member it trains with.
    Returns the nodes per member.
    """
    layout = getattr(getattr(ml, 'func', ml), 'member_layout', None)
    if not getattr(ml, 'join', False) or layout is None:
        return ml_config.get('member_nodes', 1)
    device, gpus_per_member, member_nodes = layout(ml_config)
    validated_member_nodes(member_nodes, gpus_per_member)
    for label in member_labels:
        if label in executors:
            check_member_gpus(executors[label], master.get('gpus_per_node'), device, gpus_per_member, member_nodes,
                              label)
    return member_nodes


def resolve_sizing(master, sampler_config, qm_config, executors, builder=None, sampler=None, ml_config=None, ml=None):
    """Apply the master configuration's ``sizing`` block to the configurations, in place, and return
    ``(block, derived)``.

    The sampler slots come from the sampler task's ``executors``, and the builder and sampler widths from the tasks'
    signatures. Without a task (``None``), as when a tool reads a run on another machine, a builder counts as taking
    one ID and a sampler as batching, and the slots must be given as ``sizing.sampler_tasks_at_once``. A training
    layout (``sizing.labels_per_member_gpu``) goes into ``ml_config``; without the ML task it isn't checked to train
    members across GPUs.
    """
    block = sizing_block(master)
    slots = task_slots(_labels(sampler, executors), executors) if sampler is not None else None
    round_labels = None
    if block and 'labels_per_member_gpu' in block and 'labels_per_round' not in block:
        retraining = retraining_settings(master, required=False)
        if retraining is not None and retraining['counts'] == 'labels':
            round_labels = retraining['after']
    derived = derive_sizing(
        block, sampler_slots=slots, builder_batches=builder is not None and builds_batches(builder),
        sampler_batches=sampler is None or 'molecule_objects' in inspect.signature(sampler.func).parameters,
        round_labels=round_labels, gpus_per_node=master.get('gpus_per_node'),
        n_models=(ml_config or {}).get('n_models'),
        member_layout=ml is None or getattr(getattr(ml, 'func', ml), 'member_layout', None) is not None)
    apply_sizing(derived, master, sampler_config, qm_config, ml_config)
    layout = getattr(getattr(ml, 'func', ml), 'member_layout', None) if ml is not None else None
    if derived['ML'] and layout is not None:
        _, gpus, nodes = layout(ml_config)
        if (gpus, nodes) != (derived['ML']['gpus_per_member'], derived['ML']['member_nodes']):
            task = master.get('ML_task', 'The ML task')
            raise ValueError(f'{task} trains each member on {gpus} GPU(s) and {nodes} node(s) whatever '
                             'gpus_per_member and member_nodes say, so sizing.labels_per_member_gpu cannot size it. '
                             'Use a task that trains members across GPUs, such as '
                             'alframework.ml_interfaces.hippynn.interface.train_HIPPYNN_ensemble_multinode_task.')
    return block, derived


def resolve_settings(configs, *, executors=None, builder=None, sampler=None):
    """The settings a run uses, for tools that read a run without starting it: ``configs`` (from
    :func:`~alframework.tools.config.load_workflow_configs`) with the ``scaling`` and ``sizing`` blocks applied, in
    place, as the coordinator applies them at startup. Returns ``configs``."""
    apply_scaling(configs.master, configs.sampler, configs.ML, check_tasks=False)
    resolve_sizing(configs.master, configs.sampler, configs.QM, executors or {}, builder, sampler, configs.ML)
    return configs


def reapply_settings(configs, scaling, derived):
    """Apply the ``scaling`` block and the ``sizing`` values the coordinator started with to reloaded ``configs``,
    in place. An edit to either block waits for a restart, so a reload keeps their startup meaning."""
    apply_sizing(derived, configs.master, configs.sampler, configs.QM, configs.ML)
    apply_scaling(configs.master, configs.sampler, configs.ML, scaling or {}, check_tasks=False)
    return configs


def check_workflow(master, builder_config, sampler_config, qm_config, ml_config, *, debug=False,
                   test_sampler=False, test_qm=False):
    """Validate the loaded configuration and return the tasks and settings for runtime."""
    scaling = apply_scaling(master, sampler_config, ml_config)
    if scaling:
        print(describe_scaling(scaling), flush=True)
    poll_interval = config.coordinator_poll_interval(master)
    label_budget = config.stored_label_budget(master)
    trim_snapshot_memory = memory.snapshot_memory_trim(master)
    memory.coordinator_huge_pages_setting(master)
    spare_seeds = bootstrap_spare_count(master)
    rewrite_status_on_change = status_on_change_setting(master)
    loop_timer = coordinator_loop_timer(master)
    label_storage_batch_size(master)
    held_labels_directory(master)
    label_units = label_units_setting(master['properties_list'], master.get('label_units'))
    h5_options = label_h5_options(master)
    check_label_grouping_for_ml(h5_options, master.get('ML_task'))
    serializer = data_serializer(master)
    config_key = 'parsl_debug_configuration' if debug and 'parsl_debug_configuration' in master else 'parsl_configuration'
    parsl_config = config.load_module_from_string(master[config_key])
    executors = {executor.label: executor for executor in parsl_config.executors}
    builder, sampler, qm, ml = (bind_task(master[key], executors)
                                for key in ('builder_task', 'sampler_task', 'QM_task', 'ML_task'))
    validate_task_config(builder, builder_config, master)
    validate_task_config(sampler, sampler_config, master)

    sizing, derived = resolve_sizing(master, sampler_config, qm_config, executors, builder, sampler, ml_config, ml)
    if sizing:
        print(describe_sizing(sizing, derived), flush=True)
    retraining = retraining_settings(master, required=not debug)  # the test modes never retrain
    width = builder_width(builder, master)
    if not builds_batches(builder) and master.get('maximum_builder_structures', 1) > 1:
        print(f"Warning: maximum_builder_structures {master['maximum_builder_structures']:,} has no effect: "
              f"{master['builder_task']} takes one moleculeid, so each builder task builds one structure.", flush=True)
    warning = large_run_warning(master, sampler_config, sampler)
    if warning:
        print('Warning: ' + warning, flush=True)
    sampling_master = master
    if test_sampler and 'data_by_reference' in master:
        sampling_master = dict(master)
        sampling_master.pop('data_by_reference')
    sampling = SamplerCoordinator(sampler, master['sampler_task'], sampler_config, sampling_master, start_spool=False)
    print('Sampling: ' + sampling.describe())

    reference = data_by_reference_settings(master) or {}
    handoff = None
    if reference.get('labels') and not test_qm:
        handoff = dict(directory=str(incoming_directory(Path(master['h5_path'].format(0)).resolve())),
                       properties=master['properties_list'], schema=json.dumps(master['properties_list'], sort_keys=True),
                       h5_options=h5_options, label_units=label_units)
    labeling = LabelingCoordinator(qm, master['QM_task'], qm_config, master,
                                   sampled_frame_handoff=sampling.spool_root is not None, label_handoff=handoff)
    print('QM labeling: ' + labeling.describe())
    for key, check in (('h5_read_workers', read_workers_setting), ('h5_read_max_inflight_bytes', read_budget_setting)):
        if ml_config.get(key) is not None:
            require(ml, master['ML_task'], key)
            check(ml_config[key])
    validate_task_config(ml, ml_config, master)

    members = ml_config.get('n_models') if 'n_models' in ml_config else (ml_config.get('fine_tune') or {}).get('n_models')
    if members == 1:
        try:
            reads_spread = uses_spread(sampler_config)
        except ValueError:
            reads_spread = False
        if reads_spread:
            raise ValueError(SINGLE_MEMBER.format(what='The ML configuration (n_models 1)'))
    member_labels = getattr(ml.func, 'member_executors', ('alf_ML_executor', 'alf_ML_standby_executor'))
    member_nodes = _check_members(ml, ml_config, master, executors, member_labels)
    lines, warnings = capacity_report(
        master, executors,
        builder_labels=_labels(builder, executors), builder_width=width, builder_batches=builds_batches(builder),
        sampler_labels=_labels(sampler, executors), sampler_width=sampling.settings['batch_size'] if sampling.batched else 1,
        qm_labels=_labels(qm, executors), qm_width=labeling.settings['batch_size'] if labeling.batched else 1,
        ml_labels=member_labels,
        n_models=members if getattr(ml, 'join', False) else None, member_nodes=member_nodes)
    print('Capacity:\n  ' + '\n  '.join(lines), flush=True)
    for warning in warnings:
        print('Warning: ' + warning, flush=True)
    return SimpleNamespace(parsl_config=parsl_config, serializer=serializer, scaling=scaling, sizing=derived,
                           builder=builder, sampler=sampler, qm=qm, ml=ml, sampling=sampling, labeling=labeling,
                           label_handoff=handoff, reference=reference, poll_interval=poll_interval,
                           label_budget=label_budget, trim_snapshot_memory=trim_snapshot_memory, spare_seeds=spare_seeds,
                           rewrite_status_on_change=rewrite_status_on_change, loop_timer=loop_timer,
                           retraining=retraining)
