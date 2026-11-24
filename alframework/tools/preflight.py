"""Resolve and validate a workflow before Parsl or the coordinator starts it."""

import inspect
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

from alframework.samplers.acquisition import SINGLE_MEMBER, uses_spread
from alframework.tools import config, memory
from alframework.tools.bootstrap import bootstrap_spare_count
from alframework.tools.capabilities import require, validate_task_config
from alframework.tools.capacity import capacity_report, task_slots
from alframework.tools.h5_parallel import read_budget_setting, read_workers_setting
from alframework.tools.label_hdf5 import label_units_setting
from alframework.tools.label_parts import incoming_directory
from alframework.tools.labeling_coordinator import LabelingCoordinator
from alframework.tools.loop_timing import coordinator_loop_timer
from alframework.tools.restarts import held_labels_directory, label_h5_options, label_storage_batch_size, status_on_change_setting
from alframework.tools.sampled_frames import data_by_reference_settings
from alframework.tools.sampler_coordinator import SamplerCoordinator
from alframework.tools.scaling import apply_scaling, describe as describe_scaling, large_run_warning
from alframework.tools.serialization import data_serializer
from alframework.tools.sizing import apply_sizing, derive_sizing, describe as describe_sizing, sizing_block


def _bind_task(path, executors):
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
    serializer = data_serializer(master)
    config_key = 'parsl_debug_configuration' if debug and 'parsl_debug_configuration' in master else 'parsl_configuration'
    parsl_config = config.load_module_from_string(master[config_key])
    executors = {executor.label: executor for executor in parsl_config.executors}
    builder, sampler, qm, ml = (_bind_task(master[key], executors)
                                for key in ('builder_task', 'sampler_task', 'QM_task', 'ML_task'))
    validate_task_config(builder, builder_config, master)
    validate_task_config(sampler, sampler_config, master)

    sizing = sizing_block(master)
    derived = derive_sizing(
        sizing, sampler_slots=task_slots(_labels(sampler, executors), executors),
        builder_batches='moleculeids' in inspect.signature(builder.func).parameters,
        sampler_batches='molecule_objects' in inspect.signature(sampler.func).parameters)
    apply_sizing(derived, master, sampler_config, qm_config)
    if sizing:
        print(describe_sizing(sizing, derived), flush=True)
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
    lines, warnings = capacity_report(
        master, executors,
        builder_labels=_labels(builder, executors), builder_width=master.get('maximum_builder_structures', 1),
        builder_batches=master.get('maximum_builder_structures', 1) > 1,
        sampler_labels=_labels(sampler, executors), sampler_width=sampling.settings['batch_size'] if sampling.batched else 1,
        qm_labels=_labels(qm, executors), qm_width=labeling.settings['batch_size'] if labeling.batched else 1,
        ml_labels=member_labels,
        n_models=members if getattr(ml, 'join', False) else None, member_nodes=ml_config.get('member_nodes', 1))
    print('Capacity:\n  ' + '\n  '.join(lines), flush=True)
    for warning in warnings:
        print('Warning: ' + warning, flush=True)
    return SimpleNamespace(parsl_config=parsl_config, serializer=serializer, scaling=scaling, sizing=derived,
                           builder=builder, sampler=sampler, qm=qm, ml=ml, sampling=sampling, labeling=labeling,
                           label_handoff=handoff, reference=reference, poll_interval=poll_interval,
                           label_budget=label_budget, trim_snapshot_memory=trim_snapshot_memory, spare_seeds=spare_seeds,
                           rewrite_status_on_change=rewrite_status_on_change, loop_timer=loop_timer)
