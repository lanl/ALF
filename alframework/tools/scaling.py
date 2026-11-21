"""One place in the master configuration for the settings that change how ALF runs at scale.

The master configuration's ``scaling`` block takes these settings by their usual names and passes each to the
configuration that uses it, so a large run states all of them together. Each keeps its own default, and each
can still be set where it lives instead:

- the master configuration's ``MASTER_SETTINGS``, such as ``parsl_data_serializer`` and
  ``label_storage_batch_size``;
- the sampler configuration's ``SAMPLER_SETTINGS``, for the batched MLMD sampler;
- the ML configuration's ``ML_SETTINGS``, for HIPPYNN training, and ``h5_read_workers`` for fine-tuning too.

A setting in the block and in its own configuration with a different value is an error, so neither silently
wins. Settings that size the work, such as ``batching``, ``parallel_samplers`` and ``gpus_per_member``, change
what tasks run or what they run on, so they stay in their own configurations (``STAGE_SETTINGS``). The block is
read once, when the coordinator starts, and applied again to each reload of the configuration files.
"""
import json

from alframework.tools import config as workflow_config
from alframework.tools.capabilities import require, supports

SCALING = 'scaling'
MASTER_SETTINGS = ('parsl_data_serializer', 'poll_interval', 'coordinator_trim_after_snapshot',
                   'disable_transparent_huge_pages', 'label_storage_batch_size', 'label_h5_layout',
                   'data_by_reference', 'trajectory_summary_path', 'coordinator_memory_sample',
                   'coordinator_memory_path', 'coordinator_loop_path', 'write_status_on_change')
SAMPLER_SETTINGS = ('returned_metadata', 'returned_momenta', 'metadata_files')
ML_SETTINGS = ('shared_training_data', 'stream_final_evaluation', 'h5_read_workers')
# Where each sizing setting belongs instead, for the error that names it.
STAGE_SETTINGS = {
    'batching': 'the sampler or QM configuration, each of which has its own',
    'maximum_builder_structures': 'the master configuration, outside scaling',
    'parallel_samplers': 'the master configuration, outside scaling',
    'target_queued_QM': 'the master configuration, outside scaling',
    'gpus_per_member': 'the ML configuration',
    'member_nodes': 'the ML configuration',
    'member_launcher': 'the ML configuration',
}
# A large run, by the structures in flight: from about here, the settings in LARGE_RUN_SETTINGS were measured to
# matter (see the Running at Scale guide).
LARGE_RUN_SAMPLERS = 4096


def scaling_block(master_config):
    """The master configuration's checked ``scaling`` block, or ``None``."""
    block = master_config.get(SCALING)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f'{SCALING} must be a dictionary of settings by their usual names.')
    for name in block:
        if name in STAGE_SETTINGS:
            raise ValueError(f'{name} sizes the work, so it belongs in {STAGE_SETTINGS[name]}, not in {SCALING}.')
        if name not in MASTER_SETTINGS + SAMPLER_SETTINGS + ML_SETTINGS:
            raise ValueError(f'{SCALING} has no setting {name!r}; it takes {list(MASTER_SETTINGS)} (master), '
                             f'{list(SAMPLER_SETTINGS)} (sampler) and {list(ML_SETTINGS)} (ML).')
    return dict(block)


def _resolved(name, value, master_directory):
    """A ``*_path`` setting as ``load_config_file`` gives it: relative to ``master_directory`` unless absolute."""
    if name.endswith('path') and isinstance(value, str) and value and value[0] != '/' and master_directory:
        return master_directory + value
    return value


def _supported(name, master_config):
    """Refuse a sampler or ML setting that the configured task doesn't declare (``alframework.tools.capabilities``)."""
    task = 'sampler_task' if name in SAMPLER_SETTINGS else 'ML_task' if name in ML_SETTINGS else None
    if task is not None:
        require(workflow_config.load_module_from_string(master_config[task]), master_config[task], name,
                f'{SCALING} {name}')


def _same(first, second):
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def apply_scaling(master_config, sampler_config, ML_config, block=None, *, check_tasks=True):
    """Copy each setting of the ``scaling`` block (``block``, or the master configuration's) into the configuration
    that uses it, changing the three dictionaries in place. Returns the block, or ``None`` without one.

    Path settings are resolved as ``load_config_file`` resolves them, with their ``*_dir`` entry, and a setting
    that its own configuration also gives, with a different value, is an error. ``check_tasks`` refuses a sampler
    or ML setting that the configured task doesn't read.
    """
    block = scaling_block(master_config) if block is None else block
    if not block:
        return block
    targets = dict.fromkeys(MASTER_SETTINGS, ('master', master_config))
    targets.update(dict.fromkeys(SAMPLER_SETTINGS, ('sampler', sampler_config)))
    targets.update(dict.fromkeys(ML_SETTINGS, ('ML', ML_config)))
    for name, value in block.items():
        where, config = targets[name]
        value = _resolved(name, value, master_config.get('master_directory'))
        if name in config and not _same(config[name], value):
            raise ValueError(f'{name} is {config[name]!r} in the {where} configuration but {value!r} in {SCALING}; '
                             'set it in one place.')
        if check_tasks:
            _supported(name, master_config)
        config[name] = value
        if name.endswith('path') and isinstance(value, str):
            config[name[:-4] + 'dir'] = '/'.join(value.split('/')[:-1]) + '/'
    return block


def describe(block):
    """One line naming the block's settings, for the coordinator's output."""
    return 'Scaling settings: ' + ', '.join(f'{name}={json.dumps(value)}' for name, value in block.items())


def large_run_warning(master_config, sampler_config, sampler_task=None):
    """For a large run, a warning naming the measured settings it leaves off, or ``None``. The sampler settings are
    named only for a ``sampler_task`` that declares them."""
    samplers = master_config.get('parallel_samplers') or 0
    if not isinstance(samplers, (int, float)) or samplers < LARGE_RUN_SAMPLERS:
        return None
    off = []
    if master_config.get('parsl_data_serializer', 'dill') != 'pickle':
        off.append('parsl_data_serializer "pickle"')
    if master_config.get('label_storage_batch_size') is None:
        off.append('label_storage_batch_size')
    if sampler_task is not None:
        if supports(sampler_task, 'metadata_files') and \
                sampler_config.get('metadata_files', 'per_trajectory') == 'per_trajectory':
            off.append('metadata_files "per_task"')
        if supports(sampler_task, 'returned_metadata') and sampler_config.get('returned_metadata', 'full') != 'compact':
            off.append('returned_metadata "compact"')
    if not off:
        return None
    return (f'A large run (parallel_samplers {samplers:,}) without these settings, which were measured to matter '
            f'at this size: {", ".join(off)}. See "Scaling Settings in One Place" in the Running at Scale guide.')


def setting(master_config, name, default=None):
    """A master setting from the master configuration or its ``scaling`` block, for readers of a run's raw master
    configuration, such as run reports."""
    if name in master_config:
        return master_config[name]
    block = master_config.get(SCALING)
    return block.get(name, default) if isinstance(block, dict) else default
