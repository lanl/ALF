"""Size a run by stating what each task holds and how many labels each round adds; ALF derives the rest.

Without it, a run's size is several settings that have to agree: the sampler and QM ``batching.batch_size``,
``maximum_builder_structures``, ``parallel_samplers``, ``target_queued_QM`` and ``save_h5_threshold``, and the
number of sampler tasks the Parsl configuration runs at once. The master configuration's opt-in ``sizing`` block
states the intent instead::

    "sizing": {"trajectories_per_sampler_task": 256, "structures_per_qm_task": 256, "labels_per_round": 16384}

and ALF fills in (see :func:`derive_sizing`):

- from ``trajectories_per_sampler_task`` (T): the sampler's ``batching.batch_size`` T; ``maximum_builder_structures``
  T, when the builder takes ``moleculeids``, so one builder task fills one sampler task; and ``parallel_samplers``,
  the sampler tasks the Parsl configuration runs at once times T, which keeps every sampler slot busy. Set
  ``sampler_tasks_at_once`` where the Parsl configuration doesn't say, or to use fewer;
- from ``structures_per_qm_task`` (Q): the QM ``batching.batch_size`` Q, with its mode chosen from the QM task;
- from ``labels_per_round`` (L): ``target_queued_QM`` L, and ``save_h5_threshold`` L - 1, since a retraining starts
  once more than ``save_h5_threshold`` labels are held: once L new labels are.

Each key is optional, and fills in only its own settings. A setting given in its own place with the same value is
fine; a different value stops the coordinator, naming both, so neither wins silently. ``minimum_QM``,
``bootstrap_set``, ``maximum_completed_QM`` and ``maximum_stored_labels`` stay the user's. The block is read once,
when the coordinator starts, which prints what it derived, and the same values apply to each reload.
"""

SIZING = 'sizing'
SIZING_KEYS = ('trajectories_per_sampler_task', 'structures_per_qm_task', 'labels_per_round', 'sampler_tasks_at_once')


def sizing_block(master_config):
    """The master configuration's checked ``sizing`` block, or ``None``."""
    block = master_config.get(SIZING)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f'{SIZING} must be a dictionary with keys from {list(SIZING_KEYS)}.')
    unknown = sorted(set(block) - set(SIZING_KEYS))
    if unknown:
        raise ValueError(f'{SIZING} has unknown settings {unknown}; it takes {list(SIZING_KEYS)}.')
    for key, value in block.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{SIZING}.{key} must be a positive integer; received {value!r}.')
    if 'sampler_tasks_at_once' in block and 'trajectories_per_sampler_task' not in block:
        raise ValueError(f'{SIZING}.sampler_tasks_at_once sizes parallel_samplers, which needs '
                         'trajectories_per_sampler_task too.')
    return dict(block)


def derive_sizing(block, *, sampler_slots=None, builder_batches=False, sampler_batches=True):
    """The settings ``block`` fills in, as ``{"master": {...}, "sampler": {...}, "QM": {...}}``.

    ``sampler_slots`` is how many sampler tasks the Parsl configuration runs at once (``None`` if it doesn't say),
    ``builder_batches`` whether the builder takes ``moleculeids``, and ``sampler_batches`` whether the sampler task
    takes ``molecule_objects``.
    """
    derived = {'master': {}, 'sampler': {}, 'QM': {}}
    if not block:
        return derived
    trajectories = block.get('trajectories_per_sampler_task')
    if trajectories is not None:
        if not sampler_batches:
            raise ValueError(f'{SIZING}.trajectories_per_sampler_task needs a sampler task that takes '
                             'molecule_objects, such as alframework.samplers.batched_mlmd.mlmd_trajectories_task.')
        slots = block.get('sampler_tasks_at_once', sampler_slots)
        if slots is None:
            raise ValueError(f'{SIZING} can\'t tell from the Parsl configuration how many sampler tasks run at once; '
                             f'set {SIZING}.sampler_tasks_at_once.')
        derived['sampler']['batch_size'] = trajectories
        derived['master']['parallel_samplers'] = slots * trajectories
        if builder_batches:
            derived['master']['maximum_builder_structures'] = trajectories
    if 'structures_per_qm_task' in block:
        derived['QM']['batch_size'] = block['structures_per_qm_task']
    if 'labels_per_round' in block:
        derived['master']['target_queued_QM'] = block['labels_per_round']
        derived['master']['save_h5_threshold'] = block['labels_per_round'] - 1
    return derived


def _batching(config, batch_size, where):
    block = config.get('batching')
    if block is None:
        config['batching'] = {'batch_size': batch_size}
        return
    if not isinstance(block, dict):
        raise ValueError(f'The {where} configuration\'s batching must be a dictionary.')
    if block.get('batch_size', batch_size) != batch_size:
        raise ValueError(f'batching.batch_size is {block["batch_size"]!r} in the {where} configuration but {SIZING} '
                         f'makes it {batch_size!r}; set it in one place.')
    config['batching'] = dict(block, batch_size=batch_size)


def apply_sizing(derived, master_config, sampler_config, QM_config):
    """Put ``derived`` (from :func:`derive_sizing`) into the configurations, changing them in place; a setting
    already given with a different value is an error."""
    for name, value in derived['master'].items():
        if name in master_config and master_config[name] != value:
            raise ValueError(f'{name} is {master_config[name]!r} in the master configuration but {SIZING} makes it '
                             f'{value!r}; set it in one place.')
        master_config[name] = value
    if 'batch_size' in derived['sampler']:
        _batching(sampler_config, derived['sampler']['batch_size'], 'sampler')
    if 'batch_size' in derived['QM']:
        _batching(QM_config, derived['QM']['batch_size'], 'QM')


def describe(block, derived):
    """One line naming what ``block`` filled in, for the coordinator's output."""
    parts = []
    if 'batch_size' in derived['sampler']:
        parts.append(f'sampler batching.batch_size {derived["sampler"]["batch_size"]:,}')
    for name in ('maximum_builder_structures', 'parallel_samplers'):
        if name in derived['master']:
            parts.append(f'{name} {derived["master"][name]:,}')
    if 'parallel_samplers' in derived['master']:
        slots = derived['master']['parallel_samplers'] // derived['sampler']['batch_size']
        parts[-1] += f' ({slots:,} sampler tasks at once)'
    if 'batch_size' in derived['QM']:
        parts.append(f'QM batching.batch_size {derived["QM"]["batch_size"]:,}')
    if 'target_queued_QM' in derived['master']:
        parts.append(f'target_queued_QM {derived["master"]["target_queued_QM"]:,}')
        parts.append(f'save_h5_threshold {derived["master"]["save_h5_threshold"]:,} (retrain once '
                     f'{block["labels_per_round"]:,} new labels are held)')
    return 'Sizing: ' + ', '.join(parts)
