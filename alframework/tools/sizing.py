"""Size a run by stating what each task holds and how many labels each round adds; ALF derives the rest.

Without it, a run's size is several settings that have to agree: the sampler and QM ``batching.batch_size``,
``maximum_builder_structures``, ``parallel_samplers``, ``target_queued_QM`` and ``retraining.after``, and the
number of sampler tasks the Parsl configuration runs at once. The master configuration's opt-in ``sizing`` block
states the intent instead::

    "sizing": {"trajectories_per_sampler_task": 256, "structures_per_qm_task": 256, "labels_per_round": 16384}

and ALF fills in (see :func:`derive_sizing`):

- from ``trajectories_per_sampler_task`` (T): the sampler's ``batching.batch_size`` T; ``maximum_builder_structures``
  T, when the builder takes ``moleculeids``, so one builder task fills one sampler task; and ``parallel_samplers``,
  the sampler tasks the Parsl configuration runs at once times T, which keeps every sampler slot busy. Set
  ``sampler_tasks_at_once`` where the Parsl configuration doesn't say, or to use fewer;
- from ``structures_per_qm_task`` (Q): the QM ``batching.batch_size`` Q, with its mode chosen from the QM task;
- from ``labels_per_round`` (L): ``retraining`` ``{"after": L}``, so a retraining starts once L new labels are held;
  and ``target_queued_QM`` L, unless ``qm_tasks_at_once`` is set. A ``retraining`` block, or the older
  ``save_h5_threshold``, given as well must agree. To retrain on sampled structures instead
  (:mod:`alframework.tools.retraining`), leave ``labels_per_round`` out and set the ``retraining`` block;
- from ``qm_tasks_at_once`` (N), with ``structures_per_qm_task``: ``target_queued_QM`` N × Q, which keeps N QM tasks
  of Q structures queued or running. It sizes the QM queue for the QM hardware, apart from how often the run
  retrains;
- from ``labels_per_member_gpu`` (G), for an ML task that trains each member across GPUs (one that declares
  ``member_layout``, such as the multi-node HIPPYNN tasks): the ML configuration's ``gpus_per_member`` and
  ``member_nodes``. Every member trains on all of a round's labels, the round's L from ``labels_per_round`` or a
  ``retraining`` block that counts labels, so each member gets ⌈L ÷ G⌉ GPUs, in whole nodes once it needs more than
  one (the master configuration's ``gpus_per_node``). Holding G fixed is weak scaling for training: an epoch does
  the same work per GPU whatever the run's size. ``max_nodes_per_member`` (C) caps a member at C nodes, for where
  members stop scaling; above the cap a member trains with more than G labels per GPU. The coordinator prints how
  many ML nodes the layout needs, for sizing the ML allocation.

Each key is optional, and fills in only its own settings. A setting given in its own place with the same value is
fine; a different value stops the coordinator, naming both, so neither wins silently. ``minimum_QM``,
``bootstrap_set``, ``maximum_completed_QM`` and ``maximum_stored_labels`` stay the user's. The block is read once,
when the coordinator starts, which prints what it derived, and the same values apply to each reload.
"""

from alframework.tools.retraining import OLDER_THRESHOLD, RETRAINING, retraining_settings

SIZING = 'sizing'
SIZING_KEYS = ('trajectories_per_sampler_task', 'structures_per_qm_task', 'labels_per_round', 'sampler_tasks_at_once',
               'qm_tasks_at_once', 'labels_per_member_gpu', 'max_nodes_per_member')
ML_LAYOUT = ('gpus_per_member', 'member_nodes')


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
    if 'qm_tasks_at_once' in block and 'structures_per_qm_task' not in block:
        raise ValueError(f'{SIZING}.qm_tasks_at_once sizes target_queued_QM, which needs structures_per_qm_task too.')
    if 'max_nodes_per_member' in block and 'labels_per_member_gpu' not in block:
        raise ValueError(f'{SIZING}.max_nodes_per_member caps the member layout labels_per_member_gpu derives; set '
                         'labels_per_member_gpu too.')
    return dict(block)


def derive_sizing(block, *, sampler_slots=None, builder_batches=False, sampler_batches=True, round_labels=None,
                  gpus_per_node=None, n_models=None, member_layout=True):
    """The settings ``block`` fills in, as ``{"master": {...}, "sampler": {...}, "QM": {...}, "ML": {...}}``.

    ``sampler_slots`` is how many sampler tasks the Parsl configuration runs at once (``None`` if it doesn't say),
    ``builder_batches`` whether the builder takes ``moleculeids``, and ``sampler_batches`` whether the sampler task
    takes ``molecule_objects``. For ``labels_per_member_gpu``: ``round_labels`` is the labels a round adds when the
    block has no ``labels_per_round`` (``None`` if the run doesn't count labels), ``gpus_per_node`` and ``n_models``
    come from the master and ML configurations, and ``member_layout`` says whether the ML task trains each member
    across GPUs. ``ML`` also holds ``ml_nodes``, the nodes the layout needs, which is reported and not applied.
    """
    derived = {'master': {}, 'sampler': {}, 'QM': {}, 'ML': {}}
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
        derived['master'][RETRAINING] = {'after': block['labels_per_round'], 'counts': 'labels'}
    if 'qm_tasks_at_once' in block:
        derived['master']['target_queued_QM'] = block['qm_tasks_at_once'] * block['structures_per_qm_task']
    if 'labels_per_member_gpu' in block:
        derived['ML'] = _member_layout(block, block.get('labels_per_round', round_labels), gpus_per_node, n_models,
                                       member_layout)
    return derived


def _member_layout(block, labels, gpus_per_node, n_models, member_layout):
    """``gpus_per_member``, ``member_nodes`` and the ``ml_nodes`` they need, from ``labels_per_member_gpu``."""
    if not member_layout:
        raise ValueError(f'{SIZING}.labels_per_member_gpu sizes members that train across GPUs; the ML task must '
                         'declare member_layout, as alframework.ml_interfaces.hippynn.interface.'
                         'train_HIPPYNN_ensemble_multinode_task does.')
    if labels is None:
        raise ValueError(f'{SIZING}.labels_per_member_gpu needs the labels each round adds: set {SIZING}.'
                         'labels_per_round, or a retraining block that counts labels.')
    for name, value in (('gpus_per_node', gpus_per_node), ('n_models', n_models)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{SIZING}.labels_per_member_gpu needs a positive integer {name} (in the '
                             f'{"master" if name == "gpus_per_node" else "ML"} configuration); received {value!r}.')
    wanted = -(-labels // block['labels_per_member_gpu'])
    nodes = -(-wanted // gpus_per_node)
    cap = block.get('max_nodes_per_member')
    capped = cap is not None and nodes > cap
    if capped:
        nodes = cap
    gpus = min(wanted, gpus_per_node) if nodes == 1 else nodes * gpus_per_node
    return {'gpus_per_member': gpus, 'member_nodes': nodes, 'ml_nodes': -(-n_models * gpus // gpus_per_node),
            'labels_per_gpu': -(-labels // gpus), 'capped': capped}


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


def _retraining(master_config, derived):
    """Add the derived ``retraining`` block, or check the one (or the older ``save_h5_threshold``) already given."""
    if RETRAINING not in master_config and OLDER_THRESHOLD not in master_config:
        master_config[RETRAINING] = dict(derived)
        return
    given = retraining_settings(master_config)
    if given != derived:
        where = RETRAINING if RETRAINING in master_config else OLDER_THRESHOLD
        unit = {'labels': 'labels', 'sampled': 'sampled structures'}
        raise ValueError(f'The master configuration\'s {where} retrains after {given["after"]:,} {unit[given["counts"]]}, '
                         f'but {SIZING} makes it {derived["after"]:,} {unit[derived["counts"]]}; set it in one place.')


def apply_sizing(derived, master_config, sampler_config, QM_config, ML_config=None):
    """Put ``derived`` (from :func:`derive_sizing`) into the configurations, changing them in place; a setting
    already given with a different value is an error."""
    if derived.get('ML'):
        if ML_config is None:
            raise ValueError(f'{SIZING}.labels_per_member_gpu sets the ML configuration, which wasn\'t given.')
        for name in ML_LAYOUT:
            value = derived['ML'][name]
            if name in ML_config and ML_config[name] != value:
                raise ValueError(f'{name} is {ML_config[name]!r} in the ML configuration but {SIZING} makes it '
                                 f'{value!r}; set it in one place.')
            ML_config[name] = value
    for name, value in derived['master'].items():
        if name == RETRAINING:
            _retraining(master_config, value)
            continue
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
        line = f'target_queued_QM {derived["master"]["target_queued_QM"]:,}'
        if 'qm_tasks_at_once' in block:
            line += f' ({block["qm_tasks_at_once"]:,} QM tasks at once)'
        parts.append(line)
    if RETRAINING in derived['master']:
        retraining = derived['master'][RETRAINING]
        parts.append(f'retraining.after {retraining["after"]:,} (retrain once {retraining["after"]:,} new labels are '
                     'held)')
    layout = derived.get('ML')
    if layout:
        line = (f'gpus_per_member {layout["gpus_per_member"]:,} and member_nodes {layout["member_nodes"]:,} '
                f'({layout["labels_per_gpu"]:,} labels per member GPU; the ML executor needs {layout["ml_nodes"]:,} '
                f'node{"" if layout["ml_nodes"] == 1 else "s"}')
        if layout['capped']:
            line += f', with members capped at {SIZING}.max_nodes_per_member {block["max_nodes_per_member"]:,}'
        parts.append(line + ')')
    return 'Sizing: ' + ', '.join(parts)
