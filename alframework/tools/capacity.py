"""How much work a run keeps in flight against what its Parsl executors can run, checked at startup.

ALF's queue settings count structures (``parallel_samplers``, ``target_queued_QM``, ``retraining.after``) and its
task widths count structures per task (``maximum_builder_structures`` and each ``batching.batch_size``), while the
Parsl configuration decides how many tasks can run at once. :func:`capacity_report` sets them side by side, once,
when the coordinator starts, and warns where they disagree, such as sampler GPUs that ``parallel_samplers`` would
leave idle. It only reports; nothing it finds changes the run.
"""
import math
import shlex

from alframework.tools.retraining import OLDER_THRESHOLD, RETRAINING, describe as describe_retraining, retraining_settings
from alframework.tools.tasks import executor_label

# The launcher options that start several worker pools on each node, by the Parsl launcher that reads them. A
# launcher of another kind can say how many it starts with a ``pools_per_node`` attribute.
POOLS_PER_NODE_OPTIONS = {'SrunLauncher': ('--ntasks-per-node',),
                          'MpiExecLauncher': ('-ppn', '--ppn', '-npernode', '--npernode'),
                          'AprunLauncher': ('-N',)}
SRUN_TASK_OPTIONS = ('--ntasks', '-n')


def _option(words, names):
    """The last value given for any of ``names`` in ``words`` (``--name=value`` or ``--name value``), or ``None``."""
    found = None
    for index, word in enumerate(words):
        name, equals, value = word.partition('=')
        if name in names:
            found = value if equals else (words[index + 1] if index + 1 < len(words) else None)
    return found


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def pools_per_node(provider):
    """How many worker pools ``provider``'s launcher starts on each node: 1 unless the launcher says more.

    Parsl counts one pool per node, so a launcher that starts one per GPU (``srun --ntasks-per-node=8``) runs
    several times the workers Parsl reports. This reads ``SrunLauncher``'s ``--ntasks-per-node`` (or its
    ``--ntasks`` over the block's nodes), ``MpiExecLauncher``'s ``-ppn`` and ``AprunLauncher``'s ``-N`` from their
    overrides, and a ``pools_per_node`` attribute on any other launcher.
    """
    launcher = getattr(provider, 'launcher', None)
    declared = _positive(getattr(launcher, 'pools_per_node', None))
    if declared:
        return declared
    kinds = [kind.__name__ for kind in type(launcher).__mro__]
    names = next((POOLS_PER_NODE_OPTIONS[kind] for kind in kinds if kind in POOLS_PER_NODE_OPTIONS), None)
    overrides = getattr(launcher, 'overrides', '')
    if names is None or not isinstance(overrides, str):
        return 1
    try:
        words = shlex.split(overrides)
    except ValueError:
        return 1
    per_node = _positive(_option(words, names))
    if per_node:
        return per_node
    if 'SrunLauncher' in kinds:
        tasks, nodes = _positive(_option(words, SRUN_TASK_OPTIONS)), _positive(getattr(provider, 'nodes_per_block', 1))
        if tasks and nodes and tasks % nodes == 0:
            return tasks // nodes
    return 1


def workers_per_node(executor):
    """How many workers ``executor`` runs on each node, over every worker pool its launcher starts there, or
    ``None`` when its configuration doesn't say.

    Parsl's ``workers_per_node`` counts one pool, so a launcher that starts one pool per GPU runs several times
    that many (:func:`pools_per_node`).
    """
    workers = getattr(executor, 'workers_per_node', None)
    provider = getattr(executor, 'provider', None)
    if provider is None or isinstance(workers, bool) or not isinstance(workers, (int, float)) or not math.isfinite(workers):
        return None
    return int(workers) * pools_per_node(provider)


def executor_slots(executor):
    """How many tasks ``executor`` can run at once, or ``None`` when its configuration doesn't say.

    A thread pool runs ``max_threads``; a high-throughput executor runs :func:`workers_per_node` on each node of
    each block, up to ``max_blocks``.
    """
    threads = getattr(executor, 'max_threads', None)
    if isinstance(threads, int) and not isinstance(threads, bool):
        return threads
    per_node = workers_per_node(executor)
    blocks = getattr(getattr(executor, 'provider', None), 'max_blocks', None)
    if per_node is None or blocks is None:
        return None
    return per_node * int(getattr(executor.provider, 'nodes_per_block', 1) or 1) * int(blocks)


def member_slots(executor, member_nodes=1):
    """How many ensemble members ``executor`` can train at once, each taking one worker and ``member_nodes``
    nodes' worth of its slots, or ``None`` when its configuration doesn't say."""
    slots = executor_slots(executor)
    return None if slots is None else slots // max(1, int(member_nodes))


def check_member_gpus(executor, gpus_per_node, device_string, gpus_per_member=1, member_nodes=1,
                      label=executor_label('ML')):
    """Refuse an ML executor whose members would share a GPU.

    A member on ``member_nodes`` nodes uses ``gpus_per_member / member_nodes`` GPUs of each. With
    ``available_accelerators``, each worker gets its own GPUs, and ``gpus_per_node`` may describe the sampler
    nodes, so only the narrowest worker's GPUs are checked. Otherwise every worker on a node, over all its pools,
    needs its own GPUs out of ``gpus_per_node``. A CPU device, an unknown layout or no ``gpus_per_node`` checks
    nothing.
    """
    per_member = gpus_per_member // max(1, int(member_nodes))
    if str(device_string).lower() == 'cpu':
        return
    accelerators = getattr(executor, 'available_accelerators', None)
    if isinstance(accelerators, int):
        accelerators = [str(index) for index in range(accelerators)]
    if accelerators:
        # Parsl sets each worker's CUDA_VISIBLE_DEVICES to one entry, such as "0,1".
        narrowest = min(len(str(entry).split(',')) for entry in accelerators)
        if narrowest < per_member:
            raise ValueError(f'gpus_per_member is {per_member}, but an {label} worker is given {narrowest} GPU(s) by '
                             'available_accelerators. List each worker\'s GPUs, such as ["0,1", "2,3"].')
        return
    per_node = workers_per_node(executor)
    if not gpus_per_node or per_node is None or per_node * per_member <= gpus_per_node:
        return
    pools = pools_per_node(executor.provider)
    counted = f'{per_node} workers per node' + (f' ({per_node // pools} in each of {pools} pools)' if pools > 1 else '')
    raise ValueError(f'{label} runs {counted} with {per_member} GPU(s) each, but gpus_per_node is {gpus_per_node}, '
                     'so members would share a GPU. Lower max_workers_per_node or the pools per node, or set '
                     'available_accelerators if each worker has its own GPUs.')


def task_slots(labels, executors):
    """Tasks that the executors named in ``labels`` can run at once together, or ``None`` if any can't say."""
    counts = [executor_slots(executors[label]) for label in labels if label in executors]
    if not counts or any(count is None for count in counts):
        return None
    return sum(counts)


def _count(value):
    return 'an unknown number of' if value is None else f'{value:,}'


def capacity_report(master_config, executors, *, builder_labels, builder_width, builder_batches, sampler_labels,
                    sampler_width, qm_labels, qm_width, ml_labels=(), n_models=None, member_nodes=1):
    """Lines describing the run's capacity, and warnings where its settings and executors disagree.

    ``executors`` maps labels to the Parsl configuration's executors; each ``*_labels`` lists the executors a stage's
    tasks go to (an ensemble member takes ``member_nodes`` ML slots' nodes), and each ``*_width`` is structures per
    task. ``builder_batches`` says whether the builder builds
    ``maximum_builder_structures`` structures per task. Returns ``(lines, warnings)``.
    """
    lines, warnings = [], []
    samplers = master_config.get('parallel_samplers')
    sampler_slots = task_slots(sampler_labels, executors)
    line = f'sampling: {_count(sampler_slots)} tasks at once of up to {sampler_width:,} structures'
    if sampler_slots is not None:
        line += f' = {sampler_slots * sampler_width:,}'
    lines.append(line + (f'; parallel_samplers {samplers:,}' if samplers is not None else ''))
    if samplers is not None and sampler_slots is not None and samplers < sampler_slots * sampler_width:
        busy = math.ceil(samplers / sampler_width)
        warnings.append(f'parallel_samplers {samplers:,} fills {busy:,} of {sampler_slots:,} sampler task slots, so '
                        f'{sampler_slots - busy:,} stay idle; {sampler_slots * sampler_width:,} would fill them all.')
    if samplers is not None and sampler_width > 1 and samplers % sampler_width:
        warnings.append(f'parallel_samplers {samplers:,} is not a multiple of the sampler batch size '
                        f'{sampler_width:,}, so one batch at a time leaves partly filled, after max_wait_seconds.')

    if builder_batches and builder_width > 1:
        tasks = math.ceil(samplers / builder_width) if samplers else None
        lines.append(f'builders: {builder_width:,} structures per task' +
                     (f', {tasks:,} tasks to fill sampling' if tasks else ''))
        if sampler_width > 1 and builder_width != sampler_width:
            warnings.append(f'Builder tasks of {builder_width:,} structures (maximum_builder_structures) fill sampler '
                            f'batches of {sampler_width:,} unevenly; the same number gives one builder task per '
                            'sampler task.')
    else:
        lines.append('builders: one structure per task')

    target = master_config.get('target_queued_QM')
    qm_slots = task_slots(qm_labels, executors)
    line = f'QM: {_count(qm_slots)} tasks at once of up to {qm_width:,} structures'
    if qm_slots is not None:
        line += f' = {qm_slots * qm_width:,}'
    lines.append(line + (f'; target_queued_QM {target:,}' if target is not None else ''))
    if target is not None and qm_slots is not None and target < qm_slots * qm_width:
        busy = math.ceil(target / qm_width)
        warnings.append(f'target_queued_QM {target:,} fills {busy:,} of {qm_slots:,} QM task slots, so '
                        f'{qm_slots - busy:,} stay idle; {qm_slots * qm_width:,} would fill them all.')

    if RETRAINING in master_config or OLDER_THRESHOLD in master_config:
        lines.append(describe_retraining(retraining_settings(master_config)))

    if n_models is not None:
        counts = [member_slots(executors[label], member_nodes) for label in ml_labels if label in executors]
        members = None if not counts or any(count is None for count in counts) else sum(counts)
        lines.append(f'ML: {n_models:,} ensemble members, {_count(members)} trained at once')
        if members is not None and members < n_models:
            warnings.append(f'{n_models:,} ensemble members but {members:,} ML task slots, so some members wait for '
                            'others to finish.')
    return lines, warnings
