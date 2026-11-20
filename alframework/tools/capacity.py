"""How much work a run keeps in flight against what its Parsl executors can run, checked at startup.

ALF's queue settings count structures (``parallel_samplers``, ``target_queued_QM``, ``save_h5_threshold``) and its
task widths count structures per task (``maximum_builder_structures`` and each ``batching.batch_size``), while the
Parsl configuration decides how many tasks can run at once. :func:`capacity_report` sets them side by side, once,
when the coordinator starts, and warns where they disagree, such as sampler GPUs that ``parallel_samplers`` would
leave idle. It only reports; nothing it finds changes the run.
"""
import math


def executor_slots(executor):
    """How many tasks ``executor`` can run at once, or ``None`` when its configuration doesn't say.

    A thread pool runs ``max_threads``; a high-throughput executor runs its workers per node on each node of each
    block, up to ``max_blocks``.
    """
    threads = getattr(executor, 'max_threads', None)
    if isinstance(threads, int) and not isinstance(threads, bool):
        return threads
    workers = getattr(executor, 'workers_per_node', None)
    provider = getattr(executor, 'provider', None)
    blocks = getattr(provider, 'max_blocks', None)
    if provider is None or not isinstance(workers, (int, float)) or not math.isfinite(workers) or blocks is None:
        return None
    return int(workers) * int(getattr(provider, 'nodes_per_block', 1) or 1) * int(blocks)


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

    if builder_batches:
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

    threshold = master_config.get('save_h5_threshold')
    if threshold is not None:
        lines.append(f'retraining: once more than {threshold:,} new labels are held')

    if n_models is not None:
        slots = task_slots(ml_labels, executors)
        members = None if slots is None else slots // max(1, member_nodes)
        lines.append(f'ML: {n_models:,} ensemble members, {_count(members)} trained at once')
        if members is not None and members < n_models:
            warnings.append(f'{n_models:,} ensemble members but {members:,} ML task slots, so some members wait for '
                            'others to finish.')
    return lines, warnings
