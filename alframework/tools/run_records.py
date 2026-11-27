"""Read what an ALF run records about itself, as tables, for run reports (:mod:`alframework.tools.run_report`).

Every reader returns an empty table when its record is missing, so a report shows what a run has recorded and
says what it lacks. The records:

- the task timeline (``ALF_TASK_TIMELINE_DIR``, :mod:`alframework.tools.task_timeline`): one row per task;
- Parsl's block submissions (``runinfo/*/submit_scripts``): when each executor asked for a block;
- the trajectory summary (``trajectory_summary_path``): one row per trajectory, with its model and spread;
- the stored labels (``h5_path``): one shard per training round;
- the members' ``training_summary.json`` and ``lightning_stats`` (``model_path``): one row per member per round;
- the coordinator memory samples (``coordinator_memory_path``, :mod:`alframework.tools.memory_sampler`);
- the coordinator's loop passes (``coordinator_loop_path``, :mod:`alframework.tools.loop_timing`).

A task's *cycle* is the model the run had when the task started: the latest training round finished by then
(see :func:`assign_cycles`), the same number as a trajectory's ``model``.
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from alframework.tools.task_timeline import read_timeline
from alframework.tools.tasks import executor_stages
from alframework.tools.trajectory_summary import COLUMNS as TRAJECTORY_COLUMNS, read_trajectory_summary

# The stages whose tasks ALF submits to each executor, under its current or a former label (alframework.tools.tasks).
EXECUTOR_STAGES = executor_stages()
TASK_COLUMNS = ('stage', 'task', 'items', 'start', 'end', 'seconds', 'outcome', 'host', 'slurm_job', 'parsl_block',
                'gpu_peak_gb')
SUBMIT_SCRIPT = re.compile(r'parsl\.(?P<executor>[\w-]+)\.block-(?P<block>\d+)\.(?P<time>\d+(?:\.\d+)?)$')
ROUND_DIRECTORY = re.compile(r'(\d+)$')


def load_master_config(run_dir, name='master_config.json'):
    """The run's master config, or ``{}`` if it has none under that name."""
    path = Path(run_dir) / name
    return json.loads(path.read_text()) if path.is_file() else {}


def _resolve(run_dir, path):
    return None if path is None else str(Path(run_dir) / path)


def tasks(timeline_dir):
    """One row per recorded task, oldest start first; an empty table without a timeline."""
    records = read_timeline(timeline_dir) if timeline_dir else []
    table = pd.DataFrame(records, columns=list(TASK_COLUMNS) if not records else None)
    for column in TASK_COLUMNS:
        if column not in table:
            table[column] = None
    table['gpu_peak_gb'] = pd.to_numeric(table['gpu_peak_gb'], errors='coerce')
    return table


def block_submissions(run_dir):
    """One row per Parsl block a run asked for: ``executor``, ``block``, ``submitted`` (Unix seconds)."""
    rows = []
    for script in glob.glob(os.path.join(run_dir, 'runinfo', '*', 'submit_scripts', 'parsl.*.block-*')):
        found = SUBMIT_SCRIPT.search(os.path.basename(script))
        if found:
            rows.append(dict(executor=found['executor'], block=int(found['block']), submitted=float(found['time'])))
    return pd.DataFrame(rows, columns=['executor', 'block', 'submitted']).sort_values('submitted', ignore_index=True)


def block_waits(blocks, task_table):
    """For each block of an ALF executor: from its submission to the first task of its stages that started after it.

    That wait is the scheduler's queue plus the block's startup. A block whose stages ran no task after it was
    submitted has no wait (``NaN``).
    """
    rows = []
    for block in blocks.itertuples():
        stages = EXECUTOR_STAGES.get(block.executor)
        if stages is None:
            continue
        later = task_table[task_table['stage'].isin(stages) & (task_table['start'] >= block.submitted)]
        first = later['start'].min() if not later.empty else np.nan
        rows.append(dict(executor=block.executor, stage='/'.join(stages), block=block.block,
                         submitted=block.submitted, first_task=first, wait_seconds=first - block.submitted))
    return pd.DataFrame(rows, columns=['executor', 'stage', 'block', 'submitted', 'first_task', 'wait_seconds'])


def trajectories(path):
    """One row per trajectory from the trajectory summary, with numeric spreads; empty without one."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(TRAJECTORY_COLUMNS))
    table = pd.DataFrame(read_trajectory_summary(path))
    table = table[table['model'].notna()].copy()
    table['model'] = table['model'].astype(int)
    for name in ('simtime', 'realtime', 'Es', 'Fs', 'Fsmax', 'peak_Es', 'peak_Fs', 'peak_Fsmax', 'gap_std',
                 'peak_gap_std'):
        table[name] = pd.to_numeric(table[name], errors='coerce')
    return table


FORMAT_FIELD = re.compile(r'\{[^{}]*\}')


def numbered_paths(pattern):
    """``(number, path)`` for each existing path that ``pattern.format(number)`` names, in order.

    Found by matching rather than counting up, so a round removed after the run (such as a copied starting
    ensemble) leaves a gap instead of hiding the rounds after it.
    """
    if not pattern or len(FORMAT_FIELD.findall(pattern)) != 1:
        return []
    found = []
    for path in glob.glob(glob.escape(FORMAT_FIELD.sub('\0', pattern)).replace('\0', '*')):
        for number in re.findall(r'\d+', path[len(FORMAT_FIELD.split(pattern)[0]):]):
            if pattern.format(int(number)) == path:
                found.append((int(number), path))
                break
    return sorted(found)


def labels_per_round(h5_pattern):
    """Labels stored for each training round: one row per existing ``h5_path.format(round)`` shard."""
    from alframework.tools.restarts import shard_frames
    rows = []
    for index, path in numbered_paths(h5_pattern):
        try:
            frames = shard_frames(path)
        except (OSError, ValueError, KeyError):
            frames = np.nan  # Still being written, or not a label shard.
        rows.append(dict(round=index, labels=frames, stored=os.path.getmtime(path)))
    table = pd.DataFrame(rows, columns=['round', 'labels', 'stored'])
    table['cumulative_labels'] = table['labels'].cumsum()
    return table


def _flat_metrics(prefix, metrics):
    return {f'{prefix}{split}.{name}': value for split, values in (metrics or {}).items()
            for name, value in (values or {}).items() if isinstance(value, (int, float)) or value is None}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _lightning_stats(member_dir):
    """A member's per-rank Lightning records (``ALF_LIGHTNING_STATS=1``), summed up: the spread of the ranks'
    median step times, their peak memory, and rank 0's epoch times."""
    ranks = [rank for rank in (_read_json(path) for path in
                               glob.glob(os.path.join(member_dir, 'lightning_stats', 'rank-*.json'))) if rank]
    if not ranks:
        return {}
    def values(key):
        return np.array([rank[key] for rank in ranks if isinstance(rank.get(key), (int, float))], dtype=float)
    medians, peak_rss, gpu_peak = values('step_median'), values('peak_rss_gib'), values('gpu_peak_gib')
    first = min(ranks, key=lambda rank: rank.get('rank', 0))
    return dict(ranks=len(ranks), rank_epoch_seconds=first.get('epoch_seconds'),
                step_median_min=medians.min() if medians.size else np.nan,
                step_median_max=medians.max() if medians.size else np.nan,
                rank_peak_rss_gib=peak_rss.max() if peak_rss.size else np.nan,
                rank_gpu_peak_gib=gpu_peak.max() if gpu_peak.size else np.nan)


def training(model_pattern):
    """One row per member per training round, from its ``training_summary.json`` and ``lightning_stats``.

    Columns include ``round``, ``member``, ``epochs``, ``average_epoch_time``, ``last_epoch_time``, ``finished``
    (when its record was written), ``final.<split>.<metric>`` (the final test of the best model) and
    ``best.<split>.<metric>``. A member without a summary (written before summaries existed) still gets its epoch
    times from ``lightning_stats`` if those were kept.
    """
    rows = []
    for index, directory in numbered_paths(model_pattern):
        for member_dir in sorted(glob.glob(os.path.join(directory, 'model-*'))):
            summary_path = os.path.join(member_dir, 'training_summary.json')
            summary = _read_json(summary_path) or {}
            stats = _lightning_stats(member_dir)
            if not summary and not stats:
                continue
            epoch_seconds = summary.get('epoch_seconds') or stats.get('rank_epoch_seconds') or []
            record = summary_path if summary else os.path.join(member_dir, 'lightning_stats')
            member = ROUND_DIRECTORY.search(os.path.basename(member_dir))
            row = dict(round=index, member=int(member[1]) if member else os.path.basename(member_dir),
                       epochs=summary.get('epochs', len(epoch_seconds)),
                       average_epoch_time=summary.get('average_epoch_time',
                                                      float(np.mean(epoch_seconds)) if epoch_seconds else np.nan),
                       last_epoch_time=epoch_seconds[-1] if epoch_seconds else np.nan,
                       gpus_per_member=summary.get('gpus_per_member', stats.get('ranks', 1)),
                       finished=os.path.getmtime(record))
            row.update(_flat_metrics('final.', summary.get('final_metric')))
            row.update(_flat_metrics('best.', summary.get('metric')))
            row.update({key: value for key, value in stats.items() if key != 'rank_epoch_seconds'})
            rows.append(row)
    return pd.DataFrame(rows)


def round_finish_times(training_table):
    """When each training round finished: its last member's summary."""
    if training_table.empty:
        return pd.Series(dtype=float)
    return training_table.groupby('round')['finished'].max().sort_index()


def assign_cycles(times, finishes):
    """The cycle of each time in ``times``: the latest training round finished by then, from ``finishes`` (round
    number to finish time). Before the first recorded round finished, it's the round before it: -1 for a run that
    trained its first ensemble, 0 for one that started from a copied ensemble 0 and first trained round 1."""
    times = np.asarray(times, dtype=float)
    if len(finishes) == 0:
        return np.full(len(times), -1, dtype=int)
    finishes = finishes.sort_values()
    rounds = np.concatenate([[int(finishes.index.min()) - 1], finishes.index.to_numpy(dtype=int)])
    return rounds[np.searchsorted(finishes.to_numpy(dtype=float), times, side='right')]


def coordinator_memory(path):
    """The coordinator's memory over time, per sample: ``time``, ``coordinator_gib`` (the coordinator's RSS) and
    ``tree_gib`` (PSS summed over it and its children); empty without samples."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['time', 'coordinator_gib', 'tree_gib'])
    samples = pd.read_csv(path)
    if samples.empty:
        return pd.DataFrame(columns=['time', 'coordinator_gib', 'tree_gib'])
    samples['ppid'] = pd.to_numeric(samples['ppid'], errors='coerce')
    root = samples.loc[~samples['ppid'].isin(samples['pid']), 'pid'].mode()
    root = root.iloc[0] if not root.empty else samples['pid'].iloc[0]
    tree = samples.groupby('time')['pss_kib'].sum() / 2**20
    coordinator = samples[samples['pid'] == root].groupby('time')['rss_kib'].max() / 2**20
    return pd.DataFrame({'coordinator_gib': coordinator, 'tree_gib': tree}).reset_index()


def coordinator_loop(path):
    """The coordinator's loop passes (:mod:`alframework.tools.loop_timing`), one row each; empty without them."""
    from alframework.tools.loop_timing import COLUMNS
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(COLUMNS))
    passes = pd.read_csv(path)
    for column in COLUMNS:
        if column != 'loop':
            passes[column] = pd.to_numeric(passes[column], errors='coerce') if column in passes else np.nan
    return passes.dropna(subset=['start', 'seconds']).sort_values('start', kind='stable').reset_index(drop=True)


def run_records(run_dir, master_config=None, timeline_dir=None):
    """Every record of a run as a dict of tables, with the paths it read under ``paths``."""
    run_dir = str(run_dir)
    master = master_config if master_config is not None else load_master_config(run_dir)
    if timeline_dir is None:
        timeline_dir = (_resolve(run_dir, master.get('task_timeline_dir')) or os.environ.get('ALF_TASK_TIMELINE_DIR')
                        or os.path.join(run_dir, 'task_timeline'))
    # Monitoring paths may be in the master configuration's scaling block (alframework.tools.scaling).
    from alframework.tools.scaling import setting
    paths = dict(timeline=timeline_dir if os.path.isdir(timeline_dir) else None,
                 trajectories=_resolve(run_dir, setting(master, 'trajectory_summary_path')),
                 labels=_resolve(run_dir, master.get('h5_path')),
                 models=_resolve(run_dir, master.get('model_path')),
                 memory=_resolve(run_dir, setting(master, 'coordinator_memory_path', 'coordinator_memory.csv')),
                 loop=_resolve(run_dir, setting(master, 'coordinator_loop_path')))
    task_table = tasks(paths['timeline'])
    training_table = training(paths['models'])
    finishes = round_finish_times(training_table)
    if not task_table.empty:
        task_table['cycle'] = assign_cycles(task_table['start'], finishes)
    blocks = block_submissions(run_dir)
    memory = coordinator_memory(paths['memory'])
    loop = coordinator_loop(paths['loop'])
    # The run's start: its first record of any kind.
    starts = [table[column].min() for table, column in ((task_table, 'start'), (blocks, 'submitted'),
                                                         (memory, 'time'), (loop, 'start')) if not table.empty]
    return dict(paths=paths, tasks=task_table, blocks=blocks, block_waits=block_waits(blocks, task_table),
                trajectories=trajectories(paths['trajectories']), labels=labels_per_round(paths['labels']),
                training=training_table, round_finishes=finishes, memory=memory, loop=loop,
                run_start=float(min(starts)) if starts else None)
