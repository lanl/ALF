"""Train ensemble members as separate Parsl tasks, so an ensemble can span any number of nodes.

The coordinator still sees one ML task. Its body starts ``train_members`` on the coordinator, which
submits one app per member, retries failed members, and returns when every member has succeeded or
run out of retries. Nothing here imports Torch, except ``member_device`` on a worker.
"""
import concurrent.futures
import math
import multiprocessing
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from alframework.tools.files import write_json_atomic
from alframework.tools.gpu import VISIBILITY_VARIABLES, sampling_device

SHARD_WAIT_SECONDS = 120
SHARD_POLL_SECONDS = 2


def validated_retries(member_retries):
    if isinstance(member_retries, bool) or not isinstance(member_retries, int) or member_retries < 0:
        raise ValueError(f'member_retries must be a nonnegative integer, not {member_retries!r}.')
    return member_retries


def validated_gpus_per_member(gpus_per_member):
    if isinstance(gpus_per_member, bool) or not isinstance(gpus_per_member, int) or gpus_per_member < 1:
        raise ValueError(f'gpus_per_member must be a positive integer, not {gpus_per_member!r}.')
    return gpus_per_member


def single_gpu_members(ML_config):
    """Refuse several GPUs per member in the single-node tasks, which give each member one GPU."""
    if validated_gpus_per_member((ML_config or {}).get('gpus_per_member', 1)) > 1:
        raise ValueError('gpus_per_member above 1 needs a multi-node training task, such as '
                         'train_excited_state_HIPPYNN_ensemble_multinode_task.')


def member_device(device_string, gpus_per_node, gpus_per_member=1):
    """Return this worker's device.

    It comes from the worker's rank, not the member number: members 0 and 4 can land on one node,
    where ``member % 4`` would give them the same GPU. Scheduler masks are left alone. A member
    with several GPUs uses every GPU its worker sees, and gets the first for setup and testing.
    """
    requested = str(device_string).lower()
    if requested == 'cpu':
        return 'cpu'
    if requested not in {'from_multiprocessing', 'cuda', 'none'}:
        raise ValueError("Multi-node training picks each member's GPU from its worker, so device_string must be "
                         f"'from_multiprocessing', 'cuda', or 'cpu', not {device_string!r}.")
    if validated_gpus_per_member(gpus_per_member) > 1:
        return 'cuda:0'
    return sampling_device({'device': 'cuda'}, gpus_per_node or None)


def check_capacity(n_models, gpus_per_node, device_string, executor, gpus_per_member=1):
    """Refuse members sharing a GPU, and warn when members will have to wait for a slot."""
    workers = getattr(executor, 'workers_per_node', None)
    provider = getattr(executor, 'provider', None)
    if provider is None or workers is None or not math.isfinite(workers):
        return
    # With available_accelerators each worker has its own GPUs, and gpus_per_node may describe sampler nodes.
    accelerators = getattr(executor, 'available_accelerators', None)
    if isinstance(accelerators, int):
        accelerators = [str(index) for index in range(accelerators)]
    if str(device_string).lower() == 'cpu':
        pass
    elif accelerators:
        # Parsl sets each worker's CUDA_VISIBLE_DEVICES to one entry, such as "0,1".
        narrowest = min(len(str(entry).split(',')) for entry in accelerators)
        if narrowest < gpus_per_member:
            raise ValueError(f'gpus_per_member is {gpus_per_member}, but an alf_ML_executor worker is given '
                             f'{narrowest} GPU(s) by available_accelerators. List each worker\'s GPUs, such as '
                             '["0,1", "2,3"].')
    elif gpus_per_node and workers * gpus_per_member > gpus_per_node:
        raise ValueError(f'alf_ML_executor has {workers} workers per node with {gpus_per_member} GPU(s) each, but '
                         f'gpus_per_node is {gpus_per_node}, so members would share a GPU. Lower '
                         'max_workers_per_node, or set available_accelerators if each worker has its own GPUs.')
    slots = workers * provider.nodes_per_block * provider.max_blocks
    if slots < n_models:
        print(f'Warning: alf_ML_executor has {slots} slots for {n_models} ensemble members, so some members will '
              'wait for others to finish. Check max_workers_per_node and max_blocks.', flush=True)


def expected_shards(h5_path, current_h5_id):
    """Name every shard the coordinator has stored: ``h5_path.format(i)`` for ``i < current_h5_id``.

    Returns an empty list when either is missing, so callers that don't pass them skip the check.
    """
    if h5_path is None or current_h5_id is None:
        return []
    return [str(h5_path).format(index) for index in range(int(current_h5_id))]


def wait_for_shards(paths, timeout=SHARD_WAIT_SECONDS, poll=SHARD_POLL_SECONDS):
    """Wait until every shard appears in its directory listing, because that's what trainers read.

    Another node's cached listing may not show a shard the coordinator has just written.
    """
    deadline = time.monotonic() + timeout
    while True:
        missing = [path for path in paths if Path(path).name not in os.listdir(Path(path).parent)]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f'{len(missing)} of {len(paths)} HDF5 shards were not visible after {timeout} s, '
                                    f'including {missing[0]}.')
        time.sleep(poll)


def run_member_in_child(function, *args):
    """Run one member in a fresh process, which keeps Torch and its cwd out of the long-lived worker.

    Unlike ``multiprocessing.Pool``, this raises if the child dies, instead of waiting forever.
    """
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(function, *args).result()


def member_directory(ensemble_dir, member):
    return os.path.join(str(ensemble_dir), f'model-{member:02d}')


def archive_failed_member(ensemble_dir, member, attempt):
    """Rename a failed member's directory, out of the ``model-*`` pattern that loaders read.

    A process still running on a lost node keeps writing into the renamed directory.
    """
    directory = Path(member_directory(ensemble_dir, member))
    if directory.exists():
        directory.rename(directory.parent / f'failed-{directory.name}-attempt-{attempt}')


@contextmanager
def member_placement(model_dir, attempt, device):
    """Record where a member ran, beside its directory so that a lost node still leaves a record.

    Trainers refuse a nonempty member directory, so the record can't go inside it. Set
    ``completed`` on the yielded dictionary to record the outcome.
    """
    directory = Path(model_dir)
    path = directory.parent / f'placement-{directory.name}-attempt-{attempt}.json'
    record = dict(member=directory.name, attempt=attempt, host=socket.gethostname(), device=str(device),
                  worker_rank=os.environ.get('PARSL_WORKER_RANK'), slurm_job=os.environ.get('SLURM_JOB_ID'),
                  visibility={key: os.environ.get(key) for key in VISIBILITY_VARIABLES},
                  started=time.time(), finished=None, completed=False)
    write_json_atomic(path, record)
    try:
        yield record
    finally:
        record['finished'] = time.time()
        write_json_atomic(path, record)


def train_members(submit, n_models, ensemble_dir, training_id, member_retries=1):
    """Train every member, retrying failures, and return one boolean per member, in member order.

    ``submit(member, attempt)`` returns a future for whether that attempt finished, with attempts
    counted from 1. An exception, a lost worker, and a ``False`` result all count as failures.
    Members that succeed are never rerun, and one member's failure doesn't stop the others.
    """
    validated_retries(member_retries)
    completed = [False] * n_models
    attempts = [1] * n_models
    finished = 0
    pending = {submit(member, 1): member for member in range(n_models)}
    while pending:
        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            member = pending.pop(future)
            try:
                completed[member] = bool(future.result())
                problem = None if completed[member] else 'failed its completion checks'
            except Exception as exc:
                problem = f'{type(exc).__name__}: {exc}'
            if problem is None:
                finished += 1
                continue
            print(f'Training {training_id:04d}: member {member} attempt {attempts[member]} failed: {problem}', flush=True)
            if attempts[member] > member_retries:
                finished += 1
                continue
            try:
                archive_failed_member(ensemble_dir, member, attempts[member])
                attempts[member] += 1
                pending[submit(member, attempts[member])] = member
            except Exception as exc:
                print(f'Training {training_id:04d}: could not retry member {member}: {exc}', flush=True)
                finished += 1
        retrying = sum(attempts[member] > 1 for member in pending.values())
        print(f'Training {training_id:04d}: {finished}/{n_models} members finished, {retrying} retrying, '
              f'{finished - sum(completed)} failed', flush=True)
    return completed


def train_ensemble_across_nodes(submit, n_models, ensemble_dir, training_id, gpus_per_node, device_string,
                                member_retries, executor, finish=None, gpus_per_member=1):
    """Start the body of a multi-node ML task, returning the future that a ``join_app`` waits on.

    The control loop runs on a daemon thread. Python waits for Parsl's internal threads before it runs
    exit hooks, so a loop on one of them would keep a crashed coordinator alive until training ended.
    """
    # Before the thread starts, so that these errors reach the coordinator as they are.
    validated_retries(member_retries)
    validated_gpus_per_member(gpus_per_member)
    check_capacity(n_models, gpus_per_node, device_string, executor, gpus_per_member)
    outcome = concurrent.futures.Future()

    def control_loop():
        try:
            completed = train_members(submit, n_models, ensemble_dir, training_id, member_retries)
            if finish is not None:
                finish()
            outcome.set_result((completed, training_id))
        except Exception as exc:
            outcome.set_exception(exc)

    threading.Thread(target=control_loop, name=f'alf-training-{training_id:04d}', daemon=True).start()
    return outcome
