"""Train ensemble members as separate Parsl tasks, so an ensemble can span any number of nodes.

The coordinator still sees one ML task. Its body starts ``train_members`` on the coordinator, which
submits one app per member, retries failed members, and returns when every member has succeeded or
run out of retries. Nothing here imports Torch, except ``member_device`` on a worker.
"""
import concurrent.futures
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


def validated_member_nodes(member_nodes, gpus_per_member=1):
    """``member_nodes`` after checking it, and that it shares ``gpus_per_member`` evenly."""
    if isinstance(member_nodes, bool) or not isinstance(member_nodes, int) or member_nodes < 1:
        raise ValueError(f'member_nodes must be a positive integer, not {member_nodes!r}.')
    if member_nodes > 1 and (validated_gpus_per_member(gpus_per_member) == 1 or gpus_per_member % member_nodes):
        raise ValueError(f'gpus_per_member {gpus_per_member} is a total over member_nodes {member_nodes}, so it must '
                         'be above 1 and divide evenly by it: a member on several nodes trains on all their GPUs.')
    return member_nodes


def check_capacity(n_models, gpus_per_node, device_string, executor, gpus_per_member=1, member_nodes=1):
    """Refuse members sharing a GPU, and warn when members will have to wait for a slot.

    A member on ``member_nodes`` nodes uses ``gpus_per_member / member_nodes`` GPUs of each, and its
    worker starts it on the others with ``member_launcher``, so each such member takes one worker and
    ``member_nodes`` nodes of the executor's blocks. Both counts come from :mod:`alframework.tools.capacity`,
    as the startup capacity report's do.
    """
    from alframework.tools.capacity import check_member_gpus, member_slots
    from alframework.tools.tasks import executor_label
    validated_member_nodes(member_nodes, gpus_per_member)
    check_member_gpus(executor, gpus_per_node, device_string, gpus_per_member, member_nodes)
    slots = member_slots(executor, member_nodes)
    if slots is not None and slots < n_models:
        print(f'Warning: {getattr(executor, "label", executor_label("ML"))} has {slots} slots for {n_models} ensemble members, so some members will '
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
        listings = {parent: set(os.listdir(parent)) for parent in {Path(path).parent for path in paths}}
        missing = [path for path in paths if Path(path).name not in listings[Path(path).parent]]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f'{len(missing)} of {len(paths)} HDF5 shards were not visible after {timeout} s, '
                                    f'including {missing[0]}.')
        time.sleep(poll)


def training_shards(h5_dir, h5_path=None, current_h5_id=None, *, targets=None):
    """Freeze and validate a round's label inventory before submitting members."""
    from alframework.tools.label_hdf5 import discover_label_shards, validate_training_units

    expected = expected_shards(h5_path, current_h5_id)
    if expected:
        wait_for_shards(expected)
    paths = discover_label_shards(h5_dir)
    if not paths:
        raise FileNotFoundError(f'No ALF HDF5 shards were found in {str(h5_dir)!r}.')
    validate_training_units(paths, targets)
    return paths


def run_member_in_child(function, *args):
    """Run one member in a fresh process, which keeps Torch and its cwd out of the long-lived worker.

    Unlike ``multiprocessing.Pool``, this raises if the child dies, instead of waiting forever.
    """
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(function, *args).result()


def run_members_in_slots(function, parameters, devices):
    """One supervised child at a time per device; native death cannot block a sibling's slot."""
    if not devices:
        raise ValueError('Training needs at least one execution slot.')
    completed = [False] * len(parameters)

    def run_slot(slot):
        for member in range(slot, len(parameters), len(devices)):
            params = dict(parameters[member], device_string=devices[slot])
            try:
                completed[member] = bool(run_member_in_child(function, params))
            except Exception as error:
                print(f'Training member {member} failed: {type(error).__name__}: {error}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(run_slot, range(len(devices))))
    return completed


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
                                member_retries, executor, finish=None, gpus_per_member=1, member_nodes=1):
    """Start the body of a multi-node ML task, returning the future that a ``join_app`` waits on.

    The control loop runs on a daemon thread. Python waits for Parsl's internal threads before it runs
    exit hooks, so a loop on one of them would keep a crashed coordinator alive until training ended.
    """
    # Before the thread starts, so that these errors reach the coordinator as they are.
    validated_retries(member_retries)
    validated_gpus_per_member(gpus_per_member)
    check_capacity(n_models, gpus_per_node, device_string, executor, gpus_per_member, member_nodes)
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
