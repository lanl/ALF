"""A per-task timeline: when each Parsl task ran, where, and how it ended, for measuring stage times.

Set ``ALF_TASK_TIMELINE_DIR`` in the workers' environment (``worker_init``) to turn it on; unset, ``timed`` adds
nothing. Each task appends ONE JSON line, when it ends, to ``<dir>/<host>-<pid>.jsonl``: one file per worker
process, so thousands of workers never share a file and no locking is needed on a parallel filesystem. Fields:

``stage`` (builder, sampler, QM, ML), ``task`` (function name), ``items`` (structures in the task's first
argument, 1 for a single one), ``start``/``end`` (Unix seconds) and ``seconds``, ``outcome`` (``ok`` or the
exception's class name), ``host``, ``pid``, ``slurm_job``, ``parsl_block``, ``parsl_pool``, ``parsl_rank``, and
``gpu_peak_gb``: the most GPU memory PyTorch had allocated during the task, on the device that used the most, in
a worker that uses a GPU through PyTorch (``None`` otherwise).

Stage times then come from the tasks themselves (first start to last end), rather than from Parsl's file
timestamps, which depend on job names and count a block's idle time.
"""
import functools
import json
import os
import socket
import sys
import threading
import time

ENV = 'ALF_TASK_TIMELINE_DIR'
# Every timed task also applies the workers' huge-page setting (alframework.tools.memory), once per process.
HUGE_PAGES_ENV = 'ALF_THP_DISABLE'
# Tasks that call another task's ``.func`` (UDD sampling calls MLMD sampling) are recorded once, as the
# outer task: the inner call is part of the outer task's time. The flag lives on the running thread, not in
# this module: Parsl serializes a decorated task's wrapper by value, with this module's globals, so they must
# all be picklable (a threading.local is not).
_FLAG = '_alf_timed_task_running'


def _begin_outermost():
    """Whether this call is the outermost timed task in its thread; if so, mark one as running."""
    thread = threading.current_thread()
    if getattr(thread, _FLAG, False):
        return False
    setattr(thread, _FLAG, True)
    return True


def _end_outermost():
    setattr(threading.current_thread(), _FLAG, False)


def _items(args, kwargs):
    # ALF passes builders both ``moleculeid`` and ``moleculeids``, by keyword; a multi-structure builder
    # builds all of ``moleculeids``.
    ids = kwargs.get('moleculeids')
    if isinstance(ids, (list, tuple)):
        return len(ids)
    first = args[0] if args else next(iter(kwargs.values()), None)
    return len(first) if isinstance(first, (list, tuple)) else 1


def _torch_cuda():
    """``torch.cuda`` if this process already uses a GPU through PyTorch, else ``None``; never imports torch."""
    torch = sys.modules.get('torch')
    try:
        return torch.cuda if torch is not None and torch.cuda.is_initialized() else None
    except Exception:
        return None


def _reset_gpu_peak():
    cuda = _torch_cuda()
    try:
        for device in range(cuda.device_count() if cuda is not None else 0):
            cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def _gpu_peak_gb():
    cuda = _torch_cuda()
    try:
        if cuda is not None and cuda.device_count():
            return round(max(cuda.max_memory_allocated(device) for device in range(cuda.device_count())) / 2**30, 3)
    except Exception:
        pass
    return None


def write_record(directory, record):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{record['host']}-{record['pid']}.jsonl")
    with open(path, 'a') as handle:
        handle.write(json.dumps(record) + '\n')


def timed(stage):
    """Decorate a task function (under ``@python_app``) so each call adds a line to the task timeline.

    With ``ALF_THP_DISABLE=1`` in the worker's environment, the task first turns off transparent huge pages for
    its worker and the processes the task starts (:func:`alframework.tools.memory.disable_transparent_huge_pages`).
    """
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if os.environ.get(HUGE_PAGES_ENV, '') not in ('', '0'):
                from alframework.tools.memory import disable_transparent_huge_pages_if_requested
                disable_transparent_huge_pages_if_requested()
            directory = os.environ.get(ENV)
            if not directory or not _begin_outermost():
                return function(*args, **kwargs)
            _reset_gpu_peak()
            start, outcome = time.time(), 'ok'
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                outcome = type(exc).__name__
                raise
            finally:
                _end_outermost()
                end = time.time()
                try:
                    write_record(directory, {
                        'stage': stage, 'task': function.__name__, 'items': _items(args, kwargs),
                        'start': start, 'end': end, 'seconds': end - start, 'outcome': outcome,
                        'host': socket.gethostname(), 'pid': os.getpid(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                        'parsl_block': os.environ.get('PARSL_WORKER_BLOCK_ID'),
                        'parsl_pool': os.environ.get('PARSL_WORKER_POOL_ID'),
                        'parsl_rank': os.environ.get('PARSL_WORKER_RANK'), 'gpu_peak_gb': _gpu_peak_gb()})
                except OSError:
                    pass  # A timeline that cannot be written must never fail the task itself.
        return wrapper
    return decorator


def read_timeline(directory):
    """Every record under ``directory``, oldest start first.

    A line that isn't a complete record, such as the last line of a worker killed while writing it, is skipped.
    """
    records = []
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if name.endswith('.jsonl'):
            with open(os.path.join(directory, name)) as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and 'start' in record:
                        records.append(record)
    return sorted(records, key=lambda record: record['start'])


def stage_spans(records):
    """Per stage: tasks, items, first start, last end and wall span in seconds, and task seconds summed."""
    spans = {}
    for record in records:
        span = spans.setdefault(record['stage'], {'tasks': 0, 'items': 0, 'failed': 0, 'first_start': record['start'],
                                                   'last_end': record['end'], 'task_seconds': 0.0})
        span['tasks'] += 1
        span['items'] += record['items']
        span['failed'] += record['outcome'] != 'ok'
        span['first_start'] = min(span['first_start'], record['start'])
        span['last_end'] = max(span['last_end'], record['end'])
        span['task_seconds'] += record['seconds']
    for span in spans.values():
        span['wall_seconds'] = span['last_end'] - span['first_start']
    return spans
