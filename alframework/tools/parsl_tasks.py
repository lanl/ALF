"""Track Parsl futures and manage executor shutdown."""

import atexit
import logging
import os
import signal
import sys
import threading

logger = logging.getLogger(__name__)

SUCCESSFUL_TASK_STATUSES = ("exec_done", "memo_done")
FAILED_TASK_STATUSES = ("failed", "dep_fail")
FINAL_TASK_STATUSES = SUCCESSFUL_TASK_STATUSES + FAILED_TASK_STATUSES

# Seconds Parsl's cleanup gets at exit once ALF has cancelled the blocks itself.
CLEANUP_TIMEOUT = float(os.environ.get('ALF_PARSL_CLEANUP_TIMEOUT', '120'))


def _is_parsl_future(task):
    """Parsl's AppFuture, whose running() is never true (Parsl doesn't mark its futures running)."""
    try:
        from parsl.dataflow.futures import AppFuture
    except ImportError:
        return False
    return isinstance(task, AppFuture)


def is_final_task(task):
    """
    Helper to determine if a Parsl task has reached a final state.
    Final states have either completed successfully or permanently failed.
    """
    return task.task_status() in FINAL_TASK_STATUSES


class ParslTaskQueue:
    """Tasks the coordinator submitted and hasn't harvested yet, with counts by state.

    Tasks that support ``add_done_callback`` (Parsl's futures) report their final state once, when they finish,
    so a count costs the same whatever the number of tasks; the coordinator counts several times every poll.
    Other tasks (test doubles) are checked one by one, as before. Results are harvested in the same order either way.
    """

    def __init__(self):
        self.task_list = []
        # The tasks the last get_task_results counted as failed.
        self.last_failed = []
        self._lock = threading.Lock()
        self._finished = {}  # id(task) -> final state, for tracked tasks not harvested yet
        self._sums = {'final': 0, 'exec_done': 0, 'failed': 0}  # weights of the finished tracked tasks, by state
        self._untracked = {}  # id(task) -> task, for tasks without add_done_callback
        self._may_run = {}  # id(task) -> task whose running() can be true (not Parsl's AppFuture)
        self._total = 0  # weight of every task in task_list

    def _weight(self, task):
        """What a task counts as: one task here; the structures it holds in a batched queue."""
        return 1

    def add_task(self,task):
        """Add a task to the task list
        """
        self.task_list.append(task)
        self._track(task)

    def _track(self, task):
        self._total += self._weight(task)
        if not _is_parsl_future(task):
            self._may_run[id(task)] = task
        if hasattr(task, 'add_done_callback') and hasattr(task, 'task_status'):
            task.add_done_callback(self._on_done)  # runs at once if it's already done
        else:
            self._untracked[id(task)] = task

    def _on_done(self, task):
        status = task.task_status()  # Parsl sets the final state before it completes the future
        weight = self._weight(task)
        with self._lock:
            if id(task) in self._finished:
                return
            self._finished[id(task)] = status
            self._sums['final'] += weight
            if status == 'exec_done':
                self._sums['exec_done'] += weight
            if status in FAILED_TASK_STATUSES:
                self._sums['failed'] += weight

    def _count(self, kind, check):
        with self._lock:
            tracked = self._sums[kind]
        return tracked + sum(self._weight(task) for task in list(self._untracked.values()) if check(task))

    def _take_finished(self, reverse=False, max_items=None):
        """Remove finished tasks in queue order, allowing one oversized task to make progress."""
        with self._lock:
            finished = dict(self._finished)
        untracked = {key for key, task in self._untracked.items() if is_final_task(task)}
        if not finished and not untracked:
            return []
        taken = []
        items = 0
        for task in (reversed(self.task_list) if reverse else self.task_list):
            key = id(task)
            if key not in finished and key not in untracked:
                continue
            weight = self._weight(task)
            if max_items is not None and taken and items + weight > max_items:
                break
            status = finished[key] if key in finished else task.task_status()
            taken.append((task, status))
            items += weight
        gone = {id(task) for task, _ in taken}
        self.task_list = [task for task in self.task_list if id(task) not in gone]
        with self._lock:
            for task, status in taken:
                weight = self._weight(task)
                self._total -= weight
                if self._finished.pop(id(task), None) is not None:
                    self._sums['final'] -= weight
                    if status == 'exec_done':
                        self._sums['exec_done'] -= weight
                    if status in FAILED_TASK_STATUSES:
                        self._sums['failed'] -= weight
                self._untracked.pop(id(task), None)
                self._may_run.pop(id(task), None)
        return taken

    def get_completed_number(self):
        """Get the number of completed tasks, or of their structures in a batched queue (see ``_weight``).
        """
        return int(self._count('final', is_final_task))

    def get_running_number(self):
        """Get the number of running tasks.
        """
        # Parsl never marks its futures running, so only other futures are checked.
        return int(sum(self._weight(task) for task in list(self._may_run.values()) if task.running()))

    def get_number(self):
        """Get the the number of tasks in the task list.
        """
        return len(self.task_list)

    def get_queued_number(self):
        """Get the number of queued tasks.
        """
        return int(self.get_number() - self.get_running_number() - self.get_completed_number())

    def get_exec_done_number(self):
        return int(self._count('exec_done', lambda task: task.task_status() == 'exec_done' and task.done()))

    def get_failed_number(self):
        return int(self._count('failed', lambda task: task.task_status() in FAILED_TASK_STATUSES))

    def get_task_results(self, max_items=None):
        """Get the reults of the tasks

        Returns:
            (tuple): A tuple containing, respectively, a list that stores the task results and an int that tells
                     the number of failed tasks.
        """
        results_list = []
        failed_number = 0
        self.last_failed = []
        for task, task_status in self._take_finished(reverse=True, max_items=max_items):
            if task_status in SUCCESSFUL_TASK_STATUSES:
                try:
                    results_list.append(task.result())
                except Exception:
                    logger.error("Task marked as done but raised", exc_info=True)
                    failed_number += 1
                    self.last_failed.append(task)
            else: # Handle all others as failure states
                try:
                    task.result()
                    logger.warning("Task marked as %s returned a result; discarding it", task_status)
                except Exception:
                    logger.warning("Task failed", exc_info=True)
                failed_number += 1
                self.last_failed.append(task)

        return results_list, failed_number

    def get_task_status(self):
        """Get the status of the tasks.

            Returns:
                (list): List containing the status of the tasks.
        """
        status_list = []
        for task in self.task_list:
            status_list.append(task.task_status())

        return status_list

    def print_status(self):
        """Prints the status of the tasks.
        """
        print('Total Tasks: {:d}'.format(self.get_number()))
        # print('Queued Tasks: {:d}'.format(self.get_queued_number()))
        # print('Running Tasks: {:d}'.format(self.get_running_number()))
        print('Finished Tasks: {:d}'.format(self.get_completed_number()))
        print('Finished Successfully Tasks: {:d}'.format(self.get_exec_done_number()))
        print('Finished Failed Tasks: {:d}'.format(self.get_failed_number()))



# The class's earlier name, which scripts and subclasses may still use.
parsl_task_queue = ParslTaskQueue


def cancel_blocks(dfk):
    """Cancel every block Parsl's executors have submitted, through each executor's provider.

    This needs only the scheduler (``scancel`` for Slurm), not Parsl's interchange. Blocks that have
    already ended are cancelled again harmlessly.

    Args:
        dfk (parsl.dataflow.dflow.DataFlowKernel): The object returned by ``parsl.load``.

    """
    for label, executor in getattr(dfk, 'executors', {}).items():
        provider = getattr(executor, 'provider', None)
        job_ids = [job for job in getattr(executor, 'blocks_to_job_id', {}).values() if job is not None]
        if provider is None or not job_ids:
            continue
        try:
            provider.cancel(job_ids)
        except Exception as exc:  # Keep going: the other executors' blocks still need cancelling.
            print("Could not cancel {}'s blocks {}: {}: {}".format(label, job_ids, type(exc).__name__, exc),
                  flush=True)


def release_allocations_at_exit(dfk, cleanup_timeout=CLEANUP_TIMEOUT):
    """Stop Parsl's interchange and cancel its blocks however the coordinator ends.

    Parsl's own exit hook only logs a warning. Without ``cleanup``, the interchange outlives the
    coordinator and keeps answering worker heartbeats, so the scheduler holds the nodes until walltime.
    This covers ``exit()``, uncaught errors, Ctrl-C, and ``SIGTERM`` from ``scancel``, but not ``SIGKILL``.

    The blocks are cancelled through the providers first, because Parsl's cleanup can hang: it asks each
    interchange which workers are connected and waits for the answer without a time limit. Ctrl-C in a
    terminal reaches the interchanges too (they share the coordinator's process group), so by then they
    may be gone. The cleanup then runs for at most ``cleanup_timeout`` seconds.

    Args:
        dfk (parsl.dataflow.dflow.DataFlowKernel): The object returned by ``parsl.load``.
        cleanup_timeout (float): Seconds to wait for Parsl's cleanup (default ``ALF_PARSL_CLEANUP_TIMEOUT``,
            else 120).

    """
    def cleanup():
        try:
            dfk.cleanup()
        except Exception as exc:  # Don't hide the error that ended the run.
            print("Could not release Parsl allocations: {}: {}".format(type(exc).__name__, exc), flush=True)

    def release():
        if dfk.cleanup_called:
            return
        cancel_blocks(dfk)
        worker = threading.Thread(target=cleanup, name='alf-parsl-cleanup', daemon=True)
        worker.start()
        worker.join(cleanup_timeout)
        if worker.is_alive():
            print("Parsl's cleanup did not finish within {:g} s, probably because its interchange had already "
                  "stopped; ALF cancelled the blocks itself.".format(cleanup_timeout), flush=True)

    atexit.register(release)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def stop_if_executor_failed(dfk):
    """Raise if Parsl has permanently stopped an executor that ALF needs.

    Every task on such an executor fails from then on, so the run can't make progress, though nothing
    else would stop it. Standby executors are skipped, because their primary can still do the work.

    Args:
        dfk (parsl.dataflow.dflow.DataFlowKernel): The running Parsl workflow.

    """
    for label, executor in dfk.executors.items():
        if 'standby' not in label and getattr(executor, 'bad_state_is_set', False):
            raise RuntimeError('Parsl permanently stopped {}, so ALF is stopping. Fix the cause and restart. '
                               'Parsl reported: {}'.format(label, executor.executor_exception))
