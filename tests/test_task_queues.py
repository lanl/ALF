"""Task queues count from each task's final state, recorded once when it finishes.

The coordinator counts its queues several times every poll. With thousands of tasks, checking every
task each time made a poll cost grow with the number of tasks; Parsl's futures report when they finish,
so the counts are kept as tasks finish instead. These tests check the counts and the harvest against
what checking every task gives.
"""

from concurrent.futures import Future

from alframework.tools.batching import BatchedTaskQueue
from alframework.tools.parsl_tasks import parsl_task_queue
from tests.helpers.fakes import FakeTask


class StatusFuture(Future):
    """A future with Parsl's task_status, set to a final state before the result, as Parsl does."""

    def __init__(self):
        super().__init__()
        self.status = "launched"
        self.status_reads = 0

    def task_status(self):
        self.status_reads += 1
        return self.status

    def finish(self, value):
        self.status = "exec_done"
        self.set_result(value)

    def fail(self):
        self.status = "failed"
        self.set_exception(RuntimeError("lost"))


def test_counts_follow_tasks_as_they_finish_without_checking_them():
    queue = parsl_task_queue()
    tasks = [StatusFuture() for _ in range(5)]
    for task in tasks:
        queue.add_task(task)
    assert (queue.get_number(), queue.get_completed_number(), queue.get_queued_number()) == (5, 0, 5)
    tasks[1].finish("b")
    tasks[3].fail()
    tasks[4].finish("e")
    reads = sum(task.status_reads for task in tasks)
    assert queue.get_completed_number() == 3 and queue.get_exec_done_number() == 2 and queue.get_failed_number() == 1
    assert queue.get_queued_number() == 2 and queue.get_running_number() == 0
    assert sum(task.status_reads for task in tasks) == reads, "counting doesn't ask each task again"


def test_the_harvest_matches_checking_every_task():
    """Same results, in the same order (newest first), and the same failure count as before."""
    queue = parsl_task_queue()
    tasks = [StatusFuture() for _ in range(4)]
    for task in tasks:
        queue.add_task(task)
    tasks[0].finish("a")
    tasks[2].finish("c")
    tasks[3].fail()
    results, failed = queue.get_task_results()
    assert results == ["c", "a"] and failed == 1 and queue.last_failed == [tasks[3]]
    assert queue.task_list == [tasks[1]] and queue.get_completed_number() == 0
    tasks[1].finish("b")
    assert queue.get_task_results() == (["b"], 0) and queue.get_number() == 0


def test_a_task_done_before_it_is_added_counts_at_once():
    queue = parsl_task_queue()
    task = StatusFuture()
    task.finish("x")
    queue.add_task(task)
    assert queue.get_completed_number() == 1 and queue.get_task_results() == (["x"], 0)


def test_tasks_without_callbacks_are_still_checked_one_by_one():
    queue = parsl_task_queue()
    tracked, plain_done, plain_running = StatusFuture(), FakeTask("exec_done", result="p"), \
        FakeTask("launched", done=False, running=True)
    for task in (tracked, plain_done, plain_running):
        queue.add_task(task)
    assert queue.get_completed_number() == 1 and queue.get_running_number() == 1 and queue.get_queued_number() == 1
    tracked.finish("t")
    assert queue.get_task_results() == (["p", "t"], 0) and queue.task_list == [plain_running], "newest first"


def test_a_batched_queue_counts_structures_and_keeps_failed_inputs():
    queue = BatchedTaskQueue()
    good, lost, waiting = StatusFuture(), StatusFuture(), StatusFuture()
    queue.add_task(good, width=3)
    queue.add_task(lost, width=2, inputs=["m1", "m2"])
    queue.add_task(waiting, width=4)
    assert (queue.get_number(), queue.get_task_number(), queue.get_completed_number()) == (9, 3, 0)
    good.finish([])
    lost.fail()
    assert queue.get_completed_number() == 5 and queue.get_exec_done_number() == 3 and queue.get_failed_number() == 2
    assert queue.get_task_results() == ([], 1)
    assert queue.failed_structures == 2 and queue.failed_inputs == ["m1", "m2"]
    assert (queue.get_number(), queue.get_task_number(), queue.get_completed_number()) == (4, 1, 0)
