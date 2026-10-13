from alframework.tools.tools import is_final_task, parsl_task_queue
from tests.helpers.fakes import FakeTask


def test_parsl_task_queue_counts_statuses():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("exec_done", done=True))
    queue.add_task(FakeTask("exec_done", done=False))
    queue.add_task(FakeTask("running", done=False, running=True))
    queue.add_task(FakeTask("failed", done=True))

    assert queue.get_number() == 4
    assert queue.get_completed_number() == 3
    assert queue.get_running_number() == 1
    assert queue.get_exec_done_number() == 1
    assert queue.get_failed_number() == 1
    assert queue.get_queued_number() == 0
    assert queue.get_task_status() == ["exec_done", "exec_done", "running", "failed"]

    for status in ("exec_done", "memo_done", "failed", "dep_fail"):
        assert is_final_task(FakeTask(status, done=False))
    for status in ("pending", "launched", "running", "joining", "running_ended", "fail_retryable", "unknown"):
        assert not is_final_task(FakeTask(status, done=False))


def test_parsl_task_queue_collects_results_and_removes_finished_tasks():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("running", result="keep", done=False, running=True))
    queue.add_task(FakeTask("exec_done", result="done", done=True))
    queue.add_task(FakeTask("memo_done", result="memoized", done=True))
    queue.add_task(FakeTask("failed", exception=RuntimeError("task failed")))
    queue.add_task(FakeTask("dep_fail", result="drop", done=True))
    intermediate_statuses = ["pending", "launched", "joining", "running_ended", "fail_retryable"]
    for status in intermediate_statuses:
        queue.add_task(FakeTask(status, result="keep", done=False))

    assert queue.get_failed_number() == 2

    results, failed = queue.get_task_results()

    assert sorted(results) == ["done", "memoized"]
    assert failed == 2
    assert queue.get_number() == 1 + len(intermediate_statuses)
    assert queue.get_task_status() == ["running"] + intermediate_statuses
