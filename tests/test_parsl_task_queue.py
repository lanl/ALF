from alframework.tools.tools import parsl_task_queue


class FakeTask:
    def __init__(self, status, result=None, done=True, running=False):
        self._status = status
        self._result = result
        self._done = done
        self._running = running

    def done(self):
        return self._done

    def running(self):
        return self._running

    def task_status(self):
        return self._status

    def result(self):
        return self._result


def test_parsl_task_queue_counts_statuses():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("exec_done", done=True))
    queue.add_task(FakeTask("exec_done", done=False))
    queue.add_task(FakeTask("running", done=False, running=True))
    queue.add_task(FakeTask("failed", done=True))

    assert queue.get_number() == 4
    assert queue.get_completed_number() == 2
    assert queue.get_running_number() == 1
    assert queue.get_exec_done_number() == 1
    assert queue.get_failed_number() == 1
    assert queue.get_queued_number() == 1
    assert queue.get_task_status() == ["exec_done", "exec_done", "running", "failed"]


def test_parsl_task_queue_collects_results_and_removes_finished_tasks():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("running", result="keep", done=False, running=True))
    queue.add_task(FakeTask("exec_done", result="done", done=True))
    queue.add_task(FakeTask("failed", result="drop", done=True))

    results, failed = queue.get_task_results()

    assert results == ["done"]
    assert failed == 1
    assert queue.get_number() == 1
    assert queue.get_task_status() == ["running"]
