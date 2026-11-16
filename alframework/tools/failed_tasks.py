"""Free what a failed task still holds before its worker takes the next task.

An exception keeps every frame it passed through, with their local variables, and those frames often refer
back to it, so the objects of a failed task, such as an MD engine and the tensors of the evaluation that
failed, stay allocated until Python's cycle collector happens to run. A Parsl worker is long-lived: if a task
fails because its GPU ran out of memory, the GPU is still full when the worker takes its next task, which
then fails within milliseconds. A worker that fails that quickly is free again first, so it takes most of
the tasks that follow, including those resubmitted to replace the lost ones.
"""
import functools


def release_failed_task(error):
    """Clear the frames ``error`` (and the exceptions it chains to) holds, collect, and empty the GPU cache.

    The frames keep their code and line numbers, so the traceback still prints; only their local variables
    are dropped. Frames still running, such as the caller's, are left alone.
    """
    import gc
    import sys
    import traceback
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))
    gc.collect()
    torch = sys.modules.get('torch')  # Never import torch here: a CPU task may not use it.
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass  # Releasing memory must never replace the task's own error.


def releases_memory_on_failure(function):
    """Decorate a task function (under ``@python_app`` and ``timed``) so a failure frees its memory first."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            release_failed_task(error)
            raise
    return wrapper
