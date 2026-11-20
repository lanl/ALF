"""Process memory settings for large training jobs."""
import ctypes
import gc
import os
import sys
import time

PR_SET_THP_DISABLE = 41
THP_VARIABLE = 'ALF_THP_DISABLE'


def snapshot_memory_trim(master_config):
    """Return a coordinator callback for releasing free glibc heap pages at snapshot boundaries."""
    enabled = master_config.get('coordinator_trim_after_snapshot', False)
    if not isinstance(enabled, bool):
        raise ValueError('coordinator_trim_after_snapshot must be true or false.')
    if not enabled:
        return None
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None) if sys.platform.startswith('linux') else None
    if trim is None:
        print('Coordinator allocator trim unavailable on this host.', flush=True)
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int

    from alframework.tools.memory_sampler import sample

    def release():
        before = sample(os.getpid())
        started = time.perf_counter()
        gc.collect()
        trim(0)
        elapsed = time.perf_counter() - started
        after = sample(os.getpid())
        change = ''
        if before and after and before[-1] is not None and after[-1] is not None:
            change = f' (PSS {before[-1] / 2**20:.2f} to {after[-1] / 2**20:.2f} GiB)'
        print(f'Coordinator allocator trim after label snapshot took {elapsed:.3f} s{change}.', flush=True)

    return release


def disable_transparent_huge_pages_if_requested(environ=None):
    """With ALF_THP_DISABLE=1, turn off transparent huge pages for this process and every process it starts.

    Where the system setting is "always", first writes to large new arrays can stall in the kernel's huge-page
    compaction: on one shared login node, copying into fresh arrays ran at about 60 MiB/s with huge pages and
    1.6 GiB/s without. Preparing a member's data copies its full dataset several times. The setting is kept across exec
    and inherited by child processes (Linux prctl PR_SET_THP_DISABLE). Returns whether it was applied.
    """
    environ = os.environ if environ is None else environ
    if environ.get(THP_VARIABLE, '') in ('', '0') or not sys.platform.startswith('linux'):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        print(f'{THP_VARIABLE}: prctl failed (errno {ctypes.get_errno()}); keeping the system setting', flush=True)
        return False
    return True
