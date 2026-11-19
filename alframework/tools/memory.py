"""Process memory settings for large training jobs."""
import ctypes
import os
import sys

PR_SET_THP_DISABLE = 41
THP_VARIABLE = 'ALF_THP_DISABLE'


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
