"""Sample the memory of a process and all of its descendants, one CSV row per process per sample.

The coordinator starts it when the master config sets ``coordinator_memory_sample`` (seconds between samples),
writing to ``coordinator_memory_path`` (default ``coordinator_memory.csv``). It can also be run by hand::

  python -m alframework.tools.memory_sampler PID OUT.csv INTERVAL_SECONDS

Columns: ``time`` (Unix seconds), ``pid``, ``ppid``, ``command`` (shortened), ``rss_kib`` and ``pss_kib``. PSS
divides shared pages among the processes sharing them, so summing PSS over the tree does not count a forked
child's inherited pages twice, as summing RSS does. The sampler stops when PID exits. It reads ``/proc``, so it
records nothing on systems without one.
"""
import csv
import os
import subprocess
import sys
import time

COLUMNS = ('time', 'pid', 'ppid', 'command', 'rss_kib', 'pss_kib')


def _parent(pid):
    with open(f'/proc/{pid}/stat') as handle:
        return int(handle.read().rsplit(')', 1)[1].split()[1])


def _alive(pid):
    """Whether ``pid`` is running: it exists and isn't a zombie (exited, waiting for its parent to reap it)."""
    try:
        with open(f'/proc/{pid}/stat') as handle:
            return handle.read().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
    except (OSError, IndexError):
        return False


def process_tree(root):
    """``root`` and every process descended from it, as PIDs."""
    children = {}
    for name in os.listdir('/proc'):
        if name.isdigit():
            try:
                children.setdefault(_parent(name), []).append(int(name))
            except (OSError, IndexError, ValueError):
                continue  # The process ended while the table was read.
    found, pending = [], [root]
    while pending:
        pid = pending.pop()
        found.append(pid)
        pending.extend(children.get(pid, []))
    return found


def sample(pid):
    """``[pid, ppid, command, rss_kib, pss_kib]`` for one process, or ``None`` if it can't be read."""
    try:
        parent = _parent(pid)
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            command = ' '.join(part.decode(errors='replace') for part in handle.read().split(b'\0') if part)[:120]
        rss = pss = None
        with open(f'/proc/{pid}/smaps_rollup') as handle:
            for line in handle:
                if line.startswith('Rss:'):
                    rss = int(line.split()[1])
                elif line.startswith('Pss:'):
                    pss = int(line.split()[1])
        return [pid, parent, command, rss, pss]
    except (OSError, IndexError, ValueError):
        return None


def run(root, path, interval):
    """Append one row per process in ``root``'s tree every ``interval`` seconds until ``root`` exits."""
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(COLUMNS)
        while _alive(root):
            now = round(time.time(), 1)
            for pid in process_tree(root):
                values = sample(pid)
                if values is not None:
                    writer.writerow([now, *values])
            handle.flush()
            time.sleep(interval)


def start(master_config, pid=None):
    """Start sampling ``pid`` (default: this process) in a separate process, if the master config asks for it.

    Returns the sampler's ``Popen``, or ``None`` when ``coordinator_memory_sample`` is unset, zero, or there is no
    ``/proc``. The sampler ends by itself when the sampled process exits.
    """
    interval = master_config.get('coordinator_memory_sample')
    if not interval or float(interval) <= 0 or not os.path.isdir('/proc'):
        return None
    path = master_config.get('coordinator_memory_path', 'coordinator_memory.csv')
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Its own session, so a Ctrl-C to the coordinator's terminal doesn't end it before the coordinator.
    return subprocess.Popen([sys.executable, '-m', __name__, str(pid or os.getpid()), path, str(float(interval))],
                            start_new_session=True)


if __name__ == '__main__':
    run(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]))
