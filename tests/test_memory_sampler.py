"""The opt-in coordinator memory sampler: one CSV row per process in a tree, until the root exits."""

import csv
import os
import subprocess
import sys
import time

import pytest

from alframework.tools import memory_sampler

pytestmark = pytest.mark.skipif(not os.path.isdir('/proc'), reason='the sampler reads /proc')


def _rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def test_it_is_off_unless_the_master_config_sets_an_interval(tmp_path):
    for config in ({}, {'coordinator_memory_sample': 0}, {'coordinator_memory_sample': None}):
        assert memory_sampler.start(dict(config, coordinator_memory_path=str(tmp_path / 'm.csv'))) is None
    assert not (tmp_path / 'm.csv').exists()


def test_it_samples_a_process_and_its_children_until_the_process_exits(tmp_path):
    # A parent that starts one child, then both live for about two seconds.
    parent = subprocess.Popen([sys.executable, '-c',
                               'import subprocess, sys, time; '
                               'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"]); '
                               'time.sleep(2); child.wait()'])
    path = tmp_path / 'memory' / 'coordinator_memory.csv'
    sampler = memory_sampler.start({'coordinator_memory_sample': 0.2, 'coordinator_memory_path': str(path)},
                                   pid=parent.pid)
    parent.wait(timeout=30)
    sampler.wait(timeout=30)  # It stops by itself once the sampled process has gone.
    rows = _rows(path)
    assert list(rows[0]) == list(memory_sampler.COLUMNS)
    pids = {int(row['pid']) for row in rows}
    assert parent.pid in pids and len(pids) >= 2  # the parent and its child
    assert all(int(row['rss_kib']) > 0 for row in rows if row['rss_kib'])
    assert len({row['time'] for row in rows}) >= 2  # several samples


def test_a_restarted_coordinator_appends_without_a_second_header(tmp_path):
    path = tmp_path / 'coordinator_memory.csv'
    for _ in range(2):
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.5)'])
        memory_sampler.start({'coordinator_memory_sample': 0.1, 'coordinator_memory_path': str(path)},
                             pid=sleeper.pid).wait(timeout=30)
        sleeper.wait(timeout=30)
    lines = path.read_text().splitlines()
    assert sum(line.startswith('time,') for line in lines) == 1 and len(lines) > 2
