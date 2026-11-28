import csv

import pytest

from alframework.tools.loop_timing import COLUMNS, PHASES, LoopTimer, coordinator_loop_timer


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def test_each_pass_writes_one_row_with_its_phases_counts_and_sleep(tmp_path):
    clock = Clock()
    timer = LoopTimer(str(tmp_path / 'loop' / 'passes.csv'), clock=clock, wall_clock=lambda: 1000.0)
    with timer.phase('hold'):
        clock.now += 2.0
    with timer.phase('qm_submit'):
        clock.now += 0.5
    with timer.phase('hold'):
        clock.now += 1.0
    timer.count(results_held=2, qm_queue=40)
    timer.count(results_held=1, held_labels=7)
    timer.end_pass('active', 1.0)
    with timer.phase('write_status'):
        clock.now += 0.25
    timer.end_pass('bootstrap', 0.5)
    timer.close()

    rows = _rows(tmp_path / 'loop' / 'passes.csv')
    assert tuple(rows[0]) == COLUMNS
    first, second = rows
    assert (first['pass'], first['loop'], float(first['start'])) == ('1', 'active', 1000.0)
    assert float(first['seconds']) == pytest.approx(3.5)
    assert float(first['hold']) == pytest.approx(3.0) and float(first['qm_submit']) == pytest.approx(0.5)
    assert all(float(first[phase]) == 0 for phase in PHASES if phase not in ('hold', 'qm_submit'))
    assert (first['results_held'], first['qm_queue'], first['held_labels'], first['sleep']) == ('3', '40', '7', '1.0')
    assert second['pass'] == '2' and second['loop'] == 'bootstrap' and second['results_held'] == '0'
    assert float(second['seconds']) == pytest.approx(0.25)


def test_a_restarted_coordinator_appends_without_a_second_header(tmp_path):
    path = str(tmp_path / 'passes.csv')
    for _ in range(2):
        timer = LoopTimer(path)
        timer.end_pass('active', 0)
        timer.close()
    rows = _rows(path)
    assert [row['pass'] for row in rows] == ['1', '1']


def test_without_a_path_nothing_is_timed_or_written(tmp_path):
    timer = coordinator_loop_timer({})
    assert not timer.enabled
    with timer.phase('not a phase'):  # never checked when off, so timing costs nothing
        pass
    timer.count(anything=1)
    timer.end_pass('active', 1)
    assert list(tmp_path.iterdir()) == []


def test_unknown_phases_counts_and_loops_are_refused(tmp_path):
    timer = LoopTimer(str(tmp_path / 'passes.csv'))
    with pytest.raises(ValueError, match='Unknown coordinator loop phase'):
        timer.phase('sampling')
    with pytest.raises(ValueError, match='Unknown coordinator loop count'):
        timer.count(labels=1)
    with pytest.raises(ValueError, match='Unknown coordinator loop'):
        timer.end_pass('training', 0)
    with pytest.raises(ValueError, match='coordinator_loop_path must name a file'):
        coordinator_loop_timer({'coordinator_loop_path': ''})


def test_the_coordinator_times_every_pass_of_both_loops(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.helpers.driver import LOOP, restart_tasks

    run = run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP, finite_bootstrap=True,
                         label_storage_batch_size=2, coordinator_loop_path='records/loop.csv')
    rows = _rows(tmp_path / 'records' / 'loop.csv')
    loops = [row['loop'] for row in rows]
    assert 'bootstrap' in loops and 'active' in loops
    assert loops == sorted(loops, key=('bootstrap', 'active').index), 'the bootstrap passes come first'
    assert len(rows) == len(run.sleeps), 'one row per pass, written before its sleep'
    assert [float(row['sleep']) for row in rows] == pytest.approx(run.sleeps)
    for row in rows:
        assert sum(float(row[phase]) for phase in PHASES) <= float(row['seconds']) + 1e-5
    assert sum(int(row['results_held']) for row in rows) >= run.status['lifetime_stored_labels'] > 0
    assert {int(row['ml_queue']) for row in rows if row['loop'] == 'active'} >= {1}, 'training ran during a pass'


def test_a_timing_file_that_cannot_be_written_stops_timing_but_not_the_run(tmp_path, capsys):
    blocker = tmp_path / 'file'
    blocker.write_text('')
    timer = LoopTimer(str(blocker / 'passes.csv'))
    timer.end_pass('active', 0)
    assert not timer.enabled
    assert 'Could not write the coordinator loop timing' in capsys.readouterr().out


def test_a_disk_that_fills_while_timing_stops_timing_but_not_the_run(capsys):
    """The file opens, but its writes fail, as on a full disk (/dev/full fails every write with ENOSPC). Timing
    stops with a message, later passes and closing the timer go on without raising."""
    import os
    if not os.path.exists('/dev/full'):
        pytest.skip('this system has no /dev/full')
    timer = LoopTimer('/dev/full')
    timer.end_pass('active', 0)
    assert not timer.enabled
    assert 'Could not write the coordinator loop timing' in capsys.readouterr().out
    timer.end_pass('active', 0)
    timer.close()
