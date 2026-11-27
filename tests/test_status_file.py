"""The opt-in write_status_on_change: the status file is rewritten only when it changes, and holds the same status."""
import json

import pytest

from alframework.tools import restarts
from alframework.tools.restarts import StatusFile, status_on_change_setting


@pytest.fixture
def writes(monkeypatch):
    calls = []
    real = restarts.write_status
    monkeypatch.setattr(restarts, 'write_status', lambda path, status: (calls.append(path), real(path, status)))
    return calls


def test_by_default_every_call_writes(tmp_path, writes):
    status_file, path = StatusFile(), str(tmp_path / 'status.txt')
    for _ in range(3):
        assert status_file.write(path, {'current_h5_id': 0})
    assert len(writes) == 3 and status_file.skipped == 0


def test_on_change_skips_only_an_unchanged_status(tmp_path, writes):
    status_file, path = StatusFile(only_on_change=True), str(tmp_path / 'status.txt')
    status = {'current_h5_id': 0, 'lifetime_stored_labels': 0}
    assert status_file.write(path, status)
    assert not status_file.write(path, dict(status)), 'the same status is not written again'
    status['current_h5_id'] = 1
    assert status_file.write(path, status), 'a change is written in the call that passes it'
    assert json.loads((tmp_path / 'status.txt').read_text()) == status
    assert status_file.write(str(tmp_path / 'other.txt'), status), 'another path is written'
    assert len(writes) == 3 and status_file.skipped == 1


def test_the_setting_is_off_unless_true():
    assert status_on_change_setting({}) is False and status_on_change_setting(None) is False
    assert status_on_change_setting({'write_status_on_change': True}) is True
    with pytest.raises(ValueError, match='write_status_on_change must be true or false'):
        status_on_change_setting({'write_status_on_change': 'yes'})


def test_the_coordinator_writes_fewer_status_files_and_ends_with_the_same_status(tmp_path, monkeypatch):
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    counts, statuses = {}, {}
    for arm, on in (('every_pass', False), ('on_change', True)):
        run_dir = tmp_path / arm
        run_dir.mkdir()
        calls = []
        real = restarts.write_status
        with monkeypatch.context() as patch:
            patch.setattr(restarts, 'write_status', lambda path, status: (calls.append(path), real(path, status)))
            run = run_alf_driver(run_dir, patch, **restart_tasks(run_dir, []), **LOOP, finite_bootstrap=True,
                                 write_status_on_change=on)
        counts[arm], statuses[arm] = len(calls), run.status
        assert run.status['lifetime_stored_labels'] > 0
        assert json.loads((run_dir / 'status.json').read_text()) == run.status, 'the file on disk is current'
    assert counts['on_change'] < counts['every_pass'], counts
    assert statuses['on_change']['current_h5_id'] == statuses['every_pass']['current_h5_id']
