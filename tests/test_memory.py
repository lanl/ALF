import sys
from pathlib import Path

import pytest

from alframework.tools import memory


def _thp_disabled():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('THP_enabled:'):
            return line.split()[1] == '0'
    pytest.skip('this kernel does not report THP_enabled')


def test_nothing_changes_unless_requested():
    assert memory.disable_transparent_huge_pages_if_requested({}) is False
    assert memory.disable_transparent_huge_pages_if_requested({memory.THP_VARIABLE: '0'}) is False


class FakeLibc:
    def __init__(self):
        self.calls = []

    def prctl(self, *arguments):
        self.calls.append(arguments)
        return 0


@pytest.fixture
def libc(monkeypatch):
    fake = FakeLibc()
    monkeypatch.setattr(memory.ctypes, 'CDLL', lambda *args, **kwargs: fake)
    monkeypatch.setattr(memory.sys, 'platform', 'linux')
    monkeypatch.setattr(memory, '_DISABLED_IN', None)
    return fake


def test_every_timed_task_applies_the_workers_setting_once_per_process(libc, monkeypatch):
    from alframework.tools.task_timeline import timed

    @timed('QM')
    def task(value):
        return value * 2

    monkeypatch.delenv(memory.THP_VARIABLE, raising=False)
    assert task(1) == 2 and libc.calls == []
    monkeypatch.setenv(memory.THP_VARIABLE, '1')
    assert task(2) == 4 and task(3) == 6
    assert libc.calls == [(memory.PR_SET_THP_DISABLE, 1, 0, 0, 0)]


def test_the_coordinator_turns_huge_pages_off_only_when_configured(libc, capsys):
    assert memory.coordinator_huge_pages({}) is False and libc.calls == []
    assert memory.coordinator_huge_pages({'disable_transparent_huge_pages': True}) is True
    assert libc.calls == [(memory.PR_SET_THP_DISABLE, 1, 0, 0, 0)]
    assert 'Transparent huge pages are off for the coordinator' in capsys.readouterr().out
    with pytest.raises(ValueError, match='disable_transparent_huge_pages must be true or false'):
        memory.coordinator_huge_pages({'disable_transparent_huge_pages': 'yes'})


def test_a_refused_prctl_keeps_the_system_setting(monkeypatch, capsys):
    class Refusing(FakeLibc):
        def prctl(self, *arguments):
            return -1

    monkeypatch.setattr(memory.ctypes, 'CDLL', lambda *args, **kwargs: Refusing())
    monkeypatch.setattr(memory.sys, 'platform', 'linux')
    monkeypatch.setattr(memory, '_DISABLED_IN', None)
    assert memory.disable_transparent_huge_pages() is False
    assert 'prctl failed' in capsys.readouterr().out


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux prctl')
def test_requested_setting_reaches_a_child_process():
    import subprocess
    code = ('from alframework.tools import memory; import pathlib; '
            'assert memory.disable_transparent_huge_pages_if_requested({"ALF_THP_DISABLE": "1"}); '
            'import subprocess, sys; print(subprocess.run(["grep", "THP_enabled", "/proc/self/status"], '
            'capture_output=True, text=True).stdout.strip())')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    if not out:
        pytest.skip('this kernel does not report THP_enabled')
    assert out.split()[-1] == '0'
