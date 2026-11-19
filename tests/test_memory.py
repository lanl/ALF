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
