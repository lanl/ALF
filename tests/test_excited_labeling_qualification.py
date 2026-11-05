import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from examples.excited_state_labeling import qualify


def test_preparation_exports_only_committed_source_without_submission(tmp_path):
    root = tmp_path/'run'
    site = {b+'_python': '/absolute/'+b+'/python' for b in qualify.BACKENDS}
    real = qualify.subprocess.check_output
    def checked(args, **kw):
        assert args[0] == 'git'
        return real(args, **kw)
    with patch.object(qualify.subprocess, 'check_output', checked):
        manifest = qualify.prepare(root, site, ['pyseqm'])
    assert len(manifest['fixtures']) == 5
    assert (root/'pyseqm.sbatch').exists()
    assert not (root/'gpu4pyscf.sbatch').exists()
    assert '--time=00:30:00' in (root/'pyseqm.sbatch').read_text()
    assert json.loads((root/'manifest.json').read_text())['revision'] == manifest['revision']
    with pytest.raises(FileExistsError):
        qualify.prepare(root, site, ['pyseqm'])


def test_cleanup_uses_only_numeric_owned_job_and_verifies_exit():
    calls = []
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(stdout='')
    assert qualify.cleanup_job('123', run=run)['verified_terminated']
    assert calls == [['scancel','123'], ['squeue','-h','-j','123','-o','%i']]
    with pytest.raises(ValueError): qualify.cleanup_job('--user=pjlohr', run=run)
    def active(args, **kw): return SimpleNamespace(stdout='123')
    with pytest.raises(RuntimeError): qualify.cleanup_job('123', run=active, deadline_seconds=0)


def test_deadline_does_not_submit_jobs(tmp_path):
    qualify.write_json(tmp_path/'manifest.json', {'backends':['gpu4pyscf','pyseqm']})
    with patch.object(qualify.subprocess, 'check_output', side_effect=AssertionError('must not submit')):
        qualify.execute(tmp_path, budget_seconds=0)
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['status'] == 'incomplete'
    assert result['owned_jobs'] == []
    assert len(result['stages']) == 2


def test_fixture_identity_is_deterministic_and_tagged():
    a, b = qualify.fixture_specs(), qualify.fixture_specs()
    assert a == b
    ms = qualify.molecules(a)
    assert len(ms) == 5 and len(ms[-1]) == 6
    assert all('topology_atom_ids' in m.get_atoms().arrays for m in ms)
