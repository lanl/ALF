"""Acceptance artifacts identify executed code, including detached source copies."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/torchsim_benchmark/benchmark.py'


def benchmark_module():
    spec = importlib.util.spec_from_file_location('alf_engine_benchmark', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imported_checkout_is_fingerprinted():
    benchmark = benchmark_module()
    source = benchmark.source_identity()
    expected = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert source['revision'] == expected
    assert source['files_sha256']['alframework/samplers/batched_mlmd.py'] == \
        benchmark.fingerprint(ROOT / 'alframework/samplers/batched_mlmd.py')


def test_source_copy_does_not_claim_enclosing_repository_revision(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    copy = tmp_path / 'artifacts/source'
    script = copy / 'examples/torchsim_benchmark/benchmark.py'
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    package = copy / 'alframework/__init__.py'
    package.parent.mkdir()
    package.write_text('# isolated source copy\n')
    benchmark = benchmark_module()
    monkeypatch.setattr(benchmark, 'HERE', script.parent)
    monkeypatch.setattr(benchmark, '__file__', str(script))
    monkeypatch.setattr(benchmark.alframework, '__file__', str(package))
    before = benchmark.source_identity()
    assert before['revision'] is None
    package.write_text('# a different copied implementation\n')
    assert benchmark.source_identity()['tree_sha256'] != before['tree_sha256']


def test_benchmark_import_does_not_require_either_engine():
    script = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch_sim','nvalchemi','nvalchemiops','warp'}:
   raise ImportError('Optional engine blocked: '+fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(SCRIPT)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
