"""GPU discovery preserves CUDA visibility without parsing binary UUIDs."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker_pool():
    path = Path(__file__).resolve().parents[1] / 'examples/CPP_excited_gpu4pyscf/qm_worker_pool.py'
    spec = importlib.util.spec_from_file_location('cpp_qm_worker_pool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('encode', [str, str.encode])
def test_discovery_preserves_allocated_subset_and_order(worker_pool, monkeypatch, encode):
    # These real-format UUIDs contain zero bytes that break the binary CuPy path.
    first = 'GPU-b6893978-8c00-83c9-76be-5cbaaef2e07f'
    second = 'GPU-17b8cb76-ca01-17d1-d396-a3aee39e57d3'
    hidden = 'GPU-2ce021f5-c613-722c-1070-244f631c17f6'
    visible_pci = ['0000:42:00.0', '0000:07:00.0']
    runtime = SimpleNamespace(getDeviceCount=lambda: 2,
                              deviceGetPCIBusId=lambda i: encode(visible_pci[i]))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))

    def inventory(command, **kwargs):
        assert command == ['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits']
        return f'{first}, 00000000:07:00.0\n{hidden}, 00000000:81:00.0\n{second}, 00000000:42:00.0\n'

    monkeypatch.setattr(worker_pool.subprocess, 'check_output', inventory)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(worker_pool.CUDA_UUID_QUERY, {})
    accelerators = worker_pool.visible_accelerators(query=output.getvalue)
    assert accelerators == [second, first]
    assert hidden not in accelerators
    assert worker_pool.worker_arguments(['--available-accelerators'], accelerators) == [
        '--available-accelerators', second, first]


def test_discovery_does_not_fall_back_to_unallocated_devices(worker_pool, monkeypatch):
    runtime = SimpleNamespace(getDeviceCount=lambda: 1, deviceGetPCIBusId=lambda i: '0000:42:00.0')
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    monkeypatch.setattr(worker_pool.subprocess, 'check_output',
                        lambda *args, **kwargs: 'GPU-other, 00000000:07:00.0\n')
    with pytest.raises(KeyError):
        exec(worker_pool.CUDA_UUID_QUERY, {})
