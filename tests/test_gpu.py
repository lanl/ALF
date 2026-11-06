"""Allocation-relative device selection without process-wide mask mutations."""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest
from ase import Atoms

from alframework.tools.gpu import VISIBILITY_VARIABLES, neurochem_device, sampling_device
from alframework.tools.molecules_class import MoleculesObject
from alframework.samplers import batched_mlmd


@pytest.fixture(autouse=True)
def clean_worker(monkeypatch):
    for key in (*VISIBILITY_VARIABLES, 'PARSL_WORKER_RANK'):
        monkeypatch.delenv(key, raising=False)


def runtime(monkeypatch, count, hip=None):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: count > 0, device_count=lambda: count),
        version=SimpleNamespace(hip=hip)))


@pytest.mark.parametrize('masks,count,rank,cap,expected', [
    ({}, 4, '5', 4, 'cuda:1'),
    ({}, 8, '5', 2, 'cuda:1'),
    ({}, 2, '5', 8, 'cuda:1'),
    ({}, 2, None, 0, 'cuda:0'),
    ({}, 3, '5', None, 'cuda:2'),
    ({'CUDA_VISIBLE_DEVICES': '2,5'}, 2, '3', 1, 'cuda:1'),
    ({'CUDA_VISIBLE_DEVICES': '5,2'}, 2, '2', 4, 'cuda:0'),
    ({'CUDA_VISIBLE_DEVICES': 'GPU-one,GPU-two'}, 2, '5', 4, 'cuda:1'),
    ({'CUDA_VISIBLE_DEVICES': 'MIG-one'}, 1, '5', 4, 'cuda:0'),
    ({'ROCR_VISIBLE_DEVICES': '2,5'}, 2, '5', 4, 'cuda:1'),
    ({'HIP_VISIBLE_DEVICES': '1'}, 1, '5', 4, 'cuda:0'),
    ({'ROCR_VISIBLE_DEVICES': '2,5', 'HIP_VISIBLE_DEVICES': '1'}, 1, '5', 4, 'cuda:0'),
    ({'ROCR_VISIBLE_DEVICES': 'GPU-one,GPU-two', 'CUDA_VISIBLE_DEVICES': '1'},
     1, '5', 4, 'cuda:0'),
])
def test_automatic_assignment_preserves_visibility(monkeypatch, masks, count, rank, cap, expected):
    runtime(monkeypatch, count)
    for key, value in masks.items():
        monkeypatch.setenv(key, value)
    if rank is not None:
        monkeypatch.setenv('PARSL_WORKER_RANK', rank)
    before = dict(os.environ)
    for _ in range(3):  # A reused worker must not progressively narrow/reindex its mask.
        assert sampling_device({}, cap) == expected
        assert sampling_device({'device': 'cuda'}, cap) == expected
    assert dict(os.environ) == before


def test_amd_runtime_follows_the_same_assignment_rules(monkeypatch):
    runtime(monkeypatch, 2, hip='6.4')
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', '2,5')
    monkeypatch.setenv('PARSL_WORKER_RANK', '5')
    assert sampling_device({}, 4) == 'cuda:1'
    assert os.environ['ROCR_VISIBLE_DEVICES'] == '2,5'


@pytest.mark.parametrize('mask', ['', '-1', 'NoDevFiles'])
def test_no_visible_gpu_does_not_expand_allocation(monkeypatch, mask):
    runtime(monkeypatch, 0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setenv('PARSL_WORKER_RANK', '5')
    assert sampling_device({}, 4) == 'cpu'
    with pytest.raises(RuntimeError, match='No GPU'):
        sampling_device({'device': 'cuda'}, 4)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == mask


def test_cpu_and_missing_optional_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setenv('PARSL_WORKER_RANK', 'not-an-integer')
    before = dict(os.environ)
    assert sampling_device({'device': 'cpu'}, 0) == 'cpu'
    assert sampling_device({}) == 'cpu'
    with pytest.raises(RuntimeError, match='No GPU'):
        sampling_device({'device': 'cuda'}, 4)
    assert dict(os.environ) == before


def test_explicit_local_device_overrides_rank_and_cap(monkeypatch):
    runtime(monkeypatch, 2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,2')
    monkeypatch.setenv('PARSL_WORKER_RANK', '0')
    assert sampling_device({'device': 'cuda:1'}, 1) == 'cuda:1'
    for value in ['cuda:2', 'cuda:-1', 'cuda:bad']:
        with pytest.raises(ValueError, match='CUDA device'):
            sampling_device({'device': value}, 4)


@pytest.mark.parametrize('rank,cap', [('bad', 4), ('-1', 4), ('0', -1), ('0', 1.5)])
def test_invalid_unmasked_assignment(monkeypatch, rank, cap):
    runtime(monkeypatch, 2)
    monkeypatch.setenv('PARSL_WORKER_RANK', rank)
    with pytest.raises(ValueError):
        sampling_device({}, cap)


@pytest.mark.parametrize('mask,expected', [(None, 'cuda:1'), ('2,5', 'cuda:1'),
    ('GPU-a,GPU-b', 'cuda:1'), ('MIG-a', 'cuda:0'), ('2,-1,5', 'cuda:0')])
def test_neurochem_does_not_require_torch(monkeypatch, mask, expected):
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setenv('PARSL_WORKER_RANK', '5')
    if mask is not None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    before = dict(os.environ)
    assert neurochem_device({}, 4) == expected
    assert dict(os.environ) == before


@pytest.mark.parametrize('mask', ['', '-1', 'NoDevFiles'])
def test_neurochem_disabled_gpu(monkeypatch, mask):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    with pytest.raises(RuntimeError, match='No GPU'):
        neurochem_device({}, 4)


def test_neurochem_explicit_device_and_invalid_settings(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,2')
    assert neurochem_device({'device': 'cuda:1'}, 1) == 'cuda:1'
    with pytest.raises(ValueError, match='requires a CUDA'):
        neurochem_device({'device': 'cpu'}, 4)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError, match='positive gpus_per_node'):
        neurochem_device({}, 0)


@pytest.mark.parametrize('neurochem', [False, True])
def test_legacy_task_forwards_selected_local_device(monkeypatch, neurochem):
    module = importlib.import_module('alframework.samplers.mlmd_sampling')
    runtime(monkeypatch, 2)
    monkeypatch.setenv('PARSL_WORKER_RANK', '5')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5')
    captured = {}

    def loader(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return []

    monkeypatch.setattr(module, 'load_module_from_config', lambda *args: loader)
    monkeypatch.setattr(module, 'MLMD_calculator', lambda *args, **kwargs: object())
    monkeypatch.setattr(module, 'mlmd_sampling', lambda molecule_object: molecule_object)
    config = dict(amp_temp=[0, 0], per_temp=[1, 1], srt_temp=[0, 0], end_temp=[0, 0],
                  amp_dens=None, per_dens=None, end_dens=None, MLMD_calculator_options={})
    if neurochem:
        config['use_potential_specific_code'] = 'neurochem'
    molecule = MoleculesObject(Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]]), 'device')
    before = {key: os.environ.get(key) for key in VISIBILITY_VARIABLES}
    result = module.simple_mlmd_sampling_task.func(molecule, config, '/models/{:04d}', 3, 4)
    if hasattr(result, 'reraise'):
        result.reraise()
    assert result is molecule
    if neurochem:
        assert captured['args'][0]['gpu'] == '1'
    else:
        assert captured['kwargs']['device'] == 'cuda:1'
    assert {key: os.environ.get(key) for key in VISIBILITY_VARIABLES} == before


def test_batched_task_forwards_selected_device(monkeypatch):
    runtime(monkeypatch, 2)
    monkeypatch.setenv('PARSL_WORKER_RANK', '5')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5')
    captured = {}

    def loader(path, device):
        captured['loader'] = device
        return object()

    def sample(molecules, models, config, device):
        captured['engine'] = device
        return molecules

    monkeypatch.setattr(batched_mlmd, 'load_module_from_config', lambda *args: loader)
    monkeypatch.setattr(batched_mlmd, 'sample_trajectories', sample)
    result = batched_mlmd.mlmd_trajectories_task.func([], {'batched_calculator': 'test'}, '/models/{}', 3, 4)
    if hasattr(result, 'reraise'):
        result.reraise()
    assert result == []
    assert captured == {'loader': 'cuda:1', 'engine': 'cuda:1'}
