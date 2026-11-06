"""Load tracked example settings without submitting jobs or requiring QM tools."""
import ast
import importlib
import json
from pathlib import Path

import pytest

from alframework.samplers.acquisition import acquisition_settings
from alframework.tools.batching import batching_settings
from alframework.tools.labeling import labeling_batch_settings
from alframework.tools.sampler_coordinator import driver_settings
from tests.helpers.fakes import import_neurochem_with_fake_backend


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('IL', 'UO2', 'fine_tuning', 'molten_salt', 'reactive_sampling',
            'seeded_active_learning', 'simple_water', 'simple_water_multi_builder')
MASTERS = sorted(p for name in FAMILIES for p in (ROOT/'examples'/name).glob('master*.json'))
TASKS = ('builder_task', 'sampler_task', 'QM_task', 'ML_task')


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize('path', MASTERS, ids=lambda p: str(p.relative_to(ROOT/'examples')))
def test_example_configs_are_valid(path):
    master = read(path)
    configs = {role: read(path.parent/master[role+'_config_path'])
               for role in ('builder', 'sampler', 'QM', 'ML')}
    for config in configs.values():
        assert isinstance(config, dict)
    sampler = configs['sampler']
    acquisition_settings(sampler)
    driver_settings(sampler)
    batching_settings(sampler)
    labeling_batch_settings(configs['QM'])
    for field in TASKS:
        module, name = master[field].rsplit('.', 1)
        source = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if not source.is_file():
            source = path.parent.joinpath(*module.split('.')).with_suffix('.py')
        assert source.is_file(), master[field]
        tree = ast.parse(source.read_text())
        assert any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and node.name == name for node in tree.body), master[field]
    assert all(len(spec) == 3 for spec in master['properties_list'].values())


@pytest.mark.parametrize('path', MASTERS, ids=lambda p: str(p.relative_to(ROOT/'examples')))
def test_example_tasks_can_be_imported(path, monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    for field in TASKS:
        module, name = read(path)[field].rsplit('.', 1)
        if module == 'alframework.ml_interfaces.neurochem_interface':
            loaded = import_neurochem_with_fake_backend(monkeypatch)
        else:
            loaded = importlib.import_module(module)
        assert callable(getattr(loaded, name))


@pytest.mark.parametrize('name', ['mlmd_config_torchsim.json', 'mlmd_config_alchemi.json'])
def test_packed_water_configs_are_valid(name):
    config = read(ROOT/'examples/simple_water'/name)
    assert driver_settings(config)['strategy'] == 'batched'
    batching_settings(config)
    acquisition_settings(config)
