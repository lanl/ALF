"""Legacy inference loading must preserve checkpoints without changing cwd."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_calculator


def test_absolute_checkpoint_paths_allow_concurrent_loading(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    directory = Path.cwd()
    calls = []
    for index in range(2):
        member = tmp_path / str(index)
        member.mkdir()
        (member / 'experiment_structure.pt').write_text(f'member-{index}')
        (member / 'best_checkpoint.pt').write_text(f'checkpoint-{index}')
        (member / 'best_model.pt').write_text('wrong-format')

    def load(structure, state, map_location):
        assert Path(structure).is_absolute() and Path(state).is_absolute()
        assert Path(state).name == 'best_checkpoint.pt'
        assert map_location == 'cpu'
        barrier.wait(timeout=10)
        identity = (Path(structure).read_text(), Path(state).read_text())
        calls.append(identity)
        model = SimpleNamespace(node_from_name=lambda key: (identity, key))
        return {'training_modules': SimpleNamespace(model=model)}

    class Calculator:
        def __init__(self, energy_node, en_unit):
            self.node = energy_node
            self.unit = en_unit
            self.conversions = []

        def to(self, target):
            self.conversions.append(target)

    serialization = ModuleType('hippynn.experiment.serialization')
    serialization.load_checkpoint = load
    interface = ModuleType('hippynn.interfaces.ase_interface')
    interface.HippynnCalculator = Calculator
    monkeypatch.setitem(sys.modules, serialization.__name__, serialization)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32', device=lambda x: x))

    def forbidden(*args):
        pytest.fail('Inference loading must not change process cwd')

    monkeypatch.setattr(os, 'chdir', forbidden)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: HIPNN_ASE_calculator(
            os.path.relpath(tmp_path / str(i), directory), energy_key='custom-energy', device='cpu'), range(2)))
    for index, calc in enumerate(results):
        assert calc.node == ((f'member-{index}', f'checkpoint-{index}'), 'custom-energy')
        assert calc.unit == 1.
        assert calc.conversions == ['float32', 'cpu']
    assert len(calls) == 2
    assert Path.cwd() == directory

    def missing(structure, state, **kwargs):
        Path(structure).read_bytes()

    serialization.load_checkpoint = missing
    with pytest.raises(FileNotFoundError):
        HIPNN_ASE_calculator(tmp_path / 'missing', device='cpu')
    assert Path.cwd() == directory


def test_concurrent_trained_members_match_legacy_predictions(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('hippynn')
    from ase import units
    from hippynn.experiment import serialization
    from hippynn.interfaces.ase_interface import HippynnCalculator
    from hippynn.tools import active_directory
    from tests.test_acquisition import TRAINED_ENSEMBLE, WATER

    members = [TRAINED_ENSEMBLE / f'model-{i:02d}' for i in range(2)]
    if not all((member / 'best_checkpoint.pt').is_file() for member in members):
        pytest.skip('trained Darwin checkpoint members not available')

    expected = []
    # Capture the original loader's predictions before forbidding cwd changes.
    for member in members:
        with active_directory(str(member)):
            bundle = serialization.load_checkpoint_from_cwd(map_location='cpu')
        calc = HippynnCalculator(bundle['training_modules'].model.node_from_name('energy'), en_unit=units.eV)
        calc.to(torch.float32)
        calc.to(torch.device('cpu'))
        atoms = WATER.copy()
        atoms.calc = calc
        expected.append((atoms.get_potential_energy(), atoms.get_forces()))

    original_load = serialization.load_checkpoint
    barrier = threading.Barrier(2)

    def concurrent_load(*args, **kwargs):
        barrier.wait(timeout=30)
        return original_load(*args, **kwargs)

    def forbidden(*args):
        pytest.fail('Inference loading must not change process cwd')

    monkeypatch.setattr(serialization, 'load_checkpoint', concurrent_load)
    monkeypatch.setattr(os, 'chdir', forbidden)
    before = Path.cwd()

    def predict(index):
        atoms = WATER.copy()
        atoms.calc = HIPNN_ASE_calculator(os.path.relpath(members[index]), device='cpu')
        return atoms.get_potential_energy(), atoms.get_forces()

    with ThreadPoolExecutor(max_workers=2) as pool:
        predictions = list(pool.map(predict, [0, 1, 1, 0]))
    for index, (energy, forces) in zip([0, 1, 1, 0], predictions):
        np.testing.assert_allclose(energy, expected[index][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(forces, expected[index][1], rtol=1e-5, atol=1e-5)
    assert Path.cwd() == before
