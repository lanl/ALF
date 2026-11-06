"""Engine mechanics and optional-dependency boundaries shared by both adapters."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from alframework.samplers import alchemi_engine
from tests.helpers.engines import Tether, engine_class


def test_starting_state_survives_the_engine(engine_name):
    atoms = Atoms(
        "OH2",
        positions=[[0.2, 0.1, 0.3], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]],
        cell=[[8.0, 0.0, 0.0], [0.4, 7.0, 0.0], [0.3, 0.2, 6.0]],
        pbc=True,
    )
    velocities = np.array([[0.01, -0.02, 0.03], [-0.04, 0.01, 0.02], [0.02, 0.03, -0.01]])
    atoms.set_velocities(velocities)
    energy = atoms.get_kinetic_energy()
    engine = engine_class(engine_name)(
        [atoms.copy()], [Tether(0.0), Tether(0.0)], 0.25, [0.0], [np.random.RandomState(5)], friction=0.0
    )
    np.testing.assert_allclose(engine.atoms(0).get_velocities(), velocities, rtol=1e-14, atol=1e-14)
    assert engine.atoms(0).get_kinetic_energy() == pytest.approx(energy, rel=1e-14)
    engine.scale_cell(0, 1.1)
    np.testing.assert_allclose(engine.atoms(0).cell.array, atoms.cell.array * 1.1)
    np.testing.assert_allclose(engine.atoms(0).get_velocities(), velocities, rtol=1e-14, atol=1e-14)
    assert not engine.step([0], 3)
    engine.finish(0)
    frozen = engine.atoms(0).positions.copy()
    assert engine.step([], 2) == {}
    np.testing.assert_array_equal(engine.atoms(0).positions, frozen)


def test_atom_constraints_are_rejected(engine_name):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NotImplementedError, match="constraints"):
        engine_class(engine_name)([atoms], [Tether(1.0), Tether(2.0)], 0.1, [0.0], [np.random.RandomState(0)])


@pytest.mark.parametrize("unsupported", ["0.1.0", "0.4.0"])
def test_unsupported_alchemi_version_is_rejected(monkeypatch, unsupported):
    monkeypatch.setattr(alchemi_engine, "version", lambda name: unsupported)
    with pytest.raises(ImportError, match="0.2.0"):
        alchemi_engine.require_alchemi()


def test_ase_engine_runs_without_torch():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','torch_sim','nvalchemi','nvalchemiops','warp'}:
   raise ImportError('Optional engine blocked: '+fullname)
sys.meta_path.insert(0,Block())
from alframework.samplers.batched_mlmd import make_engine
from alframework.samplers.packed_engine import PackedState
from tests.helpers.fakes import spring_ensemble
from ase import Atoms
import numpy as np
engine=make_engine('ase',[Atoms('H2',positions=[[0,0,0],[1,0,0]])],spring_ensemble(),.1,[0.],
                   [np.random.RandomState(0)])
assert not engine.step([0],1)
assert 'torch' not in sys.modules
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
