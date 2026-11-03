"""Periodic and overlap screening at the shared helper and sampler boundaries."""
import numpy as np
import pytest
from ase import Atoms

from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.samplers.batched_mlmd import minimum_distance, sample_trajectories
from alframework.samplers.mlmd_sampling import mlmd_sampling
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import FixedCalculator


@pytest.mark.parametrize('atoms, expected', [
    (Atoms('H2', positions=[[.1, 0, 0], [9.9, 0, 0]], cell=[10]*3, pbc=True), .2),
    (Atoms('H2', positions=[[.1, 0, 0], [9.9, 0, 0]], cell=[10]*3), 9.8),
    (Atoms('H3', positions=[[0, 0, 0], [0, 0, 0], [2, 0, 0]]), 0.),
    (Atoms('H2', positions=[[0, 0, 0], [10, 0, 0]], cell=[10]*3, pbc=True), 0.),
    (Atoms('H2', positions=[[.1, .1, 0], [9.9, 9.9, 0]], cell=[10]*3,
           pbc=[True, False, False]), np.hypot(.2, 9.8)),
    (Atoms('H2', scaled_positions=[[.1, .01, .1], [.1, .99, .1]],
           cell=[[8, 0, 0], [2, 7, 0], [1, .5, 6]], pbc=True), .02*np.sqrt(53)),
    (Atoms('He', cell=[.1]*3, pbc=True), np.inf),
    (Atoms(), np.inf),
])
def test_minimum_distance(atoms, expected):
    positions, cell, pbc = atoms.positions.copy(), atoms.cell.copy(), atoms.pbc.copy()
    assert minimum_distance(atoms) == pytest.approx(expected)
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(atoms.cell, cell)
    np.testing.assert_array_equal(atoms.pbc, pbc)


@pytest.mark.parametrize('legacy', [True, False])
@pytest.mark.parametrize('positions, periodic, cutoff, rejected, expected', [
    ([[.1, 0, 0], [9.9, 0, 0]], True, .5, True, .2),
    ([[.1, 0, 0], [9.9, 0, 0]], False, .5, False, 9.8),
    ([[0, 0, 0], [0, 0, 0]], False, .5, True, 0.),
    ([[0, 0, 0], [10, 0, 0]], True, .5, True, 0.),
    ([[0, 0, 0], [1, 0, 0]], False, 1., False, 1.),
])
def test_sampler_screening(legacy, positions, periodic, cutoff, rejected, expected):
    atoms = Atoms('H2', positions=positions, cell=[10]*3, pbc=periodic)
    molecule = MoleculesObject(atoms, 'screen')
    models = [FixedCalculator(0., np.zeros((2, 3))) for _ in range(2)]
    config = dict(dt=.5, maxt=.001, Ncheck=1, Escut=1., Fscut=1., distcut=cutoff, min_time=0.)
    if legacy:
        result = mlmd_sampling(molecule, MLMD_calculator(models, well_params=None), **config,
                               Tamp=0., Tper=1., Tsrt=0., Tend=0., Ramp=None, Rper=None, Rend=None)
    else:
        config.update(srt_temp=[0., 0.], end_temp=[0., 0.], amp_temp=[0., 0.], per_temp=[1., 1.],
                      amp_dens=None, per_dens=None, end_dens=None,
                      MLMD_calculator_options={'well_params': None})
        result, = sample_trajectories([molecule], models, config)
        assert result.get_metadata()['trajectory_outcome'] == ('rejected' if rejected else 'completed')
    assert bool(result.get_metadata()['distcrit']) is rejected
    assert result.get_metadata()['distmin'] == pytest.approx(expected)
    assert result.get_atoms() is None
