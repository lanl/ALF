"""Thermal starts for MLMD: an opt-in sampler policy that draws momenta only for atoms that have none.

The draw has its own random stream, so it never changes the schedule or Langevin draws, and dynamics engines
keep whatever momenta they are given.
"""
import numpy as np
import pytest
from ase import Atoms, units
from ase.constraints import FixAtoms
from parsl.app.errors import RemoteExceptionWrapper

from alframework.samplers import mlmd_sampling as scalar_mlmd
from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.samplers.udd_sampling import simple_udd_sampling_task
from alframework.tools.molecules import MoleculesObject
from alframework.tools.sampler_coordinator import driver_settings
from alframework.tools.velocities import initialize_velocities, velocity_initialization_mode
from tests.helpers.engines import require_engine
from tests.helpers.fakes import FixedCalculator

THERMAL = 'maxwell_boltzmann_if_missing'

# A short constant-temperature run at 400 K; random_seed makes the batched runs repeatable.
CONFIG = dict(dt=0.5, maxt=0.02, Ncheck=10, Escut=0.05, Fscut=0.05, min_time=0.005, distcut=0.7,
              srt_temp=[400.0, 400.0], end_temp=[400.0, 400.0], amp_temp=[0.0, 0.0], per_temp=[1.0, 1.0],
              amp_dens=None, per_dens=None, end_dens=None, random_seed=42, friction_per_fs=0.01)


def gas(n=128):
    """``n`` argon atoms in a line, 4 Angstrom apart, with no momenta."""
    return Atoms('Ar' * n, positions=np.column_stack([np.arange(n) * 4.0, np.zeros((n, 2))]))


def ensemble(n):
    """Two models that agree on zero forces, so trajectories run to ``maxt`` without stopping on forces."""
    return [FixedCalculator(energy=energy, forces=np.zeros((n, 3))) for energy in (0.0, 0.2)]


def thermal(atoms, molecule_id='a', temperature=400.0, **options):
    """Draw thermal momenta for ``atoms`` and return the metadata the sampler records."""
    return initialize_velocities(atoms, temperature, THERMAL, molecule_id=molecule_id, **options)


@pytest.mark.parametrize('bad', [None, True, {}, 'thermal', 1])
def test_an_unknown_velocity_policy_is_rejected(bad):
    with pytest.raises(ValueError, match='velocity_initialization'):
        velocity_initialization_mode(bad)


@pytest.mark.parametrize('mode,temperature', [
    ('preserve', 600), (THERMAL, -1), (THERMAL, np.nan), (THERMAL, np.inf), (THERMAL, True), (THERMAL, '600'),
])
def test_a_bad_initial_temperature_stops_the_sampler_at_startup(mode, temperature):
    config = dict(velocity_initialization=mode, initial_velocity_temperature_K=temperature)
    with pytest.raises(ValueError, match='initial_velocity_temperature_K'):
        driver_settings(config)


def test_by_default_atoms_without_momenta_stay_cold():
    atoms = gas()
    metadata = initialize_velocities(atoms, 400.0, molecule_id='a')
    assert not atoms.has('momenta')
    assert not metadata['velocities_initialized']


def test_by_default_no_random_numbers_are_drawn():
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)
    initialize_velocities(gas(), 400.0, molecule_id='a')
    np.testing.assert_array_equal(np.random.random(5), expected)


@pytest.mark.parametrize('value', [0.0, 1.0], ids=['zeros', 'ones'])
def test_momenta_already_on_the_atoms_are_kept_even_when_zero(value):
    atoms = gas()
    momenta = np.full((len(atoms), 3), value)
    atoms.set_momenta(momenta)
    metadata = thermal(atoms, random_seed=1, initial_velocity_temperature_K=600.0)
    assert not metadata['velocities_initialized']
    assert metadata['initial_velocity_temperature_K'] is None
    np.testing.assert_array_equal(atoms.get_momenta(), momenta)


def test_drawn_momenta_follow_the_maxwell_boltzmann_distribution_for_each_mass():
    atoms = Atoms('HC' * 4000)
    metadata = thermal(atoms, molecule_id='distribution', random_seed=8)
    # Each component, divided by sqrt(m kB T), is a standard normal draw.
    normalized = atoms.get_momenta() / np.sqrt(atoms.get_masses()[:, None] * units.kB * 400.0)
    assert metadata['velocities_initialized'] and metadata['initial_velocity_seed']
    assert abs(normalized.mean()) < 0.025
    assert normalized.var() == pytest.approx(1.0, abs=0.025)
    assert atoms.get_temperature() == pytest.approx(400.0, rel=0.03)


def test_constrained_atoms_get_no_momentum():
    atoms = gas()
    atoms.set_constraint(FixAtoms(indices=[0]))
    thermal(atoms, molecule_id='constraint', random_seed=8)
    np.testing.assert_array_equal(atoms.get_momenta()[0], 0.0)


def test_with_a_seed_each_molecule_id_always_gets_the_same_momenta():
    first, second, other = gas(), gas(), gas()
    thermal(first, molecule_id='a', random_seed=42)
    thermal(second, molecule_id='a', random_seed=42)
    thermal(other, molecule_id='b', random_seed=42)
    np.testing.assert_array_equal(first.get_momenta(), second.get_momenta())
    assert not np.array_equal(first.get_momenta(), other.get_momenta())


def test_a_thermal_draw_leaves_the_global_random_stream_alone():
    np.random.seed(19)
    expected = np.random.random(3)
    np.random.seed(19)
    thermal(gas(), random_seed=42)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_a_zero_kelvin_draw_gives_zero_momenta():
    atoms = gas()
    thermal(atoms, molecule_id='zero', random_seed=1, initial_velocity_temperature_K=0.0)
    np.testing.assert_array_equal(atoms.get_momenta(), 0.0)


@pytest.mark.parametrize('temperature', [-1.0, np.nan, np.inf])
def test_a_bad_thermostat_temperature_is_rejected_when_drawing(temperature):
    with pytest.raises(ValueError, match='temperature'):
        thermal(gas(), molecule_id='bad', temperature=temperature)


@pytest.mark.parametrize('task', [scalar_mlmd.simple_mlmd_sampling_task, simple_udd_sampling_task],
                         ids=['scalar', 'udd'])
@pytest.mark.parametrize('initial_temperature', [None, 600.0])
def test_scalar_tasks_start_hot_and_record_the_draw(task, initial_temperature, monkeypatch):
    atoms = gas(512)
    monkeypatch.setattr(scalar_mlmd, 'load_module_from_config',
                        lambda *_: lambda path, device: ensemble(len(atoms)))
    config = dict(CONFIG, device='cpu', ase_calculator='unused', MLMD_calculator_options={},
                  velocity_initialization=THERMAL, udd_bias_weight=0.5)
    if initial_temperature is not None:
        config['initial_velocity_temperature_K'] = initial_temperature
    result = task.func(MoleculesObject(atoms, 'scalar'), config, 'model-{:04d}', 0, 1)
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    metadata = result.get_metadata()
    target = 400.0 if initial_temperature is None else initial_temperature
    assert metadata['velocities_initialized']
    assert metadata['initial_velocity_temperature_K'] == target
    assert metadata['initial_kinetic_temperature_K'] == pytest.approx(target, rel=0.12)
    # The draw sets the starting temperature only; the thermostat keeps its own schedule.
    assert metadata['thermostat_temperature_K'] == 400.0
    assert metadata['simulationtime'] == 0.005


def sample(names, config):
    """Run one batched task over molecules with these IDs; return the results by ID."""
    molecules = [MoleculesObject(gas(), name) for name in names]
    return {molecule.get_moleculeid(): molecule
            for molecule in sample_trajectories(molecules, ensemble(128), config, device='cpu')}


@pytest.mark.parametrize('engine', ['ase', 'torchsim', 'alchemi'])
@pytest.mark.parametrize('initial_temperature', [None, 600.0])
def test_every_engine_starts_hot_at_the_requested_temperature(engine, initial_temperature):
    if engine != 'ase':
        require_engine(engine)
    config = dict(CONFIG, velocity_initialization=THERMAL, driver=dict(engine=engine))
    if initial_temperature is not None:
        config['initial_velocity_temperature_K'] = initial_temperature
    target = 400.0 if initial_temperature is None else initial_temperature
    for molecule in sample(['a', 'b'], config).values():
        metadata = molecule.get_metadata()
        assert metadata['velocities_initialized']
        assert metadata['initial_velocity_temperature_K'] == target
        assert metadata['initial_kinetic_temperature_K'] == pytest.approx(target, rel=0.25)
        assert metadata['thermostat_temperature_K'] == 400.0


@pytest.mark.parametrize('engine', ['ase', 'torchsim', 'alchemi'])
def test_a_thermal_trajectory_does_not_depend_on_its_batch_mates(engine):
    if engine != 'ase':
        require_engine(engine)
    config = dict(CONFIG, velocity_initialization=THERMAL, driver=dict(engine=engine))
    together = sample(['a', 'b'], config)['a']
    alone = sample(['a'], config)['a']
    np.testing.assert_allclose(together.get_atoms().positions, alone.get_atoms().positions, atol=1e-12)
    np.testing.assert_allclose(together.get_atoms().get_momenta(), alone.get_atoms().get_momenta(), atol=1e-12)
    assert together.get_metadata()['initial_velocity_seed'] == alone.get_metadata()['initial_velocity_seed']


def test_by_default_a_batched_trajectory_starts_cold_and_the_thermostat_warms_it():
    metadata = sample(['cold'], CONFIG)['cold'].get_metadata()
    assert not metadata['velocities_initialized']
    assert metadata['initial_kinetic_temperature_K'] == 0.0
    assert 0.0 < metadata['kinetic_temperature_K'] < metadata['thermostat_temperature_K'] == 400.0
