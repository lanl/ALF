"""Thermal starts are optional sampler policy; engines preserve provided state."""
import numpy as np
import pytest
from ase import Atoms, units
from ase.constraints import FixAtoms
from parsl.app.errors import RemoteExceptionWrapper

from alframework.samplers import mlmd_sampling as scalar_mlmd
from alframework.samplers.batched_mlmd import sample_trajectories
from alframework.samplers.udd_sampling import simple_udd_sampling_task
from alframework.samplers.velocities import initialize_velocities, velocity_initialization_mode
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.engines import require_engine
from tests.helpers.fakes import FixedCalculator

THERMAL = 'maxwell_boltzmann_if_missing'


def gas(n=128):
    return Atoms('Ar'*n, positions=np.column_stack([np.arange(n)*4., np.zeros((n,2))]))


def ensemble(n):
    return [FixedCalculator(energy=e, forces=np.zeros((n,3))) for e in (0., .2)]


@pytest.mark.parametrize('bad', [None, True, {}, 'thermal', 1])
def test_invalid_policy_is_rejected(bad):
    with pytest.raises(ValueError, match='velocity_initialization'):
        velocity_initialization_mode(bad)


def test_default_preserves_missing_momenta_and_random_stream():
    atoms=gas()
    np.random.seed(123); expected=np.random.random(5)
    np.random.seed(123)
    metadata=initialize_velocities(atoms,400.,molecule_id='a')
    assert not atoms.has('momenta') and not metadata['velocities_initialized']
    np.testing.assert_array_equal(np.random.random(5),expected)


@pytest.mark.parametrize('explicit_zero', [False, True])
def test_thermal_option_preserves_explicit_momenta_even_zeros(explicit_zero):
    atoms=gas();momenta=np.zeros((len(atoms),3)) if explicit_zero else np.ones((len(atoms),3))
    atoms.set_momenta(momenta)
    metadata=initialize_velocities(atoms,400.,THERMAL,molecule_id='a',random_seed=1)
    assert not metadata['velocities_initialized']
    np.testing.assert_array_equal(atoms.get_momenta(),momenta)


def test_thermal_draw_has_mass_scaled_maxwell_statistics_and_applies_constraints():
    atoms=Atoms('HC'*4000)
    metadata=initialize_velocities(atoms,400.,THERMAL,molecule_id='distribution',random_seed=8)
    normalized=atoms.get_momenta()/np.sqrt(atoms.get_masses()[:,None]*units.kB*400.)
    assert metadata['velocities_initialized'] and metadata['initial_velocity_seed']
    assert abs(normalized.mean())<.025
    assert normalized.var()==pytest.approx(1.,abs=.025)
    assert atoms.get_temperature()==pytest.approx(400.,rel=.03)
    constrained=gas();constrained.set_constraint(FixAtoms(indices=[0]))
    initialize_velocities(constrained,400.,THERMAL,molecule_id='constraint',random_seed=8)
    np.testing.assert_array_equal(constrained.get_momenta()[0],0.)


def test_velocity_draw_is_repeatable_per_id_and_does_not_consume_global_rng():
    original=gas();first=original.copy();second=original.copy();other=original.copy()
    np.random.seed(19);expected=np.random.random(3);np.random.seed(19)
    initialize_velocities(first,400.,THERMAL,molecule_id='a',random_seed=42)
    initialize_velocities(second,400.,THERMAL,molecule_id='a',random_seed=42)
    initialize_velocities(other,400.,THERMAL,molecule_id='b',random_seed=42)
    np.testing.assert_array_equal(first.get_momenta(),second.get_momenta())
    assert not np.array_equal(first.get_momenta(),other.get_momenta())
    np.testing.assert_array_equal(np.random.random(3),expected)


def test_zero_temperature_draw_is_zero_and_bad_temperature_is_rejected():
    atoms=gas()
    initialize_velocities(atoms,0.,THERMAL,molecule_id='zero',random_seed=1)
    np.testing.assert_array_equal(atoms.get_momenta(),0.)
    for temperature in [-1.,np.nan,np.inf]:
        with pytest.raises(ValueError,match='temperature'):
            initialize_velocities(gas(),temperature,THERMAL,molecule_id='bad')


CONFIG=dict(dt=.5,maxt=.02,Ncheck=10,Escut=.05,Fscut=.05,min_time=.005,distcut=.7,
            srt_temp=[400.,400.],end_temp=[400.,400.],amp_temp=[0.,0.],per_temp=[1.,1.],
            amp_dens=None,per_dens=None,end_dens=None,random_seed=42,friction_per_fs=.01)


@pytest.mark.parametrize('task', [scalar_mlmd.simple_mlmd_sampling_task, simple_udd_sampling_task],
                         ids=['scalar', 'udd'])
def test_scalar_tasks_select_a_thermal_frame_before_equilibration(task, monkeypatch):
    atoms = gas(512)
    monkeypatch.setattr(scalar_mlmd, 'load_module_from_config',
                        lambda *_: lambda path, device: ensemble(len(atoms)))
    config = dict(CONFIG, device='cpu', ase_calculator='unused', MLMD_calculator_options={},
                  velocity_initialization=THERMAL, udd_bias_weight=0.5)
    result = task.func(MoleculesObject(atoms, 'scalar'), config, 'model-{:04d}', 0, 1)
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    metadata = result.get_metadata()
    assert metadata['velocities_initialized']
    assert 350.0 < metadata['initial_kinetic_temperature_K'] < 450.0
    assert 350.0 < metadata['kinetic_temperature_K'] < 450.0
    assert metadata['thermostat_temperature_K'] == 400.0
    assert metadata['simulationtime'] == 0.005


@pytest.mark.parametrize('name',['ase','torchsim','alchemi'])
def test_thermal_starts_work_on_each_engine_and_do_not_depend_on_grouping(name):
    if name!='ase':
        require_engine(name)
    config=dict(CONFIG,velocity_initialization=THERMAL,driver=dict(engine=name))
    def run(names):
        molecules=[MoleculesObject(gas(),mid) for mid in names]
        results=sample_trajectories(molecules,ensemble(128),config,device='cpu')
        return {m.get_moleculeid():m for m in results}
    together=run(['a','b'])
    solo=run(['a'])['a']
    for molecule in together.values():
        meta=molecule.get_metadata()
        assert meta['velocities_initialized']
        assert 250.<meta['kinetic_temperature_K']<550.
        assert meta['thermostat_temperature_K']==400.
    np.testing.assert_allclose(together['a'].get_atoms().positions,solo.get_atoms().positions,atol=1e-12)
    np.testing.assert_allclose(together['a'].get_atoms().get_momenta(),solo.get_atoms().get_momenta(),atol=1e-12)
    assert together['a'].get_metadata()['initial_velocity_seed']==solo.get_metadata()['initial_velocity_seed']


def test_batched_default_still_has_cold_short_time_behavior():
    result=sample_trajectories([MoleculesObject(gas(),'cold')],ensemble(128),CONFIG)[0]
    metadata=result.get_metadata()
    assert not metadata['velocities_initialized'] and metadata['initial_kinetic_temperature_K']==0.
    assert 20.<metadata['kinetic_temperature_K']<80.
    assert metadata['thermostat_temperature_K']==400.
