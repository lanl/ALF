"""Adsorbate membership, periodic geometry and rejection in the shared MD sampler."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.samplers.batched_mlmd import ASEEngine, sample_trajectories
from alframework.tools.molecules import MoleculesObject
from alframework.tools.surface_screen import anomaly_check_settings, prepare_surface_anomaly_check
from tests.helpers.engines import require_engine
from tests.helpers.fakes import FixedCalculator


ANOMALY_CHECK = {'enabled': True}
SETTINGS = anomaly_check_settings({'anomaly_check': ANOMALY_CHECK})
SAMPLER = dict(dt=0.5, maxt=0.004, Ncheck=2, Escut=0.1, Fscut=100., distcut=0.3, min_time=0.,
               srt_temp=[0., 0.], end_temp=[0., 0.], amp_temp=[0., 0.], per_temp=[1., 1.],
               amp_dens=None, per_dens=None, end_dens=None, anomaly_check=ANOMALY_CHECK)
# Keep every trajectory running to its end, so one retirement can be seen beside the others.
CONTINUE = dict(trajectory_policy='continue', selection=dict(mode='cohort_top_k', cohort_size=2, k=2))


def slab(name, *, legacy=False, padded=False):
    """Two Cu atoms under two C2 ligands; the first ligand straddles the periodic a boundary."""
    atoms = Atoms('Cu2C4', tags=[0, 1, 2, 2, 2, 2], pbc=True,
                  cell=[[10, 0, 0], [2, 9, 0], [0, 0, 24]],
                  positions=[[0, 0, 3], [0, 0, 5], [0.2, 0, 7], [8.8, 0, 7],
                             [1.75, 0, 7.5], [3.3, 0, 7.5]])
    if legacy:
        atoms = atoms[:4]
        metadata = {'adsorbate_file': 'ligand.xyz'}
    else:
        metadata = {'adsorbates': [{'atom_indices': [2, 3]}, {'atom_indices': [4, 5]}]}
    if padded:  # One more substrate atom, so a calculator can tell the two slabs apart.
        atoms += Atoms('Cu', positions=[[5, 4.5, 3]], tags=[0])
    molecule = MoleculesObject(atoms, name)
    molecule.update_metadata({'surface_builder': metadata})
    return molecule


def zero_force_models(atom_count):
    return [FixedCalculator(energy, np.zeros((atom_count, 3))) for energy in (0., 1.)]


def split_first_ligand(atoms):
    atoms.positions[3, 0] = 2.75


def lift_ligand(atoms):
    atoms.positions[2:, 2] += 4.


# --- Settings ---------------------------------------------------------------------------------

def test_anomaly_check_is_off_by_default():
    assert anomaly_check_settings({}) is None


@pytest.mark.parametrize(('raw', 'message'), [
    ({'enabled': 'yes'}, 'Boolean enabled setting'),
    ({'enabled': True, 'desorption_distance': 4.}, 'Unknown anomaly_check settings: desorption_distance'),
    ({'enabled': True, 'connectivity_scale': 0.}, 'connectivity_scale must be finite and positive'),
    ({'enabled': True, 'desorption_distance_A': float('inf')}, 'desorption_distance_A must be finite and positive'),
], ids=['enabled-not-boolean', 'unknown-key', 'zero-scale', 'infinite-distance'])
def test_anomaly_check_refuses_invalid_settings(raw, message):
    with pytest.raises(ValueError, match=message):
        anomaly_check_settings({'anomaly_check': raw})


# --- Fragmentation ----------------------------------------------------------------------------

@pytest.fixture
def fragmented_report():
    """The screen's report after the first ligand splits across the periodic boundary."""
    molecule = slab('coverage')
    atoms = molecule.get_atoms()
    guard = prepare_surface_anomaly_check(molecule, SETTINGS)
    guard.inspect(atoms, 0, 0.)
    split_first_ligand(atoms)
    return guard.inspect(atoms, 1, 0.001)


def test_ligands_bonded_across_the_periodic_boundary_pass():
    molecule = slab('coverage')
    report = prepare_surface_anomaly_check(molecule, SETTINGS).inspect(molecule.get_atoms(), 0, 0.)
    assert report['surface_reject_reason'] is None


def test_a_split_ligand_is_rejected_as_fragmented(fragmented_report):
    assert fragmented_report['surface_reject_reason'] == 'adsorbate_fragmentation'


def test_fragments_name_the_separated_atoms(fragmented_report):
    assert fragmented_report['surface_adsorbates'][0]['fragments'] == [[2], [3]]


def test_fragmentation_is_reported_per_ligand(fragmented_report):
    assert 'fragments' not in fragmented_report['surface_adsorbates'][1]


def test_monatomic_adsorbate_passes_connectivity_screen():
    molecule = slab('single-atom', legacy=True)
    atoms = molecule.get_atoms()[:3]
    molecule.update_atoms(atoms)
    report = prepare_surface_anomaly_check(molecule, SETTINGS).inspect(atoms, 0, 0.)
    assert report['surface_reject_reason'] is None
    assert 'fragments' not in report['surface_adsorbates'][0]


# --- Desorption -------------------------------------------------------------------------------

@pytest.fixture
def desorbed():
    """A single-ligand adslab's screen and atoms, inspected at step 0 and after a 4 Å lift at step 5."""
    molecule = slab('legacy', legacy=True)
    atoms = molecule.get_atoms()
    guard = prepare_surface_anomaly_check(molecule, SETTINGS)
    initial = guard.inspect(atoms, 0, 0.)
    lift_ligand(atoms)
    return guard, atoms, initial, guard.inspect(atoms, 5, 0.0025)


def test_a_ligand_in_contact_records_no_desorption_at_the_start(desorbed):
    _, _, initial, _ = desorbed
    assert initial['surface_adsorbates'][0]['first_desorption'] is None


def test_desorption_does_not_reject_the_trajectory(desorbed):
    _, _, _, report = desorbed
    assert report['surface_reject_reason'] is None


def test_desorption_records_the_first_step_time_and_distance(desorbed):
    _, _, _, report = desorbed
    event = report['surface_adsorbates'][0]['first_desorption']
    assert event['step'] == 5 and event['time_ps'] == pytest.approx(0.0025)
    assert event['min_surface_distance_A'] > 4.0


def test_desorption_keeps_the_first_event(desorbed):
    guard, atoms, _, report = desorbed
    first = report['surface_adsorbates'][0]['first_desorption']
    assert guard.inspect(atoms, 6, 0.003)['surface_adsorbates'][0]['first_desorption'] == first


def test_a_ligand_that_starts_detached_records_no_desorption():
    molecule = slab('legacy', legacy=True)
    lift_ligand(molecule.get_atoms())
    report = prepare_surface_anomaly_check(molecule, SETTINGS).inspect(molecule.get_atoms(), 0, 0.)
    assert report['surface_adsorbates'][0]['first_desorption'] is None


@pytest.mark.parametrize('metadata', [{}, {'adsorbates': []}], ids=['clean', 'zero-coverage'])
def test_clean_slab_skips_ligand_checks(metadata):
    molecule = MoleculesObject(slab('coverage').get_atoms()[:2], 'clean')
    molecule.update_metadata({'surface_builder': metadata})
    assert prepare_surface_anomaly_check(molecule, SETTINGS) is None


# --- In the sampler ---------------------------------------------------------------------------

@pytest.mark.parametrize('positions', ['broken', 'duplicate-membership'])
def test_initial_rejection_happens_before_engine_construction(positions):
    molecule = slab('invalid')
    if positions == 'broken':
        split_first_ligand(molecule.get_atoms())
        reason = 'adsorbate_fragmentation'
    else:
        molecule.get_metadata()['surface_builder']['adsorbates'][1]['atom_indices'] = [3, 4]
        reason = 'adsorbate_identity'

    def unexpected_engine(*args, **kwargs):
        pytest.fail('A rejected initial structure reached engine construction')

    result, = sample_trajectories([molecule], zero_force_models(6), SAMPLER, engine_class=unexpected_engine)
    metadata = result.get_metadata()
    assert result.get_atoms() is None
    assert metadata['screen_reject_reason'] == reason
    assert metadata['screen_audit']['rejected']['step'] == 0
    assert metadata['acquisition_candidates'] == []


class StretchOneLigand(ASEEngine):
    """Splits trajectory 0's first ligand from step 3, and lifts trajectory 1's ligands at step 3."""

    def step(self, indices, n):
        failures = super().step(indices, n)
        if 0 in indices and self.steps[0] >= 3:
            split_first_ligand(self.atoms(0))
        if 1 in indices and self.steps[1] == 3:
            self.atoms(1).positions[2:, 2] += 5.
        return failures

    def observe(self, index):
        assert index != 0 or self.steps[0] < 3, 'The fragmented frame reached scoring'
        return super().observe(index)


@pytest.fixture(scope='module')
def stretched_run(tmp_path_factory):
    """A broken and an intact trajectory run together on the ASE engine, with their screen audit."""
    audit_dir = tmp_path_factory.mktemp('audit')
    config = dict(SAMPLER, screen_audit_dir=str(audit_dir), acquisition=CONTINUE)
    broken, intact = sample_trajectories([slab('broken'), slab('intact')], zero_force_models(6), config,
                                        engine_class=StretchOneLigand)
    broken, intact = broken.get_metadata(), intact.get_metadata()
    audit = json.loads(Path(broken['screen_audit_path']).read_text())['broken']
    return broken, intact, audit


def test_fragmentation_retires_the_broken_trajectory(stretched_run):
    broken, _, _ = stretched_run
    assert (broken['trajectory_outcome'], broken['screen_reject_reason']) == ('rejected', 'adsorbate_fragmentation')


def test_fragmentation_stops_at_the_checkpoint_that_found_it(stretched_run):
    broken, _, _ = stretched_run
    assert broken['step'] == 3


def test_a_fragmented_frame_is_not_an_acquisition_candidate(stretched_run):
    broken, _, _ = stretched_run
    assert [candidate['step'] for candidate in broken['acquisition_candidates']] == [1]


def test_fragmentation_leaves_the_other_trajectory_running(stretched_run):
    _, intact, _ = stretched_run
    assert intact['trajectory_outcome'] == 'eligible' and intact['step'] > 3


def test_the_other_trajectory_records_its_desorption(stretched_run):
    _, intact, _ = stretched_run
    assert intact['surface_adsorbates'][0]['first_desorption']['step'] == 3


def test_fragmentation_audit_keeps_the_initial_last_valid_and_rejected_frames(stretched_run):
    _, _, audit = stretched_run
    assert [audit[stage]['step'] for stage in ('initial', 'last_valid', 'rejected')] == [0, 1, 3]


def test_fragmentation_audit_names_the_fragments(stretched_run):
    _, _, audit = stretched_run
    assert audit['rejected']['diagnostics']['surface_adsorbates'][0]['fragments'] == [[2], [3]]


@pytest.mark.parametrize('engine', ['ase', 'torchsim', 'alchemi'])
def test_intact_ligands_pass_anomaly_check_on_each_engine(engine):
    require_engine(engine)
    config = dict(SAMPLER, maxt=0.001, Ncheck=1, device='cpu', batching={'batch_size': 1},
                  driver={'strategy': 'batched', 'engine': engine})
    result, = sample_trajectories([slab('stable')], zero_force_models(6), config)
    metadata = result.get_metadata()
    assert metadata['trajectory_outcome'] == 'eligible'
    assert metadata['screen_reject_reason'] is None
    assert len(metadata['surface_adsorbates']) == 2
    assert all('fragments' not in ligand for ligand in metadata['surface_adsorbates'])


class PullFirstLigandApart(Calculator):
    """Pushes atoms 2 and 3 apart along a in the six-atom slab only; the padded slab feels no force."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, energy):
        super().__init__()
        self.energy = energy

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(self.atoms), 3))
        if len(self.atoms) == 6:
            forces[2, 0], forces[3, 0] = 200., -200.
        self.results = dict(energy=self.energy, forces=forces)


@pytest.mark.parametrize('engine', ['torchsim', 'alchemi'])
def test_packed_engines_retire_only_the_fragmented_trajectory(engine):
    require_engine(engine)
    config = dict(SAMPLER, device='cpu', batching={'batch_size': 2}, acquisition=CONTINUE,
                  driver={'strategy': 'batched', 'engine': engine})
    models = [PullFirstLigandApart(energy) for energy in (0., 1.)]
    broken, intact = sample_trajectories([slab('pulled'), slab('padded', padded=True)], models, config)
    broken, intact = broken.get_metadata(), intact.get_metadata()
    assert (broken['trajectory_outcome'], broken['screen_reject_reason']) == ('rejected', 'adsorbate_fragmentation')
    assert intact['trajectory_outcome'] == 'eligible' and intact['step'] > broken['step']
