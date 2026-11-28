"""Optional excited-state properties keep their units, shapes and invariances."""

import json

import numpy as np
import pytest
from ase import Atoms

from alframework.tools.data_shards import store_current_data
from alframework.tools.molecules import MOLECULE_ID, MoleculesObject


# --- Finite-field permanent dipoles -----------------------------------------------------------------
# A stand-in for the SCF and TDA solves whose energies are linear in the field, so the dipoles are known:
# E_S0(F) = E0 − μ0·F and E_S1(F) = E_S0(F) + ω − (μ1 − μ0)·F, in atomic units. It checks ALF's field loop,
# signs, nuclear term, units and root following, not the electronic structure.

WATER = Atoms('OH2', positions=[[0., 0., 0.], [0., .757, .587], [0., -.757, .587]])
DIPOLES_AU = np.array([[.1, -.2, .7], [.4, .3, -.5]])  # S0 and S1, e·bohr
DIPOLE_SETTINGS = {'charge': 0, 'multiplicity': 1, 'basis': 'stand-in', 'nroots': 1, 'scf_conv_tol': 1.e-9,
                   'tda_conv_tol': 1.e-6}


class FieldStandIn:
    """``solve_tda``, ``root_data`` and ``compare_roots`` for a molecule with known state dipoles."""

    charges = np.array([8., 1., 1.])
    coords = np.array([[0., 0., 0.], [0., 1.43, 1.11], [0., -1.43, 1.11]])

    def __init__(self, *, reorder=False, overlap=.95, reported_s0=DIPOLES_AU[0], cubic=0.):
        self.reorder, self.overlap, self.reported_s0, self.cubic = reorder, overlap, np.asarray(reported_s0), cubic

    def solve_tda(self, atoms, basis, settings, nroots, field=None):
        from types import SimpleNamespace

        field = np.zeros(3) if field is None else np.asarray(field)
        nuclear = self.charges @ self.coords
        molecule = SimpleNamespace(atom_charges=lambda: self.charges, atom_coords=lambda: self.coords)
        # The SCF energy holds the electronic field term; ALF removes the nuclear one.
        scf = SimpleNamespace(e_tot=-76. - DIPOLES_AU[0] @ field + field @ nuclear,
                              dip_moment=lambda unit, verbose: self.reported_s0)
        roots = [('S1', .2 - (DIPOLES_AU[1] - DIPOLES_AU[0]) @ field - self.cubic * np.sum(field ** 3)),
                 ('S2', .25), ('S3', .3)][:nroots]
        if self.reorder and field.sum() > 0:
            roots[0], roots[1] = roots[1], roots[0]
        return molecule, scf, SimpleNamespace(e=np.array([energy for _, energy in roots]),
                                              labels=[label for label, _ in roots])

    def root_data(self, molecule, scf, td):
        return td.labels

    def compare_roots(self, zero, roots, *, min_overlap, gap_shift_eV):
        overlaps = [self.overlap if label == zero[0] else .1 for label in roots]
        return {'root_matches_1_based': [roots.index(zero[0]) + 1], 'overlap_matrix': [overlaps]}


def finite_field_dipoles(monkeypatch, stand_in=None, *, settings=DIPOLE_SETTINGS, **options):
    from examples.molecular_excited_states import permanent_dipoles as module

    stand_in = stand_in or FieldStandIn()
    for name in ('solve_tda', 'root_data', 'compare_roots'):
        monkeypatch.setattr(module, name, getattr(stand_in, name))
    return module.permanent_dipoles(WATER, settings, **options)


def test_finite_field_dipoles_are_the_state_dipoles_in_e_angstrom(monkeypatch):
    from ase import units

    dipoles, checks = finite_field_dipoles(monkeypatch)
    np.testing.assert_allclose(dipoles, DIPOLES_AU * units.Bohr, atol=1.e-8)
    assert checks['finite_field_s0_error_eA'] < 1.e-8 and checks['minimum_s1_overlap'] == .95


def test_finite_field_dipoles_follow_s1_when_the_field_reorders_roots(monkeypatch):
    from ase import units

    dipoles, _ = finite_field_dipoles(monkeypatch, FieldStandIn(reorder=True))
    np.testing.assert_allclose(dipoles[1], DIPOLES_AU[1] * units.Bohr, atol=1.e-8)


def test_finite_field_dipoles_refuse_a_lost_s1_root(monkeypatch):
    with pytest.raises(ValueError, match='below 0.8'):
        finite_field_dipoles(monkeypatch, FieldStandIn(overlap=.5))


def test_finite_field_dipoles_refuse_a_ground_state_mismatch(monkeypatch):
    with pytest.raises(ValueError, match='Ground-state finite-field check failed'):
        finite_field_dipoles(monkeypatch, FieldStandIn(reported_s0=DIPOLES_AU[0] + [.01, 0., 0.]))


@pytest.mark.parametrize('change', [{'charge': 1}, {'multiplicity': 3}, {'solvent': 'water'}])
def test_finite_field_dipoles_refuse_charged_or_solvated_states(monkeypatch, change):
    with pytest.raises(ValueError, match='neutral, gas-phase singlet'):
        finite_field_dipoles(monkeypatch, settings={**DIPOLE_SETTINGS, **change})


def test_finite_field_dipoles_refuse_a_nonpositive_step(monkeypatch):
    with pytest.raises(ValueError, match='finite and positive'):
        finite_field_dipoles(monkeypatch, step=0.)


def test_finite_field_step_check_reports_the_halved_step_change(monkeypatch):
    _, checks = finite_field_dipoles(monkeypatch, check_step=True)
    assert checks['halved_field_s1_change_eA'] < 1.e-8


def test_finite_field_step_check_refuses_a_step_dependent_s1_dipole(monkeypatch):
    # A cubic field term makes the central difference depend on the step: about 0.75 β h² per component.
    with pytest.raises(ValueError, match='S1 finite-field step check failed'):
        finite_field_dipoles(monkeypatch, FieldStandIn(cubic=1.e6), check_step=True)


# --- Charge and oscillator readouts -----------------------------------------------------------------

def charge_head():
    """A two-state charge readout on water padded with one dummy atom, and its inputs."""
    torch = pytest.importorskip('torch')
    from examples.molecular_excited_states.properties import Readout

    head = Readout(3, 2, 'charge').double()
    features = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3)
    species = torch.tensor([[8, 1, 1, 0]])
    positions = torch.tensor([[[0., 0., 0.], [0., .757, .587], [0., -.757, .587], [1., 1., 1.]]],
                             dtype=torch.float64)
    return torch, head, features, species, positions


def test_charge_readout_is_neutral():
    torch, head, features, species, positions = charge_head()
    torch.testing.assert_close(head(features, species, positions)['charges'].sum(1),
                               torch.zeros(1, 2, dtype=torch.float64))


def test_charge_readout_ignores_padding():
    _, head, features, species, positions = charge_head()
    assert bool((head(features, species, positions)['charges'][:, -1] == 0).all())


def test_charge_readout_dipoles_rotate_with_the_molecule():
    torch, head, features, species, positions = charge_head()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    normal = head(features, species, positions)
    moved = head(features, species, positions @ rotation + 2.)
    torch.testing.assert_close(moved['dipoles_eA'], normal['dipoles_eA'] @ rotation)


def test_oscillator_labels_are_retained_in_root_order(tmp_path):
    pytest.importorskip('torch')
    from examples.molecular_excited_states.prepare import state_properties
    from examples.molecular_excited_states.properties import shard_frames

    frame = MoleculesObject(Atoms('H2', positions=[[0, 0, 0], [0, 0, .75]]), 'frame')
    frame.update_metadata({MOLECULE_ID: '[H][H]'})
    frame.store_results(dict(E0=-2., E1=1., E2=3., F0=np.zeros((2, 3)), F1=np.zeros((2, 3)), F2=np.zeros((2, 3)),
                             dE01=3., dE12=2., oscillator_strengths=np.array([.2, .01]),
                             transition_dipoles=np.ones((2, 3))))
    frame.set_converged_flag(True)
    properties = state_properties(2, optical=True)
    path = tmp_path / 'data-0000.h5'
    store_current_data(str(path), [frame], properties)
    saved, = shard_frames([path], properties, optical=True)
    np.testing.assert_array_equal(saved.targets, [.2, .01])
    np.testing.assert_array_equal(saved.excitation_eV, [3., 5.])


def fitted_charge_readout(tmp_path, monkeypatch):
    """A charge readout fit on stand-in features of one frozen member, with its train/validation/test frames."""
    torch = pytest.importorskip('torch')
    from examples.molecular_excited_states import properties as readout

    member = tmp_path / 'model-0000' / 'model-00'
    member.mkdir(parents=True)
    for name in ('experiment_structure.pt', 'best_model.pt'):
        (member / name).write_text('frozen')
    (member / 'training_input_manifest.json').write_text('{"frames": [{"molecule_id": "train"}]}')
    (tmp_path / 'master_config.json').write_text(json.dumps({'properties_list': {}}))
    frames = {
        key: readout.PropertyFrame(key, key, np.array([1, 8]), np.array([[0., 0., 0.], [1., 0., 0.]]),
                                   {}, np.array([[.2, 0., 0.], [.3, 0., 0.]]))
        for key in ('train', 'validation', 'test')
    }
    monkeypatch.setattr(readout, 'load_frames', lambda kind, paths, schema: [frames[paths[0]]])

    def featurize(batch, member, device, batch_size):
        return torch.tensor([[[1.], [0.]]] * len(batch)), torch.tensor([[1, 8]] * len(batch)), torch.tensor(
            np.stack([frame.positions for frame in batch]), dtype=torch.float32)

    monkeypatch.setattr(readout, 'featurize', featurize)
    output = tmp_path / 'charges.pt'
    assert readout.fit('charge', member, tmp_path, ['train'], ['validation'], output, epochs=4) >= 0.
    return readout, member, output


def test_charge_readout_fit_scores_neutral_charges(tmp_path, monkeypatch):
    readout, _, output = fitted_charge_readout(tmp_path, monkeypatch)
    result = readout.score(output, ['test'])
    assert result['states'] == [0, 1]
    np.testing.assert_allclose(np.sum(result['frames'][0]['charges_e'], axis=0), 0., atol=1.e-7)


def test_charge_readout_refuses_its_validation_frames(tmp_path, monkeypatch):
    readout, _, output = fitted_charge_readout(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='overlap'):
        readout.score(output, ['validation'])


def test_charge_readout_refuses_a_changed_member(tmp_path, monkeypatch):
    readout, member, output = fitted_charge_readout(tmp_path, monkeypatch)
    (member / 'best_model.pt').write_text('changed')
    with pytest.raises(ValueError, match='does not match'):
        readout.score(output, ['test'])
