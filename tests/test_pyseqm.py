"""PySEQM labeling: ground-state native batching and native CIS excited states."""
import numpy as np
import pytest
from ase import Atoms
import multiprocessing as mp
import time

from alframework.qm_interfaces import pyseqm_interface as pyseqm
from alframework.tools.molecules_class import MoleculesObject


# Deliberately not heaviest-first, so the adapter must sort and restore atom order.
METHANOL = Atoms("HCOHHH", positions=[[-0.36, 1.03, 0.0], [0.0, 0.0, 0.0], [1.43, 0.0, 0.0],
                                      [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89], [1.75, 0.9, 0.0]])
WATER = Atoms("HOH", positions=[[0.96, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.24, 0.93, 0.0]])
QM_CONFIG = {"method": "AM1", "scf_eps": 1e-8, "device": "cpu"}


def molecules(*atoms_list):
    return [MoleculesObject(atoms.copy(), f"mol-{index}") for index, atoms in enumerate(atoms_list)]


def test_pack_sorts_heaviest_first_pads_and_restores_order():
    coordinates, species, orders = pyseqm.pack_molecules(molecules(METHANOL, WATER))
    assert species.tolist() == [[8, 6, 1, 1, 1, 1], [8, 1, 1, 0, 0, 0]]
    np.testing.assert_allclose(coordinates[1, 3:], 0.0)
    for atoms, sorted_xyz, order in zip((METHANOL, WATER), coordinates, orders):
        np.testing.assert_allclose(pyseqm.restore_atom_order(sorted_xyz, order), atoms.get_positions())


def test_periodic_inputs_are_rejected():
    periodic = WATER.copy()
    periodic.set_cell(np.eye(3) * 8.0)
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="periodic"):
        pyseqm.pack_molecules(molecules(periodic))


def test_excited_state_parameters_are_rejected():
    with pytest.raises(ValueError, match="ground states"):
        pyseqm.pyseqm_parameters({"pyseqm_parameters": {"excited_states": {"n_states": 2}}})


def fake_runner(converged=None, fail_when=None):
    calls = []

    def run(coordinates, species, parameters, device, charges, mult):
        calls.append(len(coordinates))
        if fail_when is not None and fail_when(len(coordinates), species):
            raise RuntimeError("batched solve failed")
        # Per-atom forces equal to the sorted atomic number make order restoration visible.
        forces = np.repeat(species[:, :, None].astype(float), 3, axis=2)
        flags = np.ones(len(coordinates), bool) if converged is None else np.asarray(converged)
        return -species.sum(axis=1).astype(float), forces, flags

    return run, calls


def test_unconverged_member_is_flagged_and_others_kept():
    run, _ = fake_runner(converged=[True, False])
    labeled = pyseqm.label_pyseqm_batch(molecules(METHANOL, WATER), QM_CONFIG, runner=run)
    assert [mol.get_moleculeid() for mol in labeled] == ["mol-0", "mol-1"]
    assert [mol.check_convergence() for mol in labeled] == [True, False]
    assert "did not converge" in labeled[1].get_metadata()["labeling_error"]
    np.testing.assert_allclose(labeled[0].get_results()["forces"][:, 0], METHANOL.get_atomic_numbers())


def test_failed_batch_falls_back_to_individual_labels():
    # The batched call fails, as does any call containing water alone.
    run, calls = fake_runner(fail_when=lambda n, species: n > 1 or species[0, 0] == 8 and species[0, 1] == 1)
    labeled = pyseqm.label_pyseqm_batch(molecules(METHANOL, WATER), QM_CONFIG, runner=run)
    assert calls == [2, 1, 1]
    assert [mol.check_convergence() for mol in labeled] == [True, False]
    assert "batch error" in labeled[1].get_metadata()["labeling_error"]


@pytest.fixture(scope="module")
def seqm():
    return pytest.importorskip("seqm.ElectronicStructure")


def test_native_batch_matches_individual_pyseqm_evaluations(seqm):
    batch = pyseqm.label_pyseqm_batch(molecules(METHANOL, WATER, METHANOL), QM_CONFIG)
    singles = [pyseqm.label_pyseqm_batch(molecules(atoms), QM_CONFIG)[0] for atoms in (METHANOL, WATER)]
    assert all(mol.check_convergence() for mol in batch + singles)
    for batched, single in zip(batch, singles + singles[:1]):
        assert batched.get_results()["energy"] == pytest.approx(single.get_results()["energy"], abs=1e-6)
        np.testing.assert_allclose(batched.get_results()["forces"], single.get_results()["forces"], atol=1e-5)
    assert batch[1].get_results()["forces"].shape == (3, 3)


def test_forces_are_negative_energy_gradient_in_input_order(seqm):
    step = 1e-4
    base = pyseqm.label_pyseqm_batch(molecules(WATER), QM_CONFIG)[0].get_results()["forces"]
    for atom in range(len(WATER)):
        shifted = []
        for sign in (1, -1):
            atoms = WATER.copy()
            atoms.positions[atom, 0] += sign * step
            shifted.append(pyseqm.label_pyseqm_batch(molecules(atoms), QM_CONFIG)[0].get_results()["energy"])
        assert base[atom, 0] == pytest.approx(-(shifted[0] - shifted[1]) / (2 * step), abs=1e-4)


def test_pyseqm_task_accepts_molecule_objects():
    import inspect
    assert list(inspect.signature(pyseqm.pyseqm_task.func).parameters) == ["molecule_objects", "QM_config"]


PROPS = {**{f'sE{i}': [f'e{i}', 'system', 1.] for i in range(4)},
         **{f'F{i}': [f'f{i}', 'atomic', 1.] for i in range(4)},
         'dE01': ['gap01', 'system', 1.]}


def fake(coordinates, species, parameters, device):
    assert parameters['excited_states'] == {'method': 'cis', 'n_states': 3, 'tolerance': 1e-8}
    energy = -species.sum(axis=1)[:, None] + np.arange(4)[None, :]
    force = np.broadcast_to(species[:, None, :, None], (len(species), 4, species.shape[1], 3 )).astype(float).copy()
    return energy, force, np.ones(len(species), bool)


def fail_batch(coordinates, species, parameters, device):
    if len(species) > 1 or np.max(coordinates) > 10:
        raise RuntimeError('failed batch or water')
    return fake(coordinates, species, parameters, device)


def sleep_child(*args):
    time.sleep(20)


def crash_child(*args):
    import os
    os._exit(3)


def test_native_cis_preserves_member_state_atom_order_and_offset():
    permuted = METHANOL[[2, 1, 0, 3, 4, 5]]
    labeled = pyseqm.label_pyseqm_excited_batch(molecules(METHANOL, permuted),
        {'energy_offset_eV': 2.}, PROPS, runner=fake)
    for mol, atoms in zip(labeled, (METHANOL, permuted)):
        assert mol.check_convergence()
        assert mol.get_results()['sE0'] == -sum(atoms.numbers)-2
        assert mol.get_results()['dE01'] == 1
        for state in range(4):
            np.testing.assert_array_equal(mol.get_results()[f'F{state}'][:, 0], atoms.numbers)
        assert mol.get_metadata()['qm_backend_width'] == 2
        assert mol.get_metadata()['qm_solve_widths'] == [2]


def test_failed_native_solve_retries_members_and_keeps_successes():
    bad = METHANOL.copy(); bad.positions[0, 0] = 20.
    output = pyseqm.label_pyseqm_excited_batch(molecules(METHANOL, bad), {}, PROPS, runner=fail_batch)
    assert [m.check_convergence() for m in output] == [True, False]
    assert output[0].get_metadata()['qm_backend_width'] == 1
    assert output[0].get_metadata()['qm_solve_widths'] == [2, 1, 1]
    assert output[0].get_metadata()['qm_retry_count'] == 2
    assert output[0].get_metadata()['qm_failed_ids'] == ['mol-1']
    assert output[1].get_results() == {}


@pytest.mark.parametrize('problem', ['nonfinite', 'unconverged', 'shape'])
def test_invalid_member_or_batch_cannot_pass(problem):
    def runner(*args):
        energy, force, ok = fake(*args)
        if problem == 'nonfinite': force[1, 2, 0, 0] = np.nan
        if problem == 'unconverged': ok[1] = False
        if problem == 'shape': energy = energy[:, :-1]
        return energy, force, ok
    out = pyseqm.label_pyseqm_excited_batch(molecules(METHANOL, METHANOL), {}, PROPS, runner=runner)
    assert not out[1].check_convergence()
    if problem != 'shape': assert out[0].check_convergence()


def test_periodic_batch_member_does_not_lose_good_member():
    a = WATER.copy(); a.pbc = True
    out = pyseqm.label_pyseqm_excited_batch(molecules(METHANOL, a), {}, PROPS, runner=fake)
    assert [m.check_convergence() for m in out] == [True, False]


@pytest.mark.parametrize('config', [{'method': 'PM3'}, {'timeout_seconds': 0},
                                  {'charge': 1}, {'cis_tol': float('nan')}])
def test_unsupported_science_and_bad_settings_fail_explicitly(config):
    out = pyseqm.label_pyseqm_excited_batch(molecules(WATER), config, PROPS, runner=fake)
    assert not out[0].check_convergence()
    assert out[0].get_metadata()['qm_error']


@pytest.mark.parametrize('runner, match', [(sleep_child, 'exceeded'), (crash_child, 'without results')])
def test_timeout_and_crash_reap_owned_children(runner, match):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises((TimeoutError, RuntimeError), match=match):
        pyseqm.run_pyseqm_cis_isolated(np.zeros((1, 3, 3)), np.ones((1, 3)), {}, 'cpu',
                                    timeout=.15 if runner is sleep_child else 10, runner=runner)
    assert {p.pid for p in mp.active_children()} == before


def test_child_success_transfers_results_and_reaps():
    params = {'excited_states': {'method': 'cis', 'n_states': 3, 'tolerance': 1e-8}}
    xyz, numbers, _ = pyseqm.pack_molecules(molecules(WATER))
    energy, force, ok = pyseqm.run_pyseqm_cis_isolated(xyz, numbers, params, 'cpu', 10, fake)
    assert energy.shape == (1, 4)
    assert force.shape == (1, 4, 3, 3)
    assert ok.tolist() == [True]


def test_both_task_signatures_match_labeling_coordinator():
    import inspect
    assert 'molecule_object' in inspect.signature(pyseqm.pyseqm_excited_state_task.func).parameters
    assert 'molecule_objects' in inspect.signature(pyseqm.pyseqm_excited_state_batch_task.func).parameters


def test_mixed_compositions_partition_native_solves_and_restore_submission_order():
    out = pyseqm.label_pyseqm_excited_batch(molecules(WATER, METHANOL, WATER, METHANOL), {}, PROPS, runner=fake)
    assert [m.get_moleculeid() for m in out] == ['mol-0', 'mol-1', 'mol-2', 'mol-3']
    assert all(m.check_convergence() and m.get_metadata()['qm_backend_width'] == 2 for m in out)
    assert out[0].get_metadata()['qm_solve_widths'] == [2, 2]
    assert out[0].get_metadata()['qm_retry_count'] == 0
    assert out[0].get_metadata()['qm_compatibility_groups'] == [['mol-0', 'mol-2'], ['mol-1', 'mol-3']]
