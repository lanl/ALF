"""Native CIS: state/atom mappings, failure isolation and owned child lifetime."""
import multiprocessing as mp
import time
import numpy as np
import pytest

from alframework.qm_interfaces import pyseqm_interface as seqm
from tests.test_pyseqm_interface import WATER, METHANOL, molecules

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
    labeled = seqm.label_pyseqm_excited_batch(molecules(METHANOL, permuted),
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
    output = seqm.label_pyseqm_excited_batch(molecules(METHANOL, bad), {}, PROPS, runner=fail_batch)
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
    out = seqm.label_pyseqm_excited_batch(molecules(METHANOL, METHANOL), {}, PROPS, runner=runner)
    assert not out[1].check_convergence()
    if problem != 'shape': assert out[0].check_convergence()


def test_periodic_batch_member_does_not_lose_good_member():
    a = WATER.copy(); a.pbc = True
    out = seqm.label_pyseqm_excited_batch(molecules(METHANOL, a), {}, PROPS, runner=fake)
    assert [m.check_convergence() for m in out] == [True, False]


@pytest.mark.parametrize('config', [{'method': 'PM3'}, {'timeout_seconds': 0},
                                  {'charge': 1}, {'cis_tol': float('nan')}])
def test_unsupported_science_and_bad_settings_fail_explicitly(config):
    out = seqm.label_pyseqm_excited_batch(molecules(WATER), config, PROPS, runner=fake)
    assert not out[0].check_convergence()
    assert out[0].get_metadata()['qm_error']


@pytest.mark.parametrize('runner, match', [(sleep_child, 'exceeded'), (crash_child, 'without results')])
def test_timeout_and_crash_reap_owned_children(runner, match):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises((TimeoutError, RuntimeError), match=match):
        seqm.run_pyseqm_cis_isolated(np.zeros((1, 3, 3)), np.ones((1, 3)), {}, 'cpu',
                                    timeout=.15 if runner is sleep_child else 10, runner=runner)
    assert {p.pid for p in mp.active_children()} == before


def test_child_success_transfers_results_and_reaps():
    params = {'excited_states': {'method': 'cis', 'n_states': 3, 'tolerance': 1e-8}}
    xyz, numbers, _ = seqm.pack_molecules(molecules(WATER))
    energy, force, ok = seqm.run_pyseqm_cis_isolated(xyz, numbers, params, 'cpu', 10, fake)
    assert energy.shape == (1, 4)
    assert force.shape == (1, 4, 3, 3)
    assert ok.tolist() == [True]


def test_both_task_signatures_match_labeling_coordinator():
    import inspect
    assert 'molecule_object' in inspect.signature(seqm.pyseqm_excited_state_task.func).parameters
    assert 'molecule_objects' in inspect.signature(seqm.pyseqm_excited_state_batch_task.func).parameters


def test_mixed_compositions_partition_native_solves_and_restore_submission_order():
    out = seqm.label_pyseqm_excited_batch(molecules(WATER, METHANOL, WATER, METHANOL), {}, PROPS, runner=fake)
    assert [m.get_moleculeid() for m in out] == ['mol-0', 'mol-1', 'mol-2', 'mol-3']
    assert all(m.check_convergence() and m.get_metadata()['qm_backend_width'] == 2 for m in out)
    assert out[0].get_metadata()['qm_solve_widths'] == [2, 2]
    assert out[0].get_metadata()['qm_retry_count'] == 0
    assert out[0].get_metadata()['qm_compatibility_groups'] == [['mol-0', 'mol-2'], ['mol-1', 'mol-3']]
