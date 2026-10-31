import numpy as np
import pytest
from ase import Atoms

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
