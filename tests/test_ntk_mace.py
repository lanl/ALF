"""NTK uncertainty on a small random MACE model, molecular and periodic."""

import copy

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

torch = pytest.importorskip("torch")
pytest.importorskip("mace")
# MACE allowlists e3nn's constants for torch.load, so import it before e3nn.
from mace import modules  # noqa: E402
from e3nn import o3  # noqa: E402

from alframework.ml_interfaces.ntk import NTKEstimator, NTKPackedEvaluator, RowRequest, compute_rows
from alframework.ml_interfaces.ntk.adapter import gaussian_probes
from alframework.ml_interfaces.ntk.mace import MACENTKAdapter

SPECIES = [1, 6, 8, 14]


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
        return modules.ScaleShiftMACE(
            r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2, interaction_cls=block,
            interaction_cls_first=block, num_interactions=2, num_elements=len(SPECIES),
            hidden_irreps=o3.Irreps("4x0e + 4x1o"), MLP_irreps=o3.Irreps("4x0e"), gate=torch.nn.functional.silu,
            atomic_energies=np.zeros(len(SPECIES)), avg_num_neighbors=6.0, atomic_numbers=SPECIES, correlation=2,
            radial_type="bessel", atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        ).double().eval()


def water():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])


def silicon(seed):
    cell = bulk("Si", "diamond", a=5.43)
    cell.rattle(0.05, seed=seed)
    return cell


@pytest.mark.parametrize("periodic", [False, True])
def test_mace_rows_match_finite_differences_and_the_loop(model, periodic):
    atoms, training = (silicon(1), [silicon(2)]) if periodic else (water(), [water()])
    adapter = MACENTKAdapter(model, dtype=torch.float64)
    estimator = NTKEstimator(adapter, training, energy_scale=0.1, force_scale=0.5, sketch_dim=8, seed=2)
    structure = estimator.structure_from_atoms(atoms)
    probes = gaussian_probes(1, len(atoms), seed=3, device="cpu", dtype=torch.float64)
    request = RowRequest(forces=True, hvp_probes=probes)
    values, batched = compute_rows(adapter, structure, request, estimator.sketch)
    adapter._ntk_batched_reverse = False
    _, looped = compute_rows(adapter, structure, request, estimator.sketch)
    for key in batched:
        torch.testing.assert_close(looped[key], batched[key], rtol=1e-9, atol=1e-11)

    direction = estimator.sketch.tangents(0)

    def predict(step):
        shifted = copy.deepcopy(adapter)
        with torch.no_grad():
            for name, parameter in shifted.live_parameters().items():
                parameter.add_(step * direction[name][0])
        x = structure.positions.clone().requires_grad_(True)
        energy = shifted.energies(structure, x)[0]
        gradient, = torch.autograd.grad(energy, x, create_graph=True)
        hvp, = torch.autograd.grad((gradient * probes[0]).sum(), x)
        return energy.detach(), -gradient.detach(), hvp

    plus, minus = predict(1e-5), predict(-1e-5)
    for key, (a, b) in zip(("energy", "forces", "hvp"), zip(plus, minus)):
        row = batched[key][0, 0, ..., 0] if key == "hvp" else batched[key][0, ..., 0]
        torch.testing.assert_close(row, (a - b) / 2e-5, atol=1e-6, rtol=1e-4)
    evaluator = NTKPackedEvaluator(estimator)
    energies, forces = evaluator.evaluate(torch.tensor(atoms.numbers), torch.tensor(atoms.positions),
                                          torch.tensor(np.asarray(atoms.cell))[None], torch.tensor(atoms.pbc),
                                          [len(atoms)])
    torch.testing.assert_close(energies[0], values["energy"])
    torch.testing.assert_close(forces[0], values["forces"][0])


def test_mace_checkpoint_basis_round_trip(model, tmp_path):
    checkpoint = tmp_path / "mace.model"
    torch.save(model, checkpoint)
    adapter = MACENTKAdapter.from_checkpoint(checkpoint, dtype=torch.float64)
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, sketch_dim=8)
    path = tmp_path / "basis.npz"
    estimator.save_basis(path)
    restored = NTKEstimator.from_basis(MACENTKAdapter(model, dtype=torch.float64), path)
    query = water()
    query.positions[1] *= 1.1
    torch.testing.assert_close(restored.predict(query, forces=False)["energy_uncertainty_score"],
                               estimator.predict(query, forces=False)["energy_uncertainty_score"])
    with pytest.raises(ValueError, match="no species"):
        estimator.predict(Atoms("Na", positions=[[0, 0, 0]]))
