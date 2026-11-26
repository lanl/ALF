"""NTK uncertainty on small random X-MACE excited-state models (github.com/rhyan10/X-MACE)."""

import copy

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")
excited = pytest.importorskip("mace.modules.excited", reason="X-MACE is not installed")
from mace import modules  # noqa: E402  (imports before e3nn; see test_ntk_mace.py)
from e3nn import o3  # noqa: E402

from alframework.ml_interfaces.ntk import NTKASECalculator, NTKEstimator, NTKPackedEvaluator, RowRequest, compute_rows
from alframework.ml_interfaces.ntk.adapter import gaussian_probes
from alframework.ml_interfaces.ntk.xmace import XMACENTKAdapter

SPECIES = [1, 6, 8]


def build(autoencoder):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        cls = excited.AutoencoderExcitedMACE if autoencoder else excited.ExcitedMACE
        extra = {"num_permutational_invariant": 6} if autoencoder else {}
        model = cls(r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
                    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
                    interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
                    num_interactions=2, num_elements=len(SPECIES), hidden_irreps=o3.Irreps("4x0e + 4x1o"),
                    MLP_irreps=o3.Irreps("4x0e"), atomic_energies=np.array([-1.0, -3.0, -5.0]),
                    avg_num_neighbors=3.0, atomic_numbers=SPECIES, correlation=2, gate=torch.nn.functional.silu,
                    n_energies=3, compute_nacs=True, **extra)
        # Readouts start at zero, which puts every state on the backbone surface. Separate the states.
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for readout in model.readouts:
                for parameter in readout.parameters():
                    parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
        return model.double().eval()


@pytest.fixture(params=[False, True], ids=["ExcitedMACE", "AutoencoderExcitedMACE"])
def model(request):
    return build(request.param)


def water(stretch=1.0):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])
    atoms.positions[1] *= stretch
    return atoms


def outputs_for(adapter):
    return ("ground_energy", "state_offsets") + (("latent", "state_matrix") if adapter.autoencoder else ())


def test_xmace_rows_match_finite_differences_and_the_loop(model):
    adapter = XMACENTKAdapter(model, dtype=torch.float64)
    assert adapter.state_names == ("E0", "E1", "E2")
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, force_scale=0.5, sketch_dim=6, seed=2)
    structure = estimator.structure_from_atoms(water(1.1))
    probes = gaussian_probes(1, 3, seed=3, device="cpu", dtype=torch.float64)
    names = outputs_for(adapter)
    request = RowRequest(forces=True, hvp_probes=probes, outputs=names)
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
        energy = shifted.energies(structure, x)
        gradients = torch.stack([torch.autograd.grad(e, x, create_graph=True)[0] for e in energy])
        hvp = torch.stack([torch.autograd.grad((g * probes[0]).sum(), x, retain_graph=True)[0] for g in gradients])
        smooth = shifted.outputs(structure, structure.positions, None, names)
        return energy.detach(), -gradients.detach(), hvp, smooth

    (e1, f1, h1, s1), (e2, f2, h2, s2) = predict(1e-5), predict(-1e-5)
    torch.testing.assert_close(batched["energy"][..., 0], (e1 - e2) / 2e-5, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(batched["forces"][..., 0], (f1 - f2) / 2e-5, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(batched["hvp"][:, 0, ..., 0], (h1 - h2) / 2e-5, atol=1e-5, rtol=1e-4)
    for name in names:
        torch.testing.assert_close(batched[f"output:{name}"][..., 0], (s1[name] - s2[name]) / 2e-5,
                                   atol=1e-6, rtol=1e-4)
    # Gap rows are differences of state rows.
    result = estimator.score(structure, forces=False)
    torch.testing.assert_close(result["gap_uncertainty_score"],
                               estimator.posterior.variance(batched["energy"][1:] - batched["energy"][:-1]).sqrt())


def test_autoencoder_state_matrix_diagonalizes_to_the_state_energies():
    adapter = XMACENTKAdapter(build(True), dtype=torch.float64)
    structure = adapter.structure(water(1.07).numbers, water(1.07).positions)
    elements = adapter.outputs(structure, structure.positions, None, ("state_matrix",))["state_matrix"]
    matrix = torch.diag(elements[:3]) + torch.diag(elements[3:], 1) + torch.diag(elements[3:], -1)
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix), adapter.energies(structure, structure.positions))
    with pytest.raises(ValueError, match="Autoencoder"):
        XMACENTKAdapter(build(False)).outputs(structure, structure.positions, None, ("latent",))


def test_frozen_backbone_is_left_out_of_the_posterior(model):
    frozen = copy.deepcopy(model)
    for parameter in frozen.backbone.parameters():
        parameter.requires_grad_(False)
    adapter = XMACENTKAdapter(frozen, dtype=torch.float64)
    trained = {name for name, p in frozen.named_parameters() if p.requires_grad}
    assert set(adapter.live_parameters()) == trained and not any(n.startswith("backbone.") for n in trained)
    everything = XMACENTKAdapter(frozen, dtype=torch.float64, parameters="all")
    assert len(everything.live_parameters()) == len(list(frozen.parameters()))
    assert adapter.digest() != everything.digest()
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, sketch_dim=4)
    assert estimator.sketch.total == sum(p.numel() for p in adapter.live_parameters().values())


def test_predictions_match_the_xmace_calculator_and_report_state_scores(model):
    from mace.calculators.excited import ExcitedMACECalculator

    adapter = XMACENTKAdapter(model, dtype=torch.float64)
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, force_scale=0.5, sketch_dim=6)
    query = water(1.1)
    reference = ExcitedMACECalculator(None, models=[copy.deepcopy(model)], default_dtype="float64", state=1)
    reference.calculate(query)
    energies, forces = NTKPackedEvaluator(estimator).estimator.adapter.predict_batch(
        torch.tensor(query.numbers), torch.tensor(query.positions), None, torch.tensor(query.pbc), [3])
    np.testing.assert_allclose(energies[:, 0].numpy(), reference.results["energies"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(forces.permute(1, 0, 2).numpy(), reference.results["forces_states"], atol=1e-8)

    calculator = NTKASECalculator(estimator, selected_state=1, hessian_probes=2,
                                  outputs=outputs_for(adapter), crossing_gap_eV=1e3)
    calculator.calculate(query)
    scores = calculator.results["alf_uncertainty"]
    assert calculator.results["energy"] == pytest.approx(reference.results["energy"])
    assert {"ntk_hessian_E1", "ntk_dE12", "ntk_state_offsets_max", "ntk_near_crossing_E1"} <= set(scores)
    # With a crossing window wider than every gap, each state is flagged.
    assert all(scores[f"ntk_near_crossing_{state}"] == 1.0 for state in adapter.state_names)
    calculator = NTKASECalculator(estimator, crossing_gap_eV=0.0)
    calculator.calculate(query)
    assert not any(calculator.results["alf_uncertainty"][f"ntk_near_crossing_{s}"] for s in adapter.state_names)
