"""NTK for HIPPYNN graphs scored on outputs other than energies, such as a coupling model on a frozen energy model."""

import copy

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")

from hippynn import custom_kernels
from hippynn.graphs import GraphModule, IdxType, inputs, networks, targets

from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import HippynnNTKAdapter, HippynnOutputsNTKAdapter, frozen_model_inputs
from alframework.ml_interfaces.ntk import NTKEstimator, RowRequest, compute_rows
from alframework.ml_interfaces.ntk.couplings import physical_coupling_uncertainty

NETWORK = {"possible_species": [0, 1, 8], "n_features": 6, "n_sensitivities": 4, "dist_soft_min": 0.5,
           "dist_soft_max": 2.5, "dist_hard_max": 3.0, "n_interaction_layers": 1, "n_atom_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels(False)
    yield
    custom_kernels.set_custom_kernels(previous)


def water(stretch=1.0):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])
    atoms.positions[1] *= stretch
    return atoms


def model_a(n_states=3):
    """A multistate energy model standing in for a trained Model A."""
    with torch.random.fork_rng():
        torch.manual_seed(5)
        species, positions = inputs.SpeciesNode("species"), inputs.PositionsNode("positions")
        net = networks.Hipnn("hipnn", (species, positions), module_kwargs=NETWORK)
        energies = [system_energy(targets.HEnergyNode(f"E{i}", net)) for i in range(n_states)]
        return GraphModule((species, positions), tuple(energies)).double().eval()


def toy_coupling_graph():
    """species, positions, and a per-system gap input; one output: an energy-like readout times the gap."""
    with torch.random.fork_rng():
        torch.manual_seed(9)
        species, positions = inputs.SpeciesNode("species"), inputs.PositionsNode("positions")
        net = networks.Hipnn("hipnn", (species, positions), module_kwargs=NETWORK)
        gap = inputs.InputNode("GAP12", db_name="GAP12", index_state=IdxType.Systems)
        out = system_energy(targets.HEnergyNode("readout", net)) * gap
        out.db_name = "coupling12"
        return GraphModule((species, positions, gap), (out,)).double().eval()


def test_outputs_adapter_fits_scores_and_matches_finite_differences(tmp_path):
    gap = {"GAP12": torch.tensor([[0.7]], dtype=torch.float64)}
    adapter = HippynnOutputsNTKAdapter(toy_coupling_graph(), extra_inputs=lambda structure: gap,
                                       supports_forward=True)
    assert adapter.output_names == ("coupling12",) and adapter.extra_names == ("GAP12",)
    estimator = NTKEstimator(adapter, [water(), water(1.08)], output_scales={"coupling12": 0.05}, sketch_dim=8)
    assert dict(estimator.posterior.counts) == {"output:coupling12": 2} and not estimator.state_names
    result = estimator.predict(water(1.15), outputs=("coupling12",))
    assert set(result) == {"output_coupling12", "output_coupling12_uncertainty_score"}
    structure = estimator.structure_from_atoms(water(1.15))
    request = RowRequest(energy=False, outputs=("coupling12",))
    _, forward = compute_rows(adapter, structure, request, estimator.sketch, engine="forward")
    values, reverse = compute_rows(adapter, structure, request, estimator.sketch, engine="reverse")
    torch.testing.assert_close(forward["output:coupling12"], reverse["output:coupling12"], rtol=1e-9, atol=1e-12)
    direction = estimator.sketch.tangents(0)

    def shifted(step):
        moved = copy.deepcopy(adapter)
        with torch.no_grad():
            for name, parameter in moved.live_parameters().items():
                parameter.add_(step * direction[name][0])
        return moved.outputs(structure, structure.positions, None, ("coupling12",))["coupling12"]

    torch.testing.assert_close(reverse["output:coupling12"][..., 0], (shifted(1e-5) - shifted(-1e-5)) / 2e-5,
                               atol=1e-7, rtol=1e-5)
    path = tmp_path / "basis.npz"
    estimator.save_basis(path)
    restored = NTKEstimator.from_basis(adapter, path)
    torch.testing.assert_close(restored.predict(water(1.15), outputs=("coupling12",))["output_coupling12_uncertainty_score"],
                               result["output_coupling12_uncertainty_score"])
    with pytest.raises(ValueError, match="energies"):
        estimator.predict(water(), hessian_probes=2)
    with pytest.raises(ValueError, match="extra_inputs"):
        HippynnOutputsNTKAdapter(toy_coupling_graph())


def test_frozen_model_supplies_gradient_differences_and_gaps():
    a = HippynnNTKAdapter(model_a(), dtype=torch.float64)
    supply = frozen_model_inputs(a, {"GDV12": ("GDV", 1, 2), "GAP12": ("GAP", 1, 2)})
    structure = a.structure(water().numbers, water().positions)
    values = supply(structure)
    energies, forces = a.predict_batch(structure.numbers, structure.positions, None, structure.pbc, [3])
    torch.testing.assert_close(values["GAP12"], (energies[2] - energies[1]).reshape(1))
    torch.testing.assert_close(values["GDV12"], (forces[1] - forces[2]).reshape(1, 9))


def test_physical_coupling_uncertainty_propagates_both_terms():
    scaled, sigma_s = torch.tensor([0.2, -0.4]), torch.tensor([0.01, 0.02])
    gap, sigma_gap = torch.tensor(0.05), torch.tensor(0.005)
    coupling, sigma = physical_coupling_uncertainty(scaled, sigma_s, gap, sigma_gap)
    torch.testing.assert_close(coupling, scaled / gap)
    torch.testing.assert_close(sigma, torch.sqrt((sigma_s / gap) ** 2 + (scaled * sigma_gap / gap ** 2) ** 2))
    # Closing the gap tenfold raises the coupling uncertainty far more than tenfold once the gap term dominates.
    _, closer = physical_coupling_uncertainty(scaled, sigma_s, gap / 10, sigma_gap)
    assert bool((closer > 10 * sigma).all())
    with pytest.raises(ValueError, match="zero gap"):
        physical_coupling_uncertainty(scaled, sigma_s, 0.0)
