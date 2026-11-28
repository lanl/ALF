"""NTK uncertainty: the core against closed forms, and its adapters for HIPPYNN, MACE, X-MACE and UMA.

The core runs on pair potentials that are linear in their weights, so the posterior is Bayesian linear regression.
Each adapter section builds small random models of its package and skips when the package is missing: HIPPYNN,
mace-torch, or X-MACE (github.com/rhyan10/X-MACE, which installs as ``mace``). Rows are checked against finite
differences in the model's parameters, which tests ALF's Jacobians, not the model's physics.
"""

import copy
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

torch = pytest.importorskip("torch")

from alframework.ml_interfaces.ntk import (NTKASECalculator, NTKEstimator, NTKPackedEvaluator, RowRequest, Sketch,
                                           compute_rows)
from alframework.ml_interfaces.ntk import estimator as estimator_module
from alframework.ml_interfaces.ntk.adapter import NTKAdapter, gaussian_probes
from alframework.ml_interfaces.ntk.couplings import physical_coupling_uncertainty
from alframework.samplers.adaptive_ntk import initial_energy_score
from alframework.samplers.engines.packed import PackedMLMDEngine
from alframework.testing import check_evaluator
from tests.helpers.engines import require_engine

DEVICES = list(dict.fromkeys(("cpu", os.environ.get("ALF_TEST_NTK_DEVICE", "cpu"))))


def water(stretch=1.0):
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])
    atoms.positions[1] *= stretch
    return atoms


def silicon(seed=1, cubic=True):
    cell = bulk("Si", "diamond", a=5.43, cubic=cubic)
    cell.rattle(0.05, seed=seed)
    return cell


def parameter_step(adapter, sketch, step):
    """A copy of ``adapter`` moved by ``step`` along the sketch's first direction."""
    moved = copy.deepcopy(adapter)
    direction = sketch.tangents(0)
    with torch.no_grad():
        for name, parameter in moved.live_parameters().items():
            parameter.add_(step * direction[name][0])
    return moved


# --- Core -------------------------------------------------------------------------------------------
# A pair potential linear in its weights, whose NTK posterior has a closed form.

WIDTHS = torch.tensor([0.4, 0.8, 1.3, 2.0, 3.1], dtype=torch.float64)


class LinearPairAdapter(NTKAdapter):
    """E = Σ_k θ_k Σ_{i<j} exp(-r_ij² / w_k²): linear in θ, so its NTK posterior is Bayesian linear regression."""

    supports_forward = True

    def __init__(self, theta=None):
        self.module = torch.nn.Module()
        self.module.theta = torch.nn.Parameter(
            torch.tensor([0.3, -1.1, 0.7, 0.2, -0.4], dtype=torch.float64) if theta is None else theta)

    def live_parameters(self):
        return {"theta": self.module.theta}

    def digest(self):
        return "linear-pair"

    @staticmethod
    def basis(x):
        i, j = torch.triu_indices(len(x), len(x), 1)
        r2 = (x[i] - x[j]).square().sum(-1)
        return torch.exp(-r2[:, None] / WIDTHS.square()).sum(0)

    def energies(self, structure, positions, params=None):
        theta = self.module.theta if params is None else params["theta"]
        return (theta * self.basis(positions)).sum()[None]


def molecule(seed, n=3):
    rng = np.random.default_rng(seed)
    return Atoms(numbers=[1] * n, positions=rng.normal(scale=0.9, size=(n, 3)))


def features(atoms):
    """Exact energy features, force features ``[N, 3, K]`` and Hessian features ``[3N, 3N, K]``."""
    x = torch.tensor(atoms.positions, dtype=torch.float64)
    f = LinearPairAdapter.basis(x)
    forces = -torch.autograd.functional.jacobian(LinearPairAdapter.basis, x).permute(1, 2, 0)
    hessian = torch.stack([torch.autograd.functional.hessian(lambda y: LinearPairAdapter.basis(y)[k], x)
                           .reshape(3 * len(atoms), 3 * len(atoms)) for k in range(len(WIDTHS))], -1)
    return f, forces, hessian


def sketch_matrix(sketch):
    return torch.cat([sketch.flat_chunk(c) for c in range(sketch.n_chunks)])


def linear_pair_estimator(**options):
    return NTKEstimator(LinearPairAdapter(), [molecule(1)], **{**dict(energy_scale=0.3, force_scale=0.7), **options})


@pytest.mark.parametrize("engine", ["forward", "reverse"])
def test_ntk_scores_equal_bayesian_linear_regression(engine):
    training, query = [molecule(1), molecule(2)], molecule(3)
    scales = dict(energy_scale=0.3, force_scale=0.7)
    estimator = NTKEstimator(LinearPairAdapter(), training, **scales, sketch_dim=8, seed=4, ridge=0.05, engine=engine)
    S = sketch_matrix(estimator.sketch)
    information = torch.zeros(8, 8, dtype=torch.float64)
    for atoms in training:
        f, g, _ = features(atoms)
        rows = torch.cat([(S @ f)[None] / scales["energy_scale"], (g.reshape(-1, 5) @ S.T) / scales["force_scale"]])
        information += rows.T @ rows
    precision = information + 0.05 * torch.eye(8, dtype=torch.float64)
    f, g, _ = features(query)

    def expected(rows):
        return 0.05 * (rows * torch.linalg.solve(precision, rows.T).T).sum(-1)

    result = estimator.predict(query, forces=True)
    torch.testing.assert_close(result["energy_uncertainty_score"][0], expected((S @ f)[None])[0].sqrt())
    torch.testing.assert_close(result["force_component_uncertainty_score"][0],
                               expected(g.reshape(-1, 5) @ S.T).reshape(3, 3).sqrt())


@pytest.mark.parametrize("engine", ["forward", "reverse"])
def test_ntk_hvp_rows_are_hessian_columns(engine):
    estimator = linear_pair_estimator(sketch_dim=8, seed=4, engine=engine)
    query = molecule(3)
    _, _, h = features(query)
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(query),
                           RowRequest(hvp_probes=torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)),
                           estimator.sketch, engine=engine)
    # Row k of the identity probes is Hessian column k.
    torch.testing.assert_close(rows["hvp"][0].reshape(9, 9, 8), (h @ sketch_matrix(estimator.sketch).T).transpose(0, 1))


@pytest.mark.parametrize("engine", ["forward", "reverse"])
def test_ntk_selected_force_rows_match_full_scores(engine, monkeypatch):
    estimator = linear_pair_estimator(sketch_dim=8, engine=engine)
    structure = estimator.structure_from_atoms(molecule(5))
    full = estimator.score(structure)
    requests = []

    def record_rows(adapter, structure, request, *args, **kwargs):
        values, rows = compute_rows(adapter, structure, request, *args, **kwargs)
        requests.append((request.output_count(1, structure.n_atoms), rows["forces"].shape))
        return values, rows

    monkeypatch.setattr("alframework.ml_interfaces.ntk.estimator.compute_rows", record_rows)
    selected = estimator.score_forces(structure, atom_indices=[2, 0], include_energy=True)
    # One energy row and the two selected atoms' six force rows, not all nine.
    assert requests == [(7, (1, 2, 3, 8))]
    torch.testing.assert_close(selected["energy_uncertainty_score"], full["energy_uncertainty_score"])
    torch.testing.assert_close(selected["force_uncertainty_score"], full["force_uncertainty_score"][:, [2, 0]])
    assert selected["force_rms_uncertainty"] == pytest.approx(
        full["force_component_uncertainty_score"][:, [2, 0]].square().mean().sqrt().item())


def test_ntk_selected_forces_refuse_probes():
    estimator = linear_pair_estimator(sketch_dim=8)
    with pytest.raises(ValueError, match="requires exact"):
        estimator.score_forces(estimator.structure_from_atoms(molecule(5)), atom_indices=[0], force_probes=8)


def test_ntk_reverse_rows_fall_back_to_one_gradient_per_output_when_batching_fails(monkeypatch):
    """Some models' backward formulas cannot be batched. The rows must not change when ALF falls back."""
    estimator = linear_pair_estimator(sketch_dim=8)
    structure = estimator.structure_from_atoms(molecule(5))
    request = RowRequest(forces=True, hvp_probes=gaussian_probes(2, 3, seed=6, device="cpu", dtype=torch.float64))
    _, batched = compute_rows(estimator.adapter, structure, request, estimator.sketch, engine="reverse")
    grad = torch.autograd.grad
    unbatched_calls = []

    def unbatchable(*args, is_grads_batched=False, **kwargs):
        if is_grads_batched:
            raise RuntimeError("this backward formula cannot be batched")
        unbatched_calls.append(1)
        return grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", unbatchable)
    _, looped = compute_rows(LinearPairAdapter(), structure, request, estimator.sketch, engine="reverse")
    assert unbatched_calls
    for key in batched:
        torch.testing.assert_close(looped[key], batched[key], rtol=1e-12, atol=1e-14)


def test_ntk_probe_estimates_match_exact_uncertainty():
    estimator = linear_pair_estimator(sketch_dim=5, seed=4, ridge=0.05)
    query = molecule(5)
    exact = estimator.predict(query, forces=True)
    structure = estimator.structure_from_atoms(query)
    identity = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    _, rows = compute_rows(estimator.adapter, structure, RowRequest(hvp_probes=identity), estimator.sketch)
    exact_hessian = (estimator.posterior.variance(rows["hvp"][0]).sum() ** 0.5) / 9
    # The probes are seeded, so this is not a convergence check: 200 probes land within about 6%, and the 30%
    # tolerance still catches a wrong normalization (a missing square root, or N in place of 3N).
    sampled = estimator.predict(query, forces=False, force_probes=200, hessian_probes=200)
    assert sampled["force_rms_uncertainty"][0] == pytest.approx(float(exact["force_rms_uncertainty"][0]), rel=0.3)
    assert sampled["hessian_rms_uncertainty"][0] == pytest.approx(float(exact_hessian), rel=0.3)


def test_ntk_training_geometries_have_small_variance():
    training = [molecule(seed) for seed in range(4)]
    estimator = NTKEstimator(LinearPairAdapter(), training, energy_scale=1.0, sketch_dim=5, seed=1, ridge=1e-9)
    # Five features, four energies: the span misses one direction, but training energies lie in it.
    for atoms in training:
        assert estimator.predict(atoms, forces=False)["energy_uncertainty_score"][0] < 1e-3


def test_ntk_posterior_with_ridge_shares_its_information():
    estimator = NTKEstimator(LinearPairAdapter(), [molecule(0)], energy_scale=1.0, sketch_dim=5, ridge=1e-9)
    loose = estimator.posterior.with_ridge(1.0)
    assert loose.ridge == 1.0 and loose.information is estimator.posterior.information


def test_ntk_sketch_regenerates_its_stored_chunks():
    shapes = {"a": (7, 3), "b": (5,)}
    stored = Sketch(shapes, 10, 3, chunk_size=4, store=True)
    regenerated = Sketch(shapes, 10, 3, chunk_size=4, store=False)
    assert stored.n_chunks == 3 and stored.digest == regenerated.digest
    full = sketch_matrix(stored)
    assert full.shape == (10, 26)
    torch.testing.assert_close(sketch_matrix(regenerated), full)
    torch.testing.assert_close(torch.cat([stored.tangents(c)["a"].reshape(-1, 21) for c in range(3)]), full[:, :21])
    assert Sketch(shapes, 10, 4, chunk_size=4).digest != stored.digest
    gradients = torch.randn(3, 26, dtype=torch.float64)
    torch.testing.assert_close(stored.project(gradients), gradients @ full.T)


def test_ntk_sketch_entries_have_variance_one_over_p():
    large = sketch_matrix(Sketch({"w": (20000,)}, 16, 0))
    assert float(large.var() * 16) == pytest.approx(1.0, rel=0.02)


def test_ntk_probes_are_seeded_unit_gaussians():
    probes = gaussian_probes(5000, 4, seed=3, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(probes, gaussian_probes(5000, 4, seed=3, device="cpu", dtype=torch.float64))
    second_moment = torch.einsum("ki,kj->ij", probes.reshape(5000, 12), probes.reshape(5000, 12)) / 5000
    torch.testing.assert_close(second_moment, torch.eye(12, dtype=torch.float64), atol=0.08, rtol=0)


def test_ntk_calculator_reports_probe_scores():
    calculator = NTKASECalculator(linear_pair_estimator(sketch_dim=5), forces=False, force_probes=3,
                                  hessian_probes=3, hvp_probes=1)
    calculator.calculate(molecule(2))
    assert set(calculator.results["alf_uncertainty"]) == {"ntk_energy", "ntk_force_rms", "ntk_hessian"}
    assert set(calculator.results["alf_atomic_uncertainty"]) == {"ntk_hvp"}


def test_ntk_basis_restores_its_id(tmp_path):
    estimator = linear_pair_estimator(sketch_dim=5)
    estimator.save_basis(tmp_path / "basis.npz")
    assert NTKEstimator.from_basis(LinearPairAdapter(), tmp_path / "basis.npz").basis_id == estimator.basis_id


def test_ntk_basis_refuses_a_changed_model(tmp_path):
    linear_pair_estimator(sketch_dim=5).save_basis(tmp_path / "basis.npz")
    changed = LinearPairAdapter()
    changed.digest = lambda: "other"
    with pytest.raises(ValueError, match="model snapshot"):
        NTKEstimator.from_basis(changed, tmp_path / "basis.npz")


@pytest.mark.parametrize("with_outputs", [False, True], ids=["previous_format", "empty_output_scales"])
def test_ntk_version_two_basis_loads_unchanged(tmp_path, with_outputs):
    fitted = linear_pair_estimator(sketch_dim=5)
    metadata = fitted.metadata
    if with_outputs:
        metadata["scales"]["outputs"] = {}
    else:
        metadata["scales"].pop("outputs", None)
    basis_id = fitted.posterior.basis_id(metadata)
    path = tmp_path / "basis.npz"
    np.savez(path, metadata=json.dumps(metadata), basis_id=basis_id, information=fitted.posterior.information.numpy())

    restored = NTKEstimator.from_basis(LinearPairAdapter(), path)
    assert restored.metadata == metadata and restored.basis_id == basis_id
    torch.testing.assert_close(restored.predict(molecule(2))["force_uncertainty_score"],
                               fitted.predict(molecule(2))["force_uncertainty_score"])


class TwoStateAdapter(NTKAdapter):
    """Two diabatic surfaces a = θ₀ exp(-r²), b = θ₁ exp(-r²/4) of one bond, coupled by a constant c.

    The adiabatic energies (a + b)/2 ∓ √(((a − b)/2)² + c²) cross where a = b once c → 0. Their curvature
    there grows like 1/c, while the diabatic elements stay smooth.
    """

    supports_forward = True
    state_names = ("E0", "E1")

    def __init__(self, coupling, crossing=1.2):
        self.module = torch.nn.Module()
        theta = torch.tensor([1.0, np.exp(-crossing ** 2) / np.exp(-crossing ** 2 / 4)], dtype=torch.float64)
        self.module.theta = torch.nn.Parameter(theta)
        self.coupling = coupling

    def live_parameters(self):
        return {"theta": self.module.theta}

    def digest(self):
        return f"two-state-{self.coupling}"

    def _diabatic(self, positions, params):
        theta = self.module.theta if params is None else params["theta"]
        r2 = (positions[1] - positions[0]).square().sum()
        return theta[0] * torch.exp(-r2), theta[1] * torch.exp(-r2 / 4)

    def energies(self, structure, positions, params=None):
        a, b = self._diabatic(positions, params)
        root = torch.sqrt(((a - b) / 2).square() + self.coupling ** 2)
        return torch.stack([(a + b) / 2 - root, (a + b) / 2 + root])

    def outputs(self, structure, positions, params=None, names=()):
        a, b = self._diabatic(positions, params)
        return {"diabatic": torch.stack([a, b])}


def bond(length):
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [length, 0.0, 0.0]])


COUPLINGS = (1e-2, 1e-3, 1e-4)


@pytest.fixture(scope="module")
def at_the_crossing():
    """Scores at the crossing geometry for each coupling. They are small next to the training geometries' gaps,
    so the fit is the same for all of them."""
    results = []
    for coupling in COUPLINGS:
        estimator = NTKEstimator(TwoStateAdapter(coupling), [bond(0.7), bond(1.8)], energy_scale=0.1, sketch_dim=2,
                                 ridge=1e-3)
        results.append(estimator.predict(bond(1.2), forces=False, hessian_probes=64, outputs=("diabatic",)))
    return results


def test_ntk_hessian_scores_diverge_at_a_crossing(at_the_crossing):
    for coupling, result in zip(COUPLINGS, at_the_crossing):
        assert result["gap"][0] == pytest.approx(2 * coupling, rel=1e-6)
    hessian = [float(result["hessian_rms_uncertainty"].max()) for result in at_the_crossing]
    assert hessian[1] > 5 * hessian[0] and hessian[2] > 5 * hessian[1]


def test_ntk_diabatic_scores_stay_smooth_at_a_crossing(at_the_crossing):
    smooth = [result["output_diabatic_uncertainty_score"] for result in at_the_crossing]
    for later in smooth[1:]:
        torch.testing.assert_close(later, smooth[0], rtol=1e-2, atol=0)
    assert all(np.isfinite(float(result["gap_uncertainty_score"][0])) for result in at_the_crossing)


def test_ntk_crossing_flags_mark_states_next_to_a_small_gap():
    from alframework.ml_interfaces.ntk import reported_uncertainty

    result = {"energy_uncertainty_score": torch.tensor([0.1, 0.2, 0.3]), "gap": torch.tensor([0.05, 0.5]),
              "gap_uncertainty_score": torch.tensor([0.01, 0.02])}
    scalar, _ = reported_uncertainty(result, ("E0", "E1", "E2"), 3, crossing_gap_eV=0.1)
    assert [scalar[f"ntk_near_crossing_E{i}"] for i in range(3)] == [1.0, 1.0, 0.0]
    scalar, _ = reported_uncertainty(result, ("E0", "E1", "E2"), 3, crossing_gap_eV=None)
    assert not any(key.startswith("ntk_near_crossing") for key in scalar)
    nonadjacent = dict(result, energy=torch.tensor([0.0, 0.05, 0.55]), gap=torch.tensor([0.55]),
                       gap_pairs=torch.tensor([[0, 2]]), gap_uncertainty_score=torch.tensor([0.03]))
    scalar, _ = reported_uncertainty(nonadjacent, ("E0", "E1", "E2"), 3, crossing_gap_eV=0.1)
    assert [scalar[f"ntk_near_crossing_E{i}"] for i in range(3)] == [1.0, 1.0, 0.0]
    assert scalar["ntk_dE02"] == pytest.approx(0.03) and "ntk_dE01" not in scalar


def test_ntk_coupling_uncertainty_propagates_the_gap_uncertainty():
    scaled, sigma_s = torch.tensor([0.2, -0.4]), torch.tensor([0.01, 0.02])
    gap, sigma_gap = torch.tensor(0.05), torch.tensor(0.005)
    coupling, sigma = physical_coupling_uncertainty(scaled, sigma_s, gap, sigma_gap)
    torch.testing.assert_close(coupling, scaled / gap)
    torch.testing.assert_close(sigma, torch.sqrt((sigma_s / gap) ** 2 + (scaled * sigma_gap / gap ** 2) ** 2))
    # Closing the gap tenfold raises the coupling uncertainty far more than tenfold once the gap term dominates.
    _, closer = physical_coupling_uncertainty(scaled, sigma_s, gap / 10, sigma_gap)
    assert bool((closer > 10 * sigma).all())


def test_ntk_coupling_uncertainty_refuses_a_zero_gap():
    with pytest.raises(ValueError, match="zero gap"):
        physical_coupling_uncertainty(torch.tensor([0.2]), torch.tensor([0.01]), 0.0)


# --- HIPPYNN ----------------------------------------------------------------------------------------
# Small random graphs, molecular, periodic and multi-state, scored with HIPPYNN's plain-PyTorch kernels.

NETWORK = {"n_features": 8, "n_sensitivities": 4, "n_interaction_layers": 1, "n_atom_layers": 1}
COUPLING_NETWORK = {"possible_species": [0, 1, 8], "n_features": 6, "n_sensitivities": 4, "dist_soft_min": 0.5,
                    "dist_soft_max": 2.5, "dist_hard_max": 3.0, "n_interaction_layers": 1, "n_atom_layers": 1}


@pytest.fixture(scope="module")
def hippynn_ntk():
    """``alframework.ml_interfaces.hippynn.ntk``, with HIPPYNN's native kernels for this module's graphs."""
    pytest.importorskip("hippynn")
    from hippynn import custom_kernels
    from alframework.ml_interfaces.hippynn import ntk

    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels(False)
    try:
        yield ntk
    finally:
        custom_kernels.set_custom_kernels(previous)


def molecular_graph(kind, n_states=1):
    from hippynn.graphs import GraphModule, inputs, networks, targets
    from alframework.ml_interfaces.hippynn.compat import system_energy

    with torch.random.fork_rng():
        torch.manual_seed(7)
        species = inputs.SpeciesNode("species")
        positions = inputs.PositionsNode("positions")
        network = getattr(networks, kind)("hipnn", (species, positions), module_kwargs={
            "possible_species": [0, 1, 8], "dist_soft_min": 0.5, "dist_soft_max": 2.5, "dist_hard_max": 3.0,
            **NETWORK})
        energies = [system_energy(targets.HEnergyNode(f"energy{i}", network)) for i in range(n_states)]
        return GraphModule((species, positions), tuple(energies)).double().eval()


def periodic_graph():
    from hippynn.graphs import GraphModule, inputs, networks, targets
    from alframework.ml_interfaces.hippynn.compat import system_energy

    with torch.random.fork_rng():
        torch.manual_seed(3)
        species, positions, cell = inputs.SpeciesNode("species"), inputs.PositionsNode("positions"), inputs.CellNode("cell")
        network = networks.Hipnn("hipnn", (species, positions, cell), periodic=True, module_kwargs={
            "possible_species": [0, 14], "dist_soft_min": 1.0, "dist_soft_max": 3.5, "dist_hard_max": 4.0,
            **NETWORK})
        energy = system_energy(targets.HEnergyNode("energy", network))
        return GraphModule((species, positions, cell), (energy,)).double().eval()


@pytest.fixture(params=["Hipnn", "HipHopnn"])
def graph(request, hippynn_ntk):
    return molecular_graph(request.param)


@pytest.fixture
def atoms():
    return water()


def all_rows(estimator, atoms, engine, n_probes=2):
    structure = estimator.structure_from_atoms(atoms)
    u = gaussian_probes(n_probes, len(atoms), seed=1, device=estimator.device, dtype=estimator.dtype)
    v = gaussian_probes(n_probes, len(atoms), seed=2, device=estimator.device, dtype=estimator.dtype)
    request = RowRequest(energy=True, forces=True, force_probes=u, hvp_probes=v, bilinear_probes=(u, v))
    return (u, v), *compute_rows(estimator.adapter, structure, request, estimator.sketch, engine=engine)


def perturbed_predictions(estimator, graph, atoms, step, probe):
    """Energy, forces and H·probe of ``graph`` moved by ``step`` along the first sketch direction."""
    perturbed = copy.deepcopy(graph)
    direction = estimator.sketch.tangents(0)
    with torch.no_grad():
        for name, parameter in perturbed.named_parameters():
            if parameter.requires_grad:
                parameter.add_(step * direction[name][0])
    cell = [torch.tensor(np.asarray(atoms.cell))[None]] if np.any(atoms.pbc) else []
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
    energy = perturbed(torch.tensor(atoms.numbers)[None], positions[None], *cell)[0].sum()
    gradient, = torch.autograd.grad(energy, positions, create_graph=True)
    hvp, = torch.autograd.grad((gradient * probe).sum(), positions)
    return energy.detach(), -gradient.detach(), hvp


def row_case(hippynn_ntk, graph, atoms, periodic):
    if periodic:
        graph, atoms = periodic_graph(), silicon()
    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    return graph, atoms, estimator


@pytest.mark.parametrize("periodic", [False, True])
def test_hippynn_ntk_rows_match_parameter_finite_differences(hippynn_ntk, graph, atoms, periodic):
    graph, atoms, estimator = row_case(hippynn_ntk, graph, atoms, periodic)
    (_, v), _, rows = all_rows(estimator, atoms, "forward")
    plus = perturbed_predictions(estimator, graph, atoms, 1e-5, v[0])
    minus = perturbed_predictions(estimator, graph, atoms, -1e-5, v[0])
    difference = [(a - b) / 2e-5 for a, b in zip(plus, minus)]
    torch.testing.assert_close(rows["energy"][0, 0], difference[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows["forces"][0, ..., 0], difference[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows["hvp"][0, 0, ..., 0], difference[2], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("periodic", [False, True])
def test_hippynn_ntk_engines_give_the_same_rows(hippynn_ntk, graph, atoms, periodic):
    _, atoms, estimator = row_case(hippynn_ntk, graph, atoms, periodic)
    _, values, rows = all_rows(estimator, atoms, "forward")
    _, reverse_values, reverse = all_rows(estimator, atoms, "reverse")
    for key in rows:
        torch.testing.assert_close(reverse[key], rows[key], rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(reverse_values[key], values[key])


@pytest.mark.parametrize("periodic", [False, True])
def test_hippynn_ntk_projection_rows_are_linear_in_the_probes(hippynn_ntk, graph, atoms, periodic):
    _, atoms, estimator = row_case(hippynn_ntk, graph, atoms, periodic)
    (u, _), _, rows = all_rows(estimator, atoms, "forward")
    torch.testing.assert_close(rows["force_projection"][0], torch.einsum("knc,ncp->kp", u, rows["forces"][0]))
    torch.testing.assert_close(rows["bilinear"][0], torch.einsum("knc,kncp->kp", u, rows["hvp"][0]))


def test_hippynn_ntk_bilinear_rows_are_symmetric(hippynn_ntk, graph, atoms):
    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16)
    structure = estimator.structure_from_atoms(atoms)
    u = gaussian_probes(3, 3, seed=4, device="cpu", dtype=torch.float64)
    v = gaussian_probes(3, 3, seed=5, device="cpu", dtype=torch.float64)
    _, uv = compute_rows(estimator.adapter, structure, RowRequest(bilinear_probes=(u, v)), estimator.sketch)
    _, vu = compute_rows(estimator.adapter, structure, RowRequest(bilinear_probes=(v, u)), estimator.sketch)
    torch.testing.assert_close(uv["bilinear"], vu["bilinear"], rtol=1e-9, atol=1e-11)


def test_hippynn_ntk_streamed_scores_match_a_direct_solve(hippynn_ntk, graph, atoms):
    stretched = water(1.15)
    training = [atoms, stretched]
    energy_scale, force_scale = 0.2, 0.8
    estimator = hippynn_ntk.EnergyForceNTK(graph, iter(training), energy_scale=energy_scale, force_scale=force_scale,
                                           sketch_dim=16, seed=13)
    observations = []
    for frame in training:
        _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(frame),
                               RowRequest(forces=True), estimator.sketch)
        observations.extend([rows["energy"][0][None] / energy_scale, rows["forces"][0].reshape(-1, 16) / force_scale])
    observations = torch.cat(observations)
    precision = observations.T @ observations + estimator.ridge * torch.eye(16, dtype=torch.float64)
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(stretched),
                           RowRequest(forces=True), estimator.sketch)
    targets_ = torch.cat([rows["energy"][0][None], rows["forces"][0].reshape(-1, 16)])
    variance = estimator.ridge * (targets_ * torch.linalg.solve(precision, targets_.T).T).sum(1)
    result = estimator.predict_energy_forces(stretched)
    torch.testing.assert_close(result["energy_uncertainty_score"], variance[0].sqrt())
    torch.testing.assert_close(result["force_component_uncertainty_score"], variance[1:].sqrt().reshape(3, 3))
    torch.testing.assert_close(result["force_uncertainty_score"], variance[1:].reshape(3, 3).sum(1).sqrt())
    torch.testing.assert_close(result["max_force_uncertainty_score"], result["force_uncertainty_score"].max())
    torch.testing.assert_close(result["force_rms_uncertainty"], (variance[1:].sum() / 9).sqrt())
    assert estimator.training_structure_count == len(training)
    assert dict(estimator.posterior.counts) == {"energy": 2, "force": 18}


def test_hippynn_ntk_scores_follow_atom_order(hippynn_ntk, graph, atoms):
    stretched = water(1.15)
    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms, stretched], energy_scale=0.2, force_scale=0.8,
                                           sketch_dim=16, seed=13)
    result = estimator.predict_energy_forces(stretched)
    permutation = [2, 0, 1]
    reordered = estimator.predict_energy_forces(stretched[permutation])
    for key in ("forces", "force_component_uncertainty_score", "force_uncertainty_score"):
        torch.testing.assert_close(reordered[key], result[key][permutation])


def test_hippynn_ntk_fits_the_same_under_inference_mode(hippynn_ntk, graph, atoms):
    result = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8).predict_energy_forces(atoms)
    assert result["max_force_uncertainty_score"] > 0
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            fitted = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
            observed = fitted.predict_energy_forces(atoms)
        for key, expected in result.items():
            torch.testing.assert_close(observed[key], expected)


def test_hippynn_ntk_calculator_reports_atom_scores(hippynn_ntk, graph, atoms):
    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
    calculator = NTKASECalculator(estimator)
    calculator.calculate(atoms)
    np.testing.assert_allclose(calculator.results["alf_atomic_uncertainty"]["ntk_force"],
                               estimator.predict_energy_forces(atoms)["force_uncertainty_score"])


def test_hippynn_ntk_molecular_graph_refuses_a_periodic_structure(hippynn_ntk, graph, atoms):
    calculator = NTKASECalculator(hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8))
    periodic = atoms.copy()
    periodic.set_cell([6.0] * 3)
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="nonperiodic"):
        calculator.calculate(periodic)
    assert not calculator.results


def test_hippynn_ntk_periodic_graph_requires_full_periodicity(hippynn_ntk, atoms):
    with pytest.raises(ValueError, match="fully periodic"):
        hippynn_ntk.EnergyForceNTK(periodic_graph(), [atoms], energy_scale=0.2, force_scale=0.8)


def silicon_estimator(hippynn_ntk):
    return hippynn_ntk.EnergyForceNTK(periodic_graph(), [silicon(), silicon(2)], energy_scale=0.05, force_scale=0.3,
                                      sketch_dim=16)


def test_hippynn_ntk_periodic_scores_are_translation_invariant(hippynn_ntk):
    """A wrapped translation or a permutation of the atoms leaves a periodic cell's scores unchanged."""
    estimator = silicon_estimator(hippynn_ntk)
    query = silicon(3)
    result = estimator.predict(query, forces=True)
    shifted = query.copy()
    shifted.translate([0.37, -0.21, 0.55])
    shifted.wrap()
    permutation = np.random.default_rng(0).permutation(len(query))
    for other, order in ((shifted, slice(None)), (query[permutation], permutation)):
        observed = estimator.predict(other, forces=True)
        torch.testing.assert_close(observed["energy_uncertainty_score"], result["energy_uncertainty_score"])
        torch.testing.assert_close(observed["force_uncertainty_score"][0], result["force_uncertainty_score"][0][order])


def test_hippynn_ntk_energy_rows_scale_with_supercell_size(hippynn_ntk):
    estimator = silicon_estimator(hippynn_ntk)
    query = silicon(3)
    # A 2×2×2 supercell has 8 times the energy row, so 64 times the energy variance.
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(query), RowRequest(), estimator.sketch)
    _, big = compute_rows(estimator.adapter, estimator.structure_from_atoms(query.repeat(2)), RowRequest(),
                          estimator.sketch)
    torch.testing.assert_close(big["energy"], 8 * rows["energy"])
    calculator = NTKASECalculator(estimator)
    calculator.calculate(query)
    small = calculator.results["alf_uncertainty"]["ntk_energy_per_atom"]
    calculator.calculate(query.repeat(2))
    assert calculator.results["alf_uncertainty"]["ntk_energy_per_atom"] == pytest.approx(small)


@pytest.mark.parametrize("device", DEVICES)
def test_hippynn_ntk_float32_matches_float64(hippynn_ntk, graph, atoms, device):
    training = [atoms, water(1.15)]
    query = atoms.copy()
    query.positions[2] *= 1.1
    scales = dict(energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    options = dict(forces=True, force_probes=4, hessian_probes=4, hvp_probes=1)
    ref = hippynn_ntk.EnergyForceNTK(graph, training, **scales).predict(query, **options)
    # The GPU case exercises the default precision. An explicitly requested GPU
    # must be usable so the qualification command cannot pass by skipping it.
    dtype = torch.float32 if device == "cpu" else None
    estimator = hippynn_ntk.EnergyForceNTK(graph, training, **scales, device=device, dtype=dtype)
    assert estimator.dtype == torch.float32
    for value in (*estimator.adapter.live_parameters().values(), estimator.sketch.flat_chunk(0),
                  estimator.posterior.factor):
        assert value.device.type == torch.device(device).type
    with torch.inference_mode():
        observed = estimator.predict(query, **options)
    for key, expected in ref.items():
        assert observed[key].device.type == torch.device(device).type
        torch.testing.assert_close(observed[key].cpu().double(), expected, rtol=3e-3, atol=5e-5)


def hessian_fits(hippynn_ntk, graph, atoms):
    """Fits without Hessians, with two fixed probes per frame, and with full Hessians probed three times."""
    frames = (atoms, water(1.1))
    scales = dict(energy_scale=0.2, force_scale=0.8, sketch_dim=32, seed=5)
    plain = hippynn_ntk.EnergyForceNTK(graph, frames, **scales)
    probes = gaussian_probes(2, 3, seed=9, device="cpu", dtype=torch.float64).numpy()
    fixed = [frame.copy() for frame in frames]
    for frame in fixed:
        frame.info["hessian_probes"] = probes
    full = [frame.copy() for frame in frames]
    for frame in full:
        frame.info["hessian"] = np.zeros((9, 9))
    return (plain, hippynn_ntk.EnergyForceNTK(graph, fixed, **scales, hessian_scale=0.5),
            hippynn_ntk.EnergyForceNTK(graph, full, **scales, hessian_scale=0.5, hessian_fit_probes=3), probes)


def test_hippynn_ntk_hessian_labels_add_rows(hippynn_ntk, graph, atoms):
    plain, with_fixed, with_full, _ = hessian_fits(hippynn_ntk, graph, atoms)
    assert with_fixed.posterior.counts["hessian"] == 2 * 2 * 9
    assert with_full.posterior.counts["hessian"] == 2 * 3 * 9
    assert "hessian" not in plain.posterior.counts


def test_hippynn_ntk_hessian_labels_lower_hessian_variance(hippynn_ntk, graph, atoms):
    plain, with_fixed, with_full, probes = hessian_fits(hippynn_ntk, graph, atoms)
    _, rows = compute_rows(plain.adapter, plain.structure_from_atoms(atoms),
                           RowRequest(hvp_probes=torch.as_tensor(probes)), plain.sketch)
    before = plain.posterior.variance(rows["hvp"]).sum()
    for estimator in (with_fixed, with_full):
        assert estimator.posterior.variance(rows["hvp"]).sum() < 0.5 * before
    hessian = with_fixed.predict(atoms, hessian_probes=8)["hessian_rms_uncertainty"]
    assert hessian < plain.predict(atoms, hessian_probes=8)["hessian_rms_uncertainty"]


TWO_STATE_HESSIAN_SCALES = dict(energy_scale=0.2, force_scale=0.8, hessian_scale=0.1, sketch_dim=32, seed=5)


def test_hippynn_ntk_hessian_rows_come_only_from_supervised_states(hippynn_ntk, atoms):
    graph = molecular_graph("Hipnn", n_states=2)
    probes = gaussian_probes(2, 3, seed=9, device="cpu", dtype=torch.float64)
    fits = {}
    for states in (None, ["E1"], ["E0", "E1"]):
        frame = atoms.copy()
        frame.info["hessian_probes"] = probes.numpy()
        if states is not None:
            frame.info["hessian_states"] = states
        fits[None if states is None else tuple(states)] = hippynn_ntk.EnergyForceNTK(graph, [frame],
                                                                                     **TWO_STATE_HESSIAN_SCALES)
    assert fits[None].posterior.counts["hessian"] == fits[("E1",)].posterior.counts["hessian"] == 2 * 9
    assert fits[("E0", "E1")].posterior.counts["hessian"] == 2 * 2 * 9
    _, rows = compute_rows(fits[None].adapter, fits[None].structure_from_atoms(atoms), RowRequest(hvp_probes=probes),
                           fits[None].sketch)
    for state, other in ((0, ("E1",)), (1, None)):
        # Supervising a state's Hessian lowers that state's HVP variance more than the other state's.
        own = fits[None if state == 0 else ("E1",)].posterior.variance(rows["hvp"][state]).sum()
        assert own < fits[other].posterior.variance(rows["hvp"][state]).sum()


def test_hippynn_ntk_refuses_hessian_states_the_model_lacks(hippynn_ntk, atoms):
    frame = atoms.copy()
    frame.info.update(hessian_probes=gaussian_probes(2, 3, seed=9, device="cpu", dtype=torch.float64).numpy(),
                      hessian_states=["E7"])
    with pytest.raises(ValueError, match="not model states"):
        hippynn_ntk.EnergyForceNTK(molecular_graph("Hipnn", n_states=2), [frame], **TWO_STATE_HESSIAN_SCALES)


def test_hippynn_ntk_gap_score_is_the_difference_row(hippynn_ntk, atoms):
    estimator = hippynn_ntk.EnergyForceNTK(molecular_graph("Hipnn", n_states=2), [atoms], energy_scale=0.2,
                                           force_scale=0.8, sketch_dim=16)
    assert estimator.state_names == ("E0", "E1")
    result = estimator.predict(atoms, forces=True, hessian_probes=2)
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(atoms), RowRequest(), estimator.sketch)
    torch.testing.assert_close(result["gap_uncertainty_score"],
                               estimator.posterior.variance(rows["energy"][1] - rows["energy"][0]).sqrt()[None])


def test_hippynn_ntk_calculator_reports_every_state(hippynn_ntk, atoms):
    estimator = hippynn_ntk.EnergyForceNTK(molecular_graph("Hipnn", n_states=2), [atoms], energy_scale=0.2,
                                           force_scale=0.8, sketch_dim=16)
    calculator = NTKASECalculator(estimator, selected_state=1, hessian_probes=2)
    calculator.calculate(atoms)
    scores = calculator.results["alf_uncertainty"]
    assert {"ntk_energy_E0", "ntk_energy_E1", "ntk_force_max_E1", "ntk_hessian_E0", "ntk_dE01"} <= set(scores)
    assert set(calculator.results["alf_atomic_uncertainty"]) == {"ntk_force_E0", "ntk_force_E1"}
    assert calculator.results["energy"] == pytest.approx(float(estimator.predict(atoms, forces=False)["energy"][1]))


@pytest.mark.parametrize("engine", ["ase", "torchsim", "alchemi"])
def test_hippynn_ntk_gap_selects_frames_on_the_configured_state(hippynn_ntk, atoms, engine):
    from alframework.samplers.batched_mlmd import sample_trajectories
    from alframework.tools.molecules import MoleculesObject

    require_engine(engine)
    estimator = hippynn_ntk.EnergyNTK(molecular_graph("Hipnn", n_states=3), [atoms], sketch_dim=16)
    evaluator = NTKPackedEvaluator(estimator, selected_state=0, forces=False, gap_pairs=[[0, 2], [1, 2]])
    config = {
        "driver": {"engine": engine}, "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 2}, "random_seed": 7,
        "dt": 0.25, "maxt": 0.002, "Ncheck": 2, "min_time": 0.0005, "distcut": 0.3,
        "metadata_files": "none", "srt_temp": [300., 300.], "end_temp": [300., 300.],
        "amp_temp": [0., 0.], "per_temp": [1., 1.], "amp_dens": None, "per_dens": None, "end_dens": None,
        "acquisition": {"terms": {"ntk_dE02": {"uncertainty": "ntk_dE02", "cutoff": 1e-12}},
                        "eligibility": ["ntk_dE02"], "strategy": "largest"},
    }
    molecules = [MoleculesObject(atoms.copy(), f"mol-0000-{i:010d}") for i in range(2)]
    for molecule in sample_trajectories(molecules, evaluator, config, device="cpu"):
        metadata = molecule.get_metadata()
        assert metadata["trajectory_outcome"] == "eligible", metadata
        assert metadata["selected_state"] == 2
        structure = estimator.structure_from_atoms(molecule.get_atoms())
        values, rows = compute_rows(estimator.adapter, structure, RowRequest(), estimator.sketch)
        score = estimator.posterior.variance(rows["energy"][2] - rows["energy"][0]).sqrt()
        assert metadata["E2_mean"] == pytest.approx(float(values["energy"][2]))
        assert metadata["ntk_dE02"] == pytest.approx(float(score))
        assert "ntk_dE01" not in metadata


@pytest.mark.parametrize("engine", ["ase", "torchsim", "alchemi"])
def test_hippynn_ntk_drives_gap_seeking_without_an_ensemble(hippynn_ntk, atoms, engine):
    """Gap-seeking moves on the mean state energies, which one multi-state model supplies; no spread is needed."""
    from alframework.samplers.batched_mlmd import sample_trajectories
    from alframework.tools.molecules import MoleculesObject

    require_engine(engine)
    evaluator = NTKPackedEvaluator(hippynn_ntk.EnergyNTK(molecular_graph("Hipnn", n_states=3), [atoms], sketch_dim=16),
                                   selected_state=1, forces=False)
    config = {
        "driver": {"engine": engine}, "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 1}, "random_seed": 7,
        "dt": 0.25, "maxt": 0.004, "Ncheck": 2, "min_time": 0.0005, "distcut": 0.3,
        "metadata_files": "none", "srt_temp": [300., 300.], "end_temp": [300., 300.],
        "amp_temp": [0., 0.], "per_temp": [1., 1.], "amp_dens": None, "per_dens": None, "end_dens": None,
        # A threshold above any gap of this model, so the trajectory switches at its first check.
        "gap_seeking": {"enabled": True, "trigger_gap_threshold_eV": 1e6, "sigma": 0.5, "alpha_eV": 0.1},
        "acquisition": {"terms": {"ntk": {"uncertainty": "ntk_energy_E1", "cutoff": 1e9}},
                        "eligibility": ["ntk"], "strategy": "largest"},
    }
    [molecule] = sample_trajectories([MoleculesObject(atoms.copy(), "mol-0000-0000000000")], evaluator, config,
                                     device="cpu")
    metadata = molecule.get_metadata()
    assert metadata["trajectory_outcome"] == "completed", metadata.get("sampling_error")
    assert metadata["gap_seeking_switched"] and metadata["gap_seeking_pair"] in ([0, 1], [1, 2])


def test_hippynn_ntk_refuses_unordered_gap_pairs(hippynn_ntk, atoms):
    estimator = hippynn_ntk.EnergyNTK(molecular_graph("Hipnn", n_states=3), [atoms], sketch_dim=16)
    with pytest.raises(ValueError, match="ordered states"):
        estimator.predict(atoms, forces=False, gap_pairs=[[2, 0]])


class StepPackedEngine(PackedMLMDEngine):
    def _baoab(self):
        # Change the current geometry without a dependency on an integrator.
        self.state.positions *= 1.001
        return self._evaluate()


def packed_engine(hippynn_ntk, graph, atoms, monkeypatch):
    """A packed engine on water and H2, recording the atoms of each NTK scoring call."""
    second = Atoms("H2", positions=[[0., 0., 0.], [1.3, 0., 0.]])
    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms, second], energy_scale=0.2, force_scale=0.8,
                                           dtype=torch.float32)
    evaluator = NTKPackedEvaluator(estimator)
    score_calls = []
    observe = evaluator.observe_uncertainty

    def record_observe(numbers, positions, cell, pbc):
        score_calls.append(numbers.clone())
        return observe(numbers, positions, cell, pbc)

    monkeypatch.setattr(evaluator, "observe_uncertainty", record_observe)
    engine = StepPackedEngine([atoms.copy(), second.copy()], evaluator, 0.5, [300., 300.],
                              [np.random.RandomState(2), np.random.RandomState(3)])
    return engine, estimator, evaluator, score_calls


def test_hippynn_ntk_packed_engine_scores_only_observed_frames(hippynn_ntk, graph, atoms, monkeypatch):
    engine, estimator, _, score_calls = packed_engine(hippynn_ntk, graph, atoms, monkeypatch)
    # Refresh ordinary forces and screen a frame without invoking NTK scoring.
    engine.scale_cell(0, 1.02)
    assert not engine.step([0, 1], 2)
    assert "uncertainty" not in engine.observe_physical(0)["prediction"]
    assert not score_calls
    for index in (0, 1):
        reference = estimator.predict_energy_forces(engine.atoms(index).copy())
        prediction = engine.observe(index)["prediction"]
        np.testing.assert_allclose(prediction["energies"][0, 0], reference["energy"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["forces"][0][0], reference["forces"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["atomic_uncertainty"][0]["ntk_force"],
                                   reference["force_uncertainty_score"], rtol=3e-5, atol=1e-6)
    assert [len(numbers) for numbers in score_calls] == [3, 2]
    engine.finish(0)
    engine.finish(1)


def test_hippynn_ntk_packed_scores_follow_the_remaining_atoms(hippynn_ntk, graph, atoms, monkeypatch):
    engine, estimator, evaluator, score_calls = packed_engine(hippynn_ntk, graph, atoms, monkeypatch)
    assert not engine.step([0, 1], 2)
    engine.finish(0)
    engine.scale_cell(1, 1.05)
    assert not engine.step([1], 1)
    observed = engine.observe(1)["prediction"]["atomic_uncertainty"][0]["ntk_force"]
    remaining = engine.atoms(1)
    assert len(score_calls) == 1
    np.testing.assert_array_equal(score_calls[0], remaining.numbers)
    expected = estimator.predict_energy_forces(remaining)["force_uncertainty_score"]
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=1e-6)
    assert check_evaluator(evaluator, [remaining])[0]["ntk_force_max"] == pytest.approx(float(expected.max()))
    engine.finish(1)


def test_hippynn_ntk_scoring_restores_the_kernel_selection(hippynn_ntk, graph, atoms):
    """Scoring leaves HIPPYNN's kernel selection as it found it, so packed MD keeps its own kernels."""
    from hippynn import custom_kernels

    estimator = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=8)
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels("pytorch")
    try:
        estimator.predict(atoms, hessian_probes=1)
        assert custom_kernels.CUSTOM_KERNELS_ACTIVE == "pytorch"
    finally:
        custom_kernels.set_custom_kernels(previous)


ENERGY_OPTIONS = dict(sketch_dim=16, seed=13, ridge=0.2)


@pytest.mark.parametrize("device", DEVICES)
def test_hippynn_energy_ntk_basis_round_trips(hippynn_ntk, graph, atoms, device, tmp_path):
    stretched = water(1.13)
    cpu = hippynn_ntk.EnergyNTK(graph, [atoms, stretched], **ENERGY_OPTIONS)
    with torch.inference_mode():
        fitted = hippynn_ntk.EnergyNTK(graph, [atoms, stretched], **ENERGY_OPTIONS, device=device)
        original = fitted.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    path = tmp_path / "energy_ntk.npz"
    fitted.save_basis(path)
    restored = hippynn_ntk.EnergyNTK.from_basis(graph, path, device=device)
    observed = restored.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    reference = cpu.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    assert restored.basis_id == observed["basis_id"] == fitted.basis_id
    np.testing.assert_allclose(observed["row"], original["row"], rtol=3e-5, atol=1e-6)
    assert initial_energy_score({"energy_ntk": observed}) == pytest.approx(
        initial_energy_score({"energy_ntk": reference}), rel=3e-3, abs=5e-5)


def test_hippynn_energy_ntk_never_differentiates_coordinates(hippynn_ntk, graph, atoms, monkeypatch):
    requests = []
    compute = estimator_module.compute_rows

    def record(adapter, structure, request, *args, **kwargs):
        requests.append(request)
        return compute(adapter, structure, request, *args, **kwargs)

    monkeypatch.setattr(estimator_module, "compute_rows", record)
    stretched = water(1.13)
    with torch.inference_mode():
        fitted = hippynn_ntk.EnergyNTK(graph, [atoms, stretched], **ENERGY_OPTIONS)
        fitted.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    assert requests and not any(request.needs_derivatives for request in requests)


def test_hippynn_energy_ntk_basis_refuses_a_changed_model(hippynn_ntk, graph, atoms, tmp_path):
    hippynn_ntk.EnergyNTK(graph, [atoms, water(1.13)], **ENERGY_OPTIONS).save_basis(tmp_path / "energy_ntk.npz")
    changed = copy.deepcopy(graph)
    with torch.no_grad():
        next(p for p in changed.parameters() if p.requires_grad).add_(1.0)
    with pytest.raises(ValueError, match="model snapshot"):
        hippynn_ntk.EnergyNTK.from_basis(changed, tmp_path / "energy_ntk.npz")


def test_hippynn_ntk_force_basis_with_hessian_rows_round_trips(hippynn_ntk, graph, atoms, tmp_path):
    frame = atoms.copy()
    frame.info["hessian"] = np.zeros((9, 9))
    fitted = hippynn_ntk.EnergyForceNTK(graph, [frame], energy_scale=0.2, force_scale=0.8, hessian_scale=0.5,
                                        sketch_dim=16)
    path = tmp_path / "ntk.npz"
    fitted.save_basis(path)
    restored = estimator_module.NTKEstimator.from_basis(fitted.adapter, path)
    assert restored.basis_id == fitted.basis_id and restored.posterior.counts == fitted.posterior.counts
    for key, value in fitted.predict(atoms, hessian_probes=2).items():
        torch.testing.assert_close(restored.predict(atoms, hessian_probes=2)[key], value)


def test_hippynn_ntk_packed_energy_selection_reports_force_diagnostics(hippynn_ntk, graph, atoms):
    energy = hippynn_ntk.EnergyNTK(graph, [atoms], sketch_dim=16)
    forces = hippynn_ntk.EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16)
    evaluator = NTKPackedEvaluator(energy, force_diagnostics=forces)
    with torch.inference_mode():
        observation = evaluator.observe_uncertainty(
            torch.tensor(atoms.numbers), torch.tensor(atoms.positions), None, torch.tensor(atoms.pbc))
    reference = forces.predict_energy_forces(atoms)
    assert observation["energy_ntk"]["basis_id"] == energy.basis_id
    assert observation["uncertainty"]["ntk_energy"] ** 2 == pytest.approx(initial_energy_score(observation))
    assert observation["uncertainty"]["ntk_force_max"] == pytest.approx(float(reference["max_force_uncertainty_score"]))
    np.testing.assert_allclose(observation["atomic_uncertainty"]["ntk_force"], reference["force_uncertainty_score"])


def model_a(n_states=3):
    """A multistate energy model standing in for a trained Model A."""
    from hippynn.graphs import GraphModule, inputs, networks, targets
    from alframework.ml_interfaces.hippynn.compat import system_energy

    with torch.random.fork_rng():
        torch.manual_seed(5)
        species, positions = inputs.SpeciesNode("species"), inputs.PositionsNode("positions")
        net = networks.Hipnn("hipnn", (species, positions), module_kwargs=COUPLING_NETWORK)
        energies = [system_energy(targets.HEnergyNode(f"E{i}", net)) for i in range(n_states)]
        return GraphModule((species, positions), tuple(energies)).double().eval()


def toy_coupling_graph():
    """species, positions, and a per-system gap input; one output: an energy-like readout times the gap."""
    from hippynn.graphs import GraphModule, IdxType, inputs, networks, targets
    from alframework.ml_interfaces.hippynn.compat import system_energy

    with torch.random.fork_rng():
        torch.manual_seed(9)
        species, positions = inputs.SpeciesNode("species"), inputs.PositionsNode("positions")
        net = networks.Hipnn("hipnn", (species, positions), module_kwargs=COUPLING_NETWORK)
        system_index = getattr(IdxType, "Systems", None) or IdxType.Molecules
        gap = inputs.InputNode("GAP12", db_name="GAP12", index_state=system_index)
        out = system_energy(targets.HEnergyNode("readout", net)) * gap
        out.db_name = "coupling12"
        return GraphModule((species, positions, gap), (out,)).double().eval()


def coupling_estimator(hippynn_ntk):
    gap = {"GAP12": torch.tensor([[0.7]], dtype=torch.float64)}
    adapter = hippynn_ntk.HippynnOutputsNTKAdapter(toy_coupling_graph(), extra_inputs=lambda structure: gap,
                                                   supports_forward=True)
    return adapter, NTKEstimator(adapter, [water(), water(1.08)], output_scales={"coupling12": 0.05}, sketch_dim=8)


def test_hippynn_ntk_output_rows_match_parameter_finite_differences(hippynn_ntk):
    adapter, estimator = coupling_estimator(hippynn_ntk)
    assert adapter.output_names == ("coupling12",) and adapter.extra_names == ("GAP12",)
    structure = estimator.structure_from_atoms(water(1.15))

    def shifted(step):
        moved = parameter_step(adapter, estimator.sketch, step)
        return moved.outputs(structure, structure.positions, None, ("coupling12",))["coupling12"]

    difference = (shifted(1e-5) - shifted(-1e-5)) / 2e-5
    for engine in ("forward", "reverse"):
        _, rows = compute_rows(adapter, structure, RowRequest(energy=False, outputs=("coupling12",)),
                               estimator.sketch, engine=engine)
        torch.testing.assert_close(rows["output:coupling12"][..., 0], difference, atol=1e-7, rtol=1e-5)


def test_hippynn_ntk_output_basis_round_trips(hippynn_ntk, tmp_path):
    adapter, estimator = coupling_estimator(hippynn_ntk)
    assert dict(estimator.posterior.counts) == {"output:coupling12": 2} and not estimator.state_names
    result = estimator.predict(water(1.15), outputs=("coupling12",))
    assert set(result) == {"output_coupling12", "output_coupling12_uncertainty_score"}
    estimator.save_basis(tmp_path / "basis.npz")
    restored = NTKEstimator.from_basis(adapter, tmp_path / "basis.npz")
    torch.testing.assert_close(restored.predict(water(1.15), outputs=("coupling12",))["output_coupling12_uncertainty_score"],
                               result["output_coupling12_uncertainty_score"])


def test_hippynn_ntk_outputs_refuse_hessian_probes(hippynn_ntk):
    _, estimator = coupling_estimator(hippynn_ntk)
    with pytest.raises(ValueError, match="energies"):
        estimator.predict(water(), hessian_probes=2)


def test_hippynn_ntk_outputs_require_extra_inputs(hippynn_ntk):
    with pytest.raises(ValueError, match="extra_inputs"):
        hippynn_ntk.HippynnOutputsNTKAdapter(toy_coupling_graph())


def test_hippynn_ntk_frozen_model_supplies_coupling_inputs(hippynn_ntk):
    a = hippynn_ntk.HippynnNTKAdapter(model_a(), dtype=torch.float64)
    supply = hippynn_ntk.frozen_model_inputs(a, {"GDV12": ("GDV", 1, 2), "GAP12": ("GAP", 1, 2)})
    structure = a.structure(water().numbers, water().positions)
    values = supply(structure)
    energies, forces = a.predict_batch(structure.numbers, structure.positions, None, structure.pbc, [3])
    torch.testing.assert_close(values["GAP12"], (energies[2] - energies[1]).reshape(1))
    torch.testing.assert_close(values["GDV12"], (forces[1] - forces[2]).reshape(1, 9))


def test_hippynn_ntk_engines_agree_through_lora(hippynn_ntk, graph):
    from alframework.ml_interfaces.hippynn.lora import apply_lora

    apply_lora(graph, rank=2)
    with torch.no_grad():
        for name, parameter in graph.named_parameters():
            if name.endswith("_lora_B"):
                parameter.normal_(0.0, 0.1)
    estimator = hippynn_ntk.EnergyForceNTK(graph, [water()], energy_scale=0.05, force_scale=0.1, sketch_dim=6)
    assert all(not n.endswith("_lora_base") for n in estimator.adapter.live_parameters())
    structure = estimator.structure_from_atoms(water())
    request = RowRequest(forces=True)
    _, forward = compute_rows(estimator.adapter, structure, request, estimator.sketch, engine="forward")
    _, reverse = compute_rows(estimator.adapter, structure, request, estimator.sketch, engine="reverse")
    for key in forward:
        torch.testing.assert_close(forward[key], reverse[key], rtol=1e-9, atol=1e-11)


# --- MACE -------------------------------------------------------------------------------------------
# A small random ScaleShiftMACE, molecular and periodic.

MACE_SPECIES = [1, 6, 8, 14]


@pytest.fixture(scope="module")
def mace_model():
    pytest.importorskip("mace")
    # MACE allowlists e3nn's constants for torch.load, so import it before e3nn.
    from mace import modules
    from e3nn import o3

    with torch.random.fork_rng():
        torch.manual_seed(0)
        block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
        return modules.ScaleShiftMACE(
            r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2, interaction_cls=block,
            interaction_cls_first=block, num_interactions=2, num_elements=len(MACE_SPECIES),
            hidden_irreps=o3.Irreps("4x0e + 4x1o"), MLP_irreps=o3.Irreps("4x0e"), gate=torch.nn.functional.silu,
            atomic_energies=np.zeros(len(MACE_SPECIES)), avg_num_neighbors=6.0, atomic_numbers=MACE_SPECIES,
            correlation=2, radial_type="bessel", atomic_inter_scale=1.0, atomic_inter_shift=0.0,
        ).double().eval()


def mace_case(model, periodic):
    from alframework.ml_interfaces.ntk.mace import MACENTKAdapter

    atoms, training = (silicon(1, cubic=False), [silicon(2, cubic=False)]) if periodic else (water(), [water()])
    adapter = MACENTKAdapter(model, dtype=torch.float64)
    estimator = NTKEstimator(adapter, training, energy_scale=0.1, force_scale=0.5, sketch_dim=8, seed=2)
    return atoms, adapter, estimator


@pytest.mark.parametrize("periodic", [False, True])
def test_mace_ntk_rows_match_parameter_finite_differences(mace_model, periodic):
    atoms, adapter, estimator = mace_case(mace_model, periodic)
    structure = estimator.structure_from_atoms(atoms)
    probes = gaussian_probes(1, len(atoms), seed=3, device="cpu", dtype=torch.float64)
    _, rows = compute_rows(adapter, structure, RowRequest(forces=True, hvp_probes=probes), estimator.sketch)

    def predict(step):
        shifted = parameter_step(adapter, estimator.sketch, step)
        x = structure.positions.clone().requires_grad_(True)
        energy = shifted.energies(structure, x)[0]
        gradient, = torch.autograd.grad(energy, x, create_graph=True)
        hvp, = torch.autograd.grad((gradient * probes[0]).sum(), x)
        return energy.detach(), -gradient.detach(), hvp

    plus, minus = predict(1e-5), predict(-1e-5)
    for key, (a, b) in zip(("energy", "forces", "hvp"), zip(plus, minus)):
        row = rows[key][0, 0, ..., 0] if key == "hvp" else rows[key][0, ..., 0]
        torch.testing.assert_close(row, (a - b) / 2e-5, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("periodic", [False, True])
def test_mace_ntk_packed_evaluator_predicts_like_the_rows(mace_model, periodic):
    atoms, adapter, estimator = mace_case(mace_model, periodic)
    values, _ = compute_rows(adapter, estimator.structure_from_atoms(atoms), RowRequest(forces=True), estimator.sketch)
    energies, forces = NTKPackedEvaluator(estimator).evaluate(
        torch.tensor(atoms.numbers), torch.tensor(atoms.positions), torch.tensor(np.asarray(atoms.cell))[None],
        torch.tensor(atoms.pbc), [len(atoms)])
    torch.testing.assert_close(energies[0], values["energy"])
    torch.testing.assert_close(forces[0], values["forces"][0])


def test_mace_ntk_checkpoint_basis_round_trips(mace_model, tmp_path):
    from alframework.ml_interfaces.ntk.mace import MACENTKAdapter

    checkpoint = tmp_path / "mace.model"
    torch.save(mace_model, checkpoint)
    estimator = NTKEstimator(MACENTKAdapter.from_checkpoint(checkpoint, dtype=torch.float64), [water()],
                             energy_scale=0.1, sketch_dim=8)
    estimator.save_basis(tmp_path / "basis.npz")
    restored = NTKEstimator.from_basis(MACENTKAdapter(mace_model, dtype=torch.float64), tmp_path / "basis.npz")
    query = water(1.1)
    torch.testing.assert_close(restored.predict(query, forces=False)["energy_uncertainty_score"],
                               estimator.predict(query, forces=False)["energy_uncertainty_score"])


def test_mace_ntk_refuses_species_the_model_lacks(mace_model):
    from alframework.ml_interfaces.ntk.mace import MACENTKAdapter

    estimator = NTKEstimator(MACENTKAdapter(mace_model, dtype=torch.float64), [water()], energy_scale=0.1,
                             sketch_dim=8)
    with pytest.raises(ValueError, match="no species"):
        estimator.predict(Atoms("Na", positions=[[0, 0, 0]]))


# --- X-MACE -----------------------------------------------------------------------------------------
# Small random three-state ExcitedMACE and AutoencoderExcitedMACE models.

XMACE_SPECIES = [1, 6, 8]


def build_xmace(autoencoder):
    excited = pytest.importorskip("mace.modules.excited", reason="X-MACE is not installed")
    from mace import modules  # Before e3nn, as for MACE.
    from e3nn import o3

    with torch.random.fork_rng():
        torch.manual_seed(7)
        cls = excited.AutoencoderExcitedMACE if autoencoder else excited.ExcitedMACE
        extra = {"num_permutational_invariant": 6} if autoencoder else {}
        model = cls(r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
                    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
                    interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
                    num_interactions=2, num_elements=len(XMACE_SPECIES), hidden_irreps=o3.Irreps("4x0e + 4x1o"),
                    MLP_irreps=o3.Irreps("4x0e"), atomic_energies=np.array([-1.0, -3.0, -5.0]),
                    avg_num_neighbors=3.0, atomic_numbers=XMACE_SPECIES, correlation=2, gate=torch.nn.functional.silu,
                    n_energies=3, compute_nacs=True, **extra)
        # Readouts start at zero, which puts every state on the backbone surface. Separate the states.
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for readout in model.readouts:
                for parameter in readout.parameters():
                    parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
        return model.double().eval()


@pytest.fixture(params=[False, True], ids=["ExcitedMACE", "AutoencoderExcitedMACE"])
def xmace_model(request):
    return build_xmace(request.param)


def xmace_adapter(model, **options):
    from alframework.ml_interfaces.ntk.xmace import XMACENTKAdapter

    return XMACENTKAdapter(model, dtype=torch.float64, **options)


def xmace_outputs(adapter):
    return ("ground_energy", "state_offsets") + (("latent", "state_matrix") if adapter.autoencoder else ())


def test_xmace_ntk_rows_match_parameter_finite_differences(xmace_model):
    adapter = xmace_adapter(xmace_model)
    assert adapter.state_names == ("E0", "E1", "E2")
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, force_scale=0.5, sketch_dim=6, seed=2)
    structure = estimator.structure_from_atoms(water(1.1))
    probes = gaussian_probes(1, 3, seed=3, device="cpu", dtype=torch.float64)
    names = xmace_outputs(adapter)
    _, rows = compute_rows(adapter, structure, RowRequest(forces=True, hvp_probes=probes, outputs=names),
                           estimator.sketch)

    def predict(step):
        shifted = parameter_step(adapter, estimator.sketch, step)
        x = structure.positions.clone().requires_grad_(True)
        energy = shifted.energies(structure, x)
        gradients = torch.stack([torch.autograd.grad(e, x, create_graph=True)[0] for e in energy])
        hvp = torch.stack([torch.autograd.grad((g * probes[0]).sum(), x, retain_graph=True)[0] for g in gradients])
        smooth = shifted.outputs(structure, structure.positions, None, names)
        return energy.detach(), -gradients.detach(), hvp, smooth

    (e1, f1, h1, s1), (e2, f2, h2, s2) = predict(1e-5), predict(-1e-5)
    torch.testing.assert_close(rows["energy"][..., 0], (e1 - e2) / 2e-5, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(rows["forces"][..., 0], (f1 - f2) / 2e-5, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(rows["hvp"][:, 0, ..., 0], (h1 - h2) / 2e-5, atol=1e-5, rtol=1e-4)
    for name in names:
        torch.testing.assert_close(rows[f"output:{name}"][..., 0], (s1[name] - s2[name]) / 2e-5,
                                   atol=1e-6, rtol=1e-4)


def test_xmace_ntk_gap_rows_are_differences_of_state_rows(xmace_model):
    estimator = NTKEstimator(xmace_adapter(xmace_model), [water()], energy_scale=0.1, force_scale=0.5, sketch_dim=6,
                             seed=2)
    structure = estimator.structure_from_atoms(water(1.1))
    _, rows = compute_rows(estimator.adapter, structure, RowRequest(), estimator.sketch)
    torch.testing.assert_close(estimator.score(structure, forces=False)["gap_uncertainty_score"],
                               estimator.posterior.variance(rows["energy"][1:] - rows["energy"][:-1]).sqrt())


def test_xmace_ntk_autoencoder_state_matrix_diagonalizes_to_the_state_energies():
    adapter = xmace_adapter(build_xmace(True))
    structure = adapter.structure(water(1.07).numbers, water(1.07).positions)
    elements = adapter.outputs(structure, structure.positions, None, ("state_matrix",))["state_matrix"]
    matrix = torch.diag(elements[:3]) + torch.diag(elements[3:], 1) + torch.diag(elements[3:], -1)
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix), adapter.energies(structure, structure.positions))


def test_xmace_ntk_latent_outputs_require_an_autoencoder():
    adapter = xmace_adapter(build_xmace(False))
    structure = adapter.structure(water(1.07).numbers, water(1.07).positions)
    with pytest.raises(ValueError, match="Autoencoder"):
        adapter.outputs(structure, structure.positions, None, ("latent",))


def test_xmace_ntk_leaves_a_frozen_backbone_out_of_the_posterior(xmace_model):
    frozen = copy.deepcopy(xmace_model)
    for parameter in frozen.backbone.parameters():
        parameter.requires_grad_(False)
    adapter = xmace_adapter(frozen)
    trained = {name for name, p in frozen.named_parameters() if p.requires_grad}
    assert set(adapter.live_parameters()) == trained and not any(n.startswith("backbone.") for n in trained)
    everything = xmace_adapter(frozen, parameters="all")
    assert len(everything.live_parameters()) == len(list(frozen.parameters()))
    assert adapter.digest() != everything.digest()
    estimator = NTKEstimator(adapter, [water()], energy_scale=0.1, sketch_dim=4)
    assert estimator.sketch.total == sum(p.numel() for p in adapter.live_parameters().values())


def xmace_estimator(model):
    return NTKEstimator(xmace_adapter(model), [water()], energy_scale=0.1, force_scale=0.5, sketch_dim=6)


def test_xmace_ntk_predicts_like_the_xmace_calculator(xmace_model):
    from mace.calculators.excited import ExcitedMACECalculator

    estimator = xmace_estimator(xmace_model)
    query = water(1.1)
    reference = ExcitedMACECalculator(None, models=[copy.deepcopy(xmace_model)], default_dtype="float64", state=1)
    reference.calculate(query)
    energies, forces = estimator.adapter.predict_batch(
        torch.tensor(query.numbers), torch.tensor(query.positions), None, torch.tensor(query.pbc), [3])
    np.testing.assert_allclose(energies[:, 0].numpy(), reference.results["energies"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(forces.permute(1, 0, 2).numpy(), reference.results["forces_states"], atol=1e-8)
    calculator = NTKASECalculator(estimator, selected_state=1)
    calculator.calculate(query)
    assert calculator.results["energy"] == pytest.approx(reference.results["energy"])


def test_xmace_ntk_calculator_reports_state_scores(xmace_model):
    estimator = xmace_estimator(xmace_model)
    calculator = NTKASECalculator(estimator, selected_state=1, hessian_probes=2,
                                  outputs=xmace_outputs(estimator.adapter), crossing_gap_eV=1e3)
    calculator.calculate(water(1.1))
    assert {"ntk_hessian_E1", "ntk_dE12", "ntk_state_offsets_max", "ntk_near_crossing_E1"} <= set(
        calculator.results["alf_uncertainty"])


def test_xmace_ntk_crossing_window_flags_states(xmace_model):
    estimator = xmace_estimator(xmace_model)
    states = estimator.adapter.state_names
    # With a crossing window wider than every gap, each state is flagged; with none, no state is.
    wide = NTKASECalculator(estimator, crossing_gap_eV=1e3)
    wide.calculate(water(1.1))
    assert all(wide.results["alf_uncertainty"][f"ntk_near_crossing_{state}"] == 1.0 for state in states)
    closed = NTKASECalculator(estimator, crossing_gap_eV=0.0)
    closed.calculate(water(1.1))
    assert not any(closed.results["alf_uncertainty"][f"ntk_near_crossing_{state}"] for state in states)


# --- UMA --------------------------------------------------------------------------------------------
# Provenance refusals, without UMA weights: they run before FairChem is needed.

def uma_member(directory):
    manifest = {'version': 1, 'task': 'omol', 'ntk': {}, 'charge': 0, 'spin': 1,
                'lora': True, 'train': {'frames': [{'id': 'frame-1'}]}}
    (directory / 'training_split.json').write_text(json.dumps(manifest))
    (directory / 'uma.pt').write_bytes(b'model')
    (directory / 'uma_lora.pt').write_bytes(b'factors')
    return directory


def test_uma_ntk_refuses_a_different_charge_or_spin(tmp_path):
    from alframework.ml_interfaces.ntk.uma import load_uma_ntk_estimator

    with pytest.raises(ValueError, match='charge and spin'):
        load_uma_ntk_estimator(uma_member(tmp_path), charge=1)


def test_uma_ntk_refuses_a_basis_for_changed_lora_factors(tmp_path):
    from alframework.ml_interfaces.ntk.uma import _member_artifacts, load_uma_ntk_estimator

    member = uma_member(tmp_path)
    # The provenance prepare_uma_ntk records, which needs UMA weights to produce for real.
    _, _, provenance = _member_artifacts(member)
    np.savez(member / 'ntk_basis.npz', metadata=json.dumps({'fit_provenance': provenance,
                                                          'training_structure_count': 1}))
    (member / 'uma_lora.pt').write_bytes(b'other factors')
    with pytest.raises(ValueError, match='does not match'):
        load_uma_ntk_estimator(member)
