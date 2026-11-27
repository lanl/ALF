"""The NTK core on a pair potential that is linear in its weights, against closed forms."""

import json

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from alframework.ml_interfaces.ntk import NTKASECalculator, NTKEstimator, RowRequest, Sketch, compute_rows
from alframework.ml_interfaces.ntk.adapter import NTKAdapter, gaussian_probes

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


@pytest.mark.parametrize("engine", ["forward", "reverse"])
def test_scores_equal_closed_form_bayesian_linear_regression(engine):
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
    f, g, h = features(query)

    def expected(rows):
        return 0.05 * (rows * torch.linalg.solve(precision, rows.T).T).sum(-1)

    result = estimator.predict(query, forces=True)
    torch.testing.assert_close(result["energy_uncertainty_score"][0], expected((S @ f)[None])[0].sqrt())
    torch.testing.assert_close(result["force_component_uncertainty_score"][0],
                               expected(g.reshape(-1, 5) @ S.T).reshape(3, 3).sqrt())
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(query),
                           RowRequest(hvp_probes=torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)),
                           estimator.sketch, engine=engine)
    # Row k of the identity probes is Hessian column k.
    torch.testing.assert_close(rows["hvp"][0].reshape(9, 9, 8), (h @ S.T).transpose(0, 1))


@pytest.mark.parametrize("engine", ["forward", "reverse"])
def test_selected_force_rows_and_total_energy_match_full_scores(engine, monkeypatch):
    estimator = NTKEstimator(LinearPairAdapter(), [molecule(1)], energy_scale=0.3, force_scale=0.7,
                             sketch_dim=8, engine=engine)
    structure = estimator.structure_from_atoms(molecule(5))
    full = estimator.score(structure)
    requests = []

    def record_rows(adapter, structure, request, *args, **kwargs):
        values, rows = compute_rows(adapter, structure, request, *args, **kwargs)
        requests.append((request.output_count(1, structure.n_atoms), rows["forces"].shape))
        return values, rows

    monkeypatch.setattr("alframework.ml_interfaces.ntk.estimator.compute_rows", record_rows)
    selected = estimator.score_forces(structure, atom_indices=[2, 0], include_energy=True)
    assert requests == [(7, (1, 2, 3, 8))]
    torch.testing.assert_close(selected["energy_uncertainty_score"], full["energy_uncertainty_score"])
    torch.testing.assert_close(selected["force_uncertainty_score"], full["force_uncertainty_score"][:, [2, 0]])
    assert selected["force_rms_uncertainty"] == pytest.approx(
        full["force_component_uncertainty_score"][:, [2, 0]].square().mean().sqrt().item())
    with pytest.raises(ValueError, match="requires exact"):
        estimator.score_forces(structure, atom_indices=[0], force_probes=8)


def test_projections_estimate_the_exact_force_and_hessian_uncertainty():
    estimator = NTKEstimator(LinearPairAdapter(), [molecule(1)], energy_scale=0.3, force_scale=0.7,
                             sketch_dim=5, seed=4, ridge=0.05)
    query = molecule(5)
    exact = estimator.predict(query, forces=True)
    structure = estimator.structure_from_atoms(query)
    identity = torch.eye(9, dtype=torch.float64).reshape(9, 3, 3)
    _, rows = compute_rows(estimator.adapter, structure, RowRequest(hvp_probes=identity), estimator.sketch)
    exact_hessian = (estimator.posterior.variance(rows["hvp"][0]).sum() ** 0.5) / 9
    sampled = estimator.predict(query, forces=False, force_probes=4000, hessian_probes=4000)
    assert sampled["force_rms_uncertainty"][0] == pytest.approx(float(exact["force_rms_uncertainty"][0]), rel=0.1)
    assert sampled["hessian_rms_uncertainty"][0] == pytest.approx(float(exact_hessian), rel=0.1)


def test_training_geometries_have_small_noiseless_variance():
    training = [molecule(seed) for seed in range(4)]
    estimator = NTKEstimator(LinearPairAdapter(), training, energy_scale=1.0, sketch_dim=5, seed=1, ridge=1e-9)
    # Five features, four energies: the span misses one direction, but training energies lie in it.
    for atoms in training:
        assert estimator.predict(atoms, forces=False)["energy_uncertainty_score"][0] < 1e-3
    loose = estimator.posterior.with_ridge(1.0)
    assert loose.ridge == 1.0 and loose.information is estimator.posterior.information


def test_sketch_chunks_regenerate_identically_and_stay_stable():
    shapes = {"a": (7, 3), "b": (5,)}
    stored = Sketch(shapes, 10, 3, chunk_size=4, store=True)
    regenerated = Sketch(shapes, 10, 3, chunk_size=4, store=False)
    assert stored.n_chunks == 3 and stored.digest == regenerated.digest
    full = sketch_matrix(stored)
    assert full.shape == (10, 26)
    torch.testing.assert_close(sketch_matrix(regenerated), full)
    torch.testing.assert_close(torch.cat([stored.tangents(c)["a"].reshape(-1, 21) for c in range(3)]), full[:, :21])
    assert Sketch(shapes, 10, 4, chunk_size=4).digest != stored.digest
    # Entries are N(0, 1/p).
    large = sketch_matrix(Sketch({"w": (20000,)}, 16, 0))
    assert float(large.var() * 16) == pytest.approx(1.0, rel=0.02)
    gradients = torch.randn(3, 26, dtype=torch.float64)
    torch.testing.assert_close(stored.project(gradients), gradients @ full.T)


def test_probes_are_seeded_and_unnormalized():
    probes = gaussian_probes(5000, 4, seed=3, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(probes, gaussian_probes(5000, 4, seed=3, device="cpu", dtype=torch.float64))
    second_moment = torch.einsum("ki,kj->ij", probes.reshape(5000, 12), probes.reshape(5000, 12)) / 5000
    torch.testing.assert_close(second_moment, torch.eye(12, dtype=torch.float64), atol=0.08, rtol=0)


def test_calculator_reports_probe_scores_and_basis_round_trip(tmp_path):
    estimator = NTKEstimator(LinearPairAdapter(), [molecule(1)], energy_scale=0.3, force_scale=0.7, sketch_dim=5)
    calculator = NTKASECalculator(estimator, forces=False, force_probes=3, hessian_probes=3, hvp_probes=1)
    calculator.calculate(molecule(2))
    assert set(calculator.results["alf_uncertainty"]) == {"ntk_energy", "ntk_force_rms", "ntk_hessian"}
    assert set(calculator.results["alf_atomic_uncertainty"]) == {"ntk_hvp"}
    path = tmp_path / "basis.npz"
    estimator.save_basis(path)
    restored = NTKEstimator.from_basis(LinearPairAdapter(), path)
    assert restored.basis_id == estimator.basis_id
    with pytest.raises(ValueError, match="model snapshot"):
        changed = LinearPairAdapter()
        changed.digest = lambda: "other"
        NTKEstimator.from_basis(changed, path)


@pytest.mark.parametrize("with_outputs", [False, True], ids=["previous_format", "empty_output_scales"])
def test_version_two_basis_preserves_its_metadata_and_id(tmp_path, with_outputs):
    fitted = NTKEstimator(LinearPairAdapter(), [molecule(1)], energy_scale=0.3, force_scale=0.7, sketch_dim=5)
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


def test_adiabatic_hessian_scores_diverge_at_a_crossing_while_smooth_scores_do_not():
    training = [bond(0.7), bond(1.8)]
    hessian, smooth, gap = [], [], []
    # Couplings small next to the training geometries' gaps, so the fit is the same for all of them.
    for coupling in (1e-2, 1e-3, 1e-4):
        estimator = NTKEstimator(TwoStateAdapter(coupling), training, energy_scale=0.1, sketch_dim=2, ridge=1e-3)
        result = estimator.predict(bond(1.2), forces=False, hessian_probes=64, outputs=("diabatic",))
        hessian.append(float(result["hessian_rms_uncertainty"].max()))
        smooth.append(result["output_diabatic_uncertainty_score"])
        gap.append(float(result["gap_uncertainty_score"][0]))
        assert result["gap"][0] == pytest.approx(2 * coupling, rel=1e-6)
    assert hessian[1] > 5 * hessian[0] and hessian[2] > 5 * hessian[1]
    for later in smooth[1:]:
        torch.testing.assert_close(later, smooth[0], rtol=1e-2, atol=0)
    assert all(np.isfinite(gap))


def test_crossing_flags_mark_states_next_to_a_small_gap():
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
