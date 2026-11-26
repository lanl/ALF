"""Numerical checks for NTK uncertainty on HIPPYNN graphs: molecular, periodic, and multi-state."""

import copy
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")

from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK, EnergyNTK, NTKASECalculator, NTKPackedEvaluator
from alframework.ml_interfaces.ntk import RowRequest, compute_rows
from alframework.ml_interfaces.ntk import estimator as estimator_module
from alframework.ml_interfaces.ntk.adapter import gaussian_probes
from alframework.samplers.adaptive_ntk import initial_energy_score
from alframework.samplers.engines.packed import PackedMLMDEngine
from alframework.testing import check_evaluator

NETWORK = {"n_features": 8, "n_sensitivities": 4, "n_interaction_layers": 1, "n_atom_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels(False)
    try:
        yield
    finally:
        custom_kernels.set_custom_kernels(previous)


def molecular_graph(kind, n_states=1):
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
    with torch.random.fork_rng():
        torch.manual_seed(3)
        species, positions, cell = inputs.SpeciesNode("species"), inputs.PositionsNode("positions"), inputs.CellNode("cell")
        network = networks.Hipnn("hipnn", (species, positions, cell), periodic=True, module_kwargs={
            "possible_species": [0, 14], "dist_soft_min": 1.0, "dist_soft_max": 3.5, "dist_hard_max": 4.0,
            **NETWORK})
        energy = system_energy(targets.HEnergyNode("energy", network))
        return GraphModule((species, positions, cell), (energy,)).double().eval()


@pytest.fixture(params=["Hipnn", "HipHopnn"])
def graph(request):
    return molecular_graph(request.param)


@pytest.fixture
def atoms():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])


def silicon(rattle_seed=1):
    cell = bulk("Si", "diamond", a=5.43, cubic=True)
    cell.rattle(0.05, seed=rattle_seed)
    return cell


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


@pytest.mark.parametrize("periodic", [False, True])
def test_rows_match_parameter_finite_differences_and_both_engines(graph, atoms, periodic):
    if periodic:
        graph, atoms = periodic_graph(), silicon()
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    (u, v), values, rows = all_rows(estimator, atoms, "forward")
    _, reverse_values, reverse = all_rows(estimator, atoms, "reverse")
    for key in rows:
        torch.testing.assert_close(reverse[key], rows[key], rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(reverse_values[key], values[key])
    plus = perturbed_predictions(estimator, graph, atoms, 1e-5, v[0])
    minus = perturbed_predictions(estimator, graph, atoms, -1e-5, v[0])
    difference = [(a - b) / 2e-5 for a, b in zip(plus, minus)]
    torch.testing.assert_close(rows["energy"][0, 0], difference[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows["forces"][0, ..., 0], difference[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows["hvp"][0, 0, ..., 0], difference[2], atol=1e-5, rtol=1e-4)
    # Projections are the same linear maps of the component rows.
    torch.testing.assert_close(rows["force_projection"][0], torch.einsum("knc,ncp->kp", u, rows["forces"][0]))
    torch.testing.assert_close(rows["bilinear"][0], torch.einsum("knc,kncp->kp", u, rows["hvp"][0]))


def test_bilinear_hessian_rows_are_symmetric(graph, atoms):
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16)
    structure = estimator.structure_from_atoms(atoms)
    u = gaussian_probes(3, 3, seed=4, device="cpu", dtype=torch.float64)
    v = gaussian_probes(3, 3, seed=5, device="cpu", dtype=torch.float64)
    _, uv = compute_rows(estimator.adapter, structure, RowRequest(bilinear_probes=(u, v)), estimator.sketch)
    _, vu = compute_rows(estimator.adapter, structure, RowRequest(bilinear_probes=(v, u)), estimator.sketch)
    torch.testing.assert_close(uv["bilinear"], vu["bilinear"], rtol=1e-9, atol=1e-11)


def test_streamed_ntk_scores_match_direct_solve_and_preserve_atom_order(graph, atoms):
    stretched = atoms.copy()
    stretched.positions[1] *= 1.15
    training = [atoms, stretched]
    energy_scale, force_scale = 0.2, 0.8
    estimator = EnergyForceNTK(graph, iter(training), energy_scale=energy_scale, force_scale=force_scale,
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
    permutation = [2, 0, 1]
    reordered = estimator.predict_energy_forces(stretched[permutation])
    for key in ("forces", "force_component_uncertainty_score", "force_uncertainty_score"):
        torch.testing.assert_close(reordered[key], result[key][permutation])
    assert estimator.training_structure_count == len(training)
    assert dict(estimator.posterior.counts) == {"energy": 2, "force": 18}


def test_inference_contexts_and_graph_periodicity_guard(graph, atoms):
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
    result = estimator.predict_energy_forces(atoms)
    assert result["max_force_uncertainty_score"] > 0
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            fitted = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8)
            observed = fitted.predict_energy_forces(atoms)
        for key, expected in result.items():
            torch.testing.assert_close(observed[key], expected)
    calculator = NTKASECalculator(estimator)
    calculator.calculate(atoms)
    np.testing.assert_allclose(calculator.results["alf_atomic_uncertainty"]["ntk_force"],
                               result["force_uncertainty_score"])
    periodic = atoms.copy()
    periodic.set_cell([6.0] * 3)
    periodic.set_pbc(True)
    with pytest.raises(ValueError, match="nonperiodic"):
        calculator.calculate(periodic)
    assert not calculator.results
    with pytest.raises(ValueError, match="fully periodic"):
        EnergyForceNTK(periodic_graph(), [atoms], energy_scale=0.2, force_scale=0.8)


def test_periodic_scores_are_invariant_and_extensive():
    graph = periodic_graph()
    cell = silicon()
    estimator = EnergyForceNTK(graph, [cell, silicon(2)], energy_scale=0.05, force_scale=0.3, sketch_dim=16)
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


@pytest.mark.parametrize("device", list(dict.fromkeys(("cpu", os.environ.get("ALF_TEST_NTK_DEVICE", "cpu")))))
def test_float32_matches_float64_reference(graph, atoms, device):
    training = [atoms, atoms.copy()]
    training[1].positions[1] *= 1.15
    query = atoms.copy()
    query.positions[2] *= 1.1
    scales = dict(energy_scale=0.2, force_scale=0.8, sketch_dim=16, seed=13)
    options = dict(forces=True, force_probes=4, hessian_probes=4, hvp_probes=1)
    ref = EnergyForceNTK(graph, training, **scales).predict(query, **options)
    # The GPU case exercises the default precision. An explicitly requested GPU
    # must be usable so the qualification command cannot pass by skipping it.
    dtype = torch.float32 if device == "cpu" else None
    estimator = EnergyForceNTK(graph, training, **scales, device=device, dtype=dtype)
    assert estimator.dtype == torch.float32
    for value in (*estimator.adapter.live_parameters().values(), estimator.sketch.flat_chunk(0),
                  estimator.posterior.factor):
        assert value.device.type == torch.device(device).type
    with torch.inference_mode():
        observed = estimator.predict(query, **options)
    for key, expected in ref.items():
        assert observed[key].device.type == torch.device(device).type
        torch.testing.assert_close(observed[key].cpu().double(), expected, rtol=3e-3, atol=5e-5)


def test_hessian_supervision_adds_rows_and_lowers_hessian_variance(graph, atoms):
    stretched = atoms.copy()
    stretched.positions[1] *= 1.1
    scales = dict(energy_scale=0.2, force_scale=0.8, sketch_dim=32, seed=5)
    plain = EnergyForceNTK(graph, [atoms, stretched], **scales)
    fixed = [frame.copy() for frame in (atoms, stretched)]
    probes = gaussian_probes(2, 3, seed=9, device="cpu", dtype=torch.float64).numpy()
    for frame in fixed:
        frame.info["hessian_probes"] = probes
    with_fixed = EnergyForceNTK(graph, fixed, **scales, hessian_scale=0.5)
    full = [frame.copy() for frame in (atoms, stretched)]
    for frame in full:
        frame.info["hessian"] = np.zeros((9, 9))
    with_full = EnergyForceNTK(graph, full, **scales, hessian_scale=0.5, hessian_fit_probes=3)
    assert with_fixed.posterior.counts["hessian"] == 2 * 2 * 9
    assert with_full.posterior.counts["hessian"] == 2 * 3 * 9
    assert "hessian" not in plain.posterior.counts
    structure = plain.structure_from_atoms(atoms)
    request = RowRequest(hvp_probes=torch.as_tensor(probes))
    _, rows = compute_rows(plain.adapter, structure, request, plain.sketch)
    before = plain.posterior.variance(rows["hvp"]).sum()
    for estimator in (with_fixed, with_full):
        assert estimator.posterior.variance(rows["hvp"]).sum() < 0.5 * before
    hessian = with_fixed.predict(atoms, hessian_probes=8)["hessian_rms_uncertainty"]
    assert hessian < plain.predict(atoms, hessian_probes=8)["hessian_rms_uncertainty"]


def test_hessian_rows_come_only_from_supervised_states(atoms):
    graph = molecular_graph("Hipnn", n_states=2)
    probes = gaussian_probes(2, 3, seed=9, device="cpu", dtype=torch.float64)
    scales = dict(energy_scale=0.2, force_scale=0.8, hessian_scale=0.1, sketch_dim=32, seed=5)
    fits = {}
    for states in (None, ["E1"], ["E0", "E1"]):
        frame = atoms.copy()
        frame.info["hessian_probes"] = probes.numpy()
        if states is not None:
            frame.info["hessian_states"] = states
        fits[None if states is None else tuple(states)] = EnergyForceNTK(graph, [frame], **scales)
    assert fits[None].posterior.counts["hessian"] == fits[("E1",)].posterior.counts["hessian"] == 2 * 9
    assert fits[("E0", "E1")].posterior.counts["hessian"] == 2 * 2 * 9
    _, rows = compute_rows(fits[None].adapter, fits[None].structure_from_atoms(atoms), RowRequest(hvp_probes=probes),
                           fits[None].sketch)
    for state, other in ((0, ("E1",)), (1, None)):
        # Supervising a state's Hessian lowers that state's HVP variance more than the other state's.
        own = fits[None if state == 0 else ("E1",)].posterior.variance(rows["hvp"][state]).sum()
        assert own < fits[other].posterior.variance(rows["hvp"][state]).sum()
    frame = atoms.copy()
    frame.info.update(hessian_probes=probes.numpy(), hessian_states=["E7"])
    with pytest.raises(ValueError, match="not model states"):
        EnergyForceNTK(graph, [frame], **scales)


def test_multistate_graph_reports_state_and_gap_scores(atoms):
    graph = molecular_graph("Hipnn", n_states=2)
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16)
    assert estimator.state_names == ("E0", "E1")
    result = estimator.predict(atoms, forces=True, hessian_probes=2)
    _, rows = compute_rows(estimator.adapter, estimator.structure_from_atoms(atoms), RowRequest(), estimator.sketch)
    torch.testing.assert_close(result["gap_uncertainty_score"],
                               estimator.posterior.variance(rows["energy"][1] - rows["energy"][0]).sqrt()[None])
    calculator = NTKASECalculator(estimator, selected_state=1, hessian_probes=2)
    calculator.calculate(atoms)
    scores = calculator.results["alf_uncertainty"]
    assert {"ntk_energy_E0", "ntk_energy_E1", "ntk_force_max_E1", "ntk_hessian_E0", "ntk_dE01"} <= set(scores)
    assert set(calculator.results["alf_atomic_uncertainty"]) == {"ntk_force_E0", "ntk_force_E1"}
    assert calculator.results["energy"] == pytest.approx(float(result["energy"][1]))


def test_packed_ntk_scores_are_lazy_and_follow_active_atom_order(graph, atoms, monkeypatch):
    second = Atoms("H2", positions=[[0., 0., 0.], [1.3, 0., 0.]])
    estimator = EnergyForceNTK(graph, [atoms, second], energy_scale=0.2, force_scale=0.8, dtype=torch.float32)
    evaluator = NTKPackedEvaluator(estimator)
    score_calls = []
    observe = evaluator.observe_uncertainty

    def record_observe(numbers, positions, cell, pbc):
        score_calls.append(numbers.clone())
        return observe(numbers, positions, cell, pbc)

    class TestPackedEngine(PackedMLMDEngine):
        def _baoab(self):
            # Change the current geometry without a dependency on an integrator.
            self.state.positions *= 1.001
            return self._evaluate()

    monkeypatch.setattr(evaluator, "observe_uncertainty", record_observe)
    engine = TestPackedEngine([atoms.copy(), second.copy()], evaluator, 0.5, [300., 300.],
                              [np.random.RandomState(2), np.random.RandomState(3)])
    # Refresh ordinary forces and screen a frame without invoking NTK scoring.
    engine.scale_cell(0, 1.02)
    assert not engine.step([0, 1], 2)
    assert "uncertainty" not in engine.observe_physical(0)["prediction"]
    assert not score_calls

    for index in (0, 1):
        snapshot = engine.atoms(index).copy()
        reference = estimator.predict_energy_forces(snapshot)
        prediction = engine.observe(index)["prediction"]
        np.testing.assert_allclose(prediction["energies"][0, 0], reference["energy"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["forces"][0][0], reference["forces"], rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(prediction["atomic_uncertainty"][0]["ntk_force"],
                                   reference["force_uncertainty_score"], rtol=3e-5, atol=1e-6)
    assert [len(numbers) for numbers in score_calls] == [3, 2]

    engine.finish(0)
    engine.scale_cell(1, 1.05)
    assert not engine.step([1], 1)
    assert len(score_calls) == 2
    observed = engine.observe(1)["prediction"]["atomic_uncertainty"][0]["ntk_force"]
    remaining = engine.atoms(1)
    np.testing.assert_array_equal(score_calls[-1], remaining.numbers)
    expected = estimator.predict_energy_forces(remaining)["force_uncertainty_score"]
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=1e-6)
    assert check_evaluator(evaluator, [remaining])[0]["ntk_force_max"] == pytest.approx(float(expected.max()))
    engine.finish(1)


def test_packed_md_keeps_the_active_kernels(graph, atoms):
    """Scoring switches to plain-PyTorch kernels and restores the previous selection afterwards."""
    estimator = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=8)
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels("pytorch")
    try:
        estimator.predict(atoms, hessian_probes=1)
        assert custom_kernels.CUSTOM_KERNELS_ACTIVE == "pytorch"
    finally:
        custom_kernels.set_custom_kernels(previous)


@pytest.mark.parametrize("device", list(dict.fromkeys(("cpu", os.environ.get("ALF_TEST_NTK_DEVICE", "cpu")))))
def test_energy_basis_round_trip_and_model_identity(graph, atoms, device, tmp_path, monkeypatch):
    stretched = atoms.copy()
    stretched.positions[1] *= 1.13
    options = dict(sketch_dim=16, seed=13, ridge=0.2)
    cpu = EnergyNTK(graph, [atoms, stretched], **options)
    requests = []
    compute = estimator_module.compute_rows

    def record(adapter, structure, request, *args, **kwargs):
        requests.append(request)
        return compute(adapter, structure, request, *args, **kwargs)

    monkeypatch.setattr(estimator_module, "compute_rows", record)
    with torch.inference_mode():
        fitted = EnergyNTK(graph, [atoms, stretched], **options, device=device)
        original = fitted.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    # Energy-only fitting and scoring never differentiate coordinates.
    assert requests and not any(request.needs_derivatives for request in requests)
    path = tmp_path / "energy_ntk.npz"
    fitted.save_basis(path)
    restored = EnergyNTK.from_basis(graph, path, device=device)
    observed = restored.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    reference = cpu.candidate_features(stretched.numbers, stretched.positions, stretched.pbc)
    assert restored.basis_id == observed["basis_id"] == fitted.basis_id
    np.testing.assert_allclose(observed["row"], original["row"], rtol=3e-5, atol=1e-6)
    assert initial_energy_score({"energy_ntk": observed}) == pytest.approx(
        initial_energy_score({"energy_ntk": reference}), rel=3e-3, abs=5e-5)
    changed = copy.deepcopy(graph)
    with torch.no_grad():
        next(p for p in changed.parameters() if p.requires_grad).add_(1.0)
    with pytest.raises(ValueError, match="model snapshot"):
        EnergyNTK.from_basis(changed, path, device=device)


def test_force_basis_with_hessian_rows_round_trips(graph, atoms, tmp_path):
    frame = atoms.copy()
    frame.info["hessian"] = np.zeros((9, 9))
    fitted = EnergyForceNTK(graph, [frame], energy_scale=0.2, force_scale=0.8, hessian_scale=0.5, sketch_dim=16)
    path = tmp_path / "ntk.npz"
    fitted.save_basis(path)
    restored = estimator_module.NTKEstimator.from_basis(fitted.adapter, path)
    assert restored.basis_id == fitted.basis_id and restored.posterior.counts == fitted.posterior.counts
    for key, value in fitted.predict(atoms, hessian_probes=2).items():
        torch.testing.assert_close(restored.predict(atoms, hessian_probes=2)[key], value)


def test_packed_energy_selection_can_also_report_force_diagnostics(graph, atoms):
    energy = EnergyNTK(graph, [atoms], sketch_dim=16)
    forces = EnergyForceNTK(graph, [atoms], energy_scale=0.2, force_scale=0.8, sketch_dim=16)
    evaluator = NTKPackedEvaluator(energy, force_diagnostics=forces)
    with torch.inference_mode():
        observation = evaluator.observe_uncertainty(
            torch.tensor(atoms.numbers), torch.tensor(atoms.positions), None, torch.tensor(atoms.pbc))
    reference = forces.predict_energy_forces(atoms)
    assert observation["energy_ntk"]["basis_id"] == energy.basis_id
    assert observation["uncertainty"]["ntk_energy"] ** 2 == pytest.approx(initial_energy_score(observation))
    assert observation["uncertainty"]["ntk_force_max"] == pytest.approx(float(reference["max_force_uncertainty_score"]))
    np.testing.assert_allclose(observation["atomic_uncertainty"]["ntk_force"], reference["force_uncertainty_score"])
