"""Fit an NTK posterior for one frozen potential and report its uncertainty to ALF."""

from io import BytesIO
import json
import math

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.ntk.adapter import gaussian_probes
from alframework.ml_interfaces.ntk.posterior import Posterior
from alframework.ml_interfaces.ntk.rows import RowRequest, compute_rows
from alframework.ml_interfaces.ntk.sketch import Sketch

BASIS_VERSION = 2
# Offsets that keep the probe streams of different quantities independent.
_FORCE_PROBES, _HVP_PROBES, _BILINEAR_U, _BILINEAR_V, _FIT_PROBES = range(5)


def read_basis_metadata(path):
    """The provenance recorded in a saved NTK basis, for example its precision."""
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(saved["metadata"].item())


def _positive(name, value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _count(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


class NTKEstimator:
    """A sketched NTK posterior fit on the training geometries of one frozen potential.

    Each training geometry adds the rows its training loss supervised, divided by that quantity's
    observation scale. Use the trained model's validation RMSE of each quantity as its scale.

    - Energy rows of every state, divided by ``energy_scale`` (eV). An energy-only fit is the AdaptNTK basis.
    - Force-component rows, when ``force_scale`` (eV/Å) is given.
    - Hessian-vector-product rows, when ``hessian_scale`` (eV/Å²) is given and the geometry has Hessian
      supervision. ``atoms.info["hessian_probes"]`` ``[k, N, 3]`` gives the fixed probes of
      Projected Hessian Learning's fixed scheme. ``atoms.info["hessian"]`` marks a stored full Hessian,
      for the randomized scheme or full-Hessian training. Its information is estimated from
      ``hessian_fit_probes`` fresh Gaussian probes, using E[v vᵀ] = I.

    Refit when the model changes. Scores are raw posterior standard deviations; calibrate them against
    errors before using them as error estimates.
    """

    @torch.inference_mode(False)
    def __init__(self, adapter, training_geometries=(), *, energy_scale=1.0, force_scale=None,
                 hessian_scale=None, hessian_fit_probes=4, sketch_dim=64, seed=0, ridge=None,
                 ridge_factor=1e-3, engine="auto", probe_seed=0, sketch_store=None):
        self.adapter = adapter
        self.energy_scale = _positive("energy_scale", energy_scale)
        self.force_scale = _positive("force_scale", force_scale)
        self.hessian_scale = _positive("hessian_scale", hessian_scale)
        self.hessian_fit_probes = _count("hessian_fit_probes", hessian_fit_probes)
        self.engine, self.probe_seed, self.seed = engine, int(probe_seed), int(seed)
        self.sketch = Sketch(adapter.parameter_shapes(), sketch_dim, seed, device=adapter.device,
                             dtype=adapter.dtype, store=sketch_store)
        self.posterior = Posterior(sketch_dim, device=adapter.device, dtype=adapter.dtype)
        self.training_structure_count = 0
        if training_geometries is None:  # restored from a saved basis
            return
        for index, atoms in enumerate(training_geometries):
            self._fit_one(index, atoms)
        self.posterior.finalize(ridge=ridge, ridge_factor=ridge_factor)
        self.basis_id = self.posterior.basis_id(self.metadata)

    # -- properties ------------------------------------------------------------------------------------
    @property
    def device(self):
        return self.adapter.device

    @property
    def dtype(self):
        return self.adapter.dtype

    @property
    def ridge(self):
        return self.posterior.ridge

    @property
    def state_names(self):
        return tuple(self.adapter.state_names)

    @property
    def energy_only(self):
        return set(self.posterior.counts) == {"energy"}

    @property
    def metadata(self):
        return {
            "version": BASIS_VERSION, "adapter": type(self.adapter).__name__, "model": self.adapter.digest(),
            "sketch": self.sketch.digest, "seed": self.seed, "sketch_dim": self.sketch.width,
            "dtype": str(self.dtype).split(".")[-1], "ridge": self.ridge, "states": list(self.state_names),
            "scales": {"energy": self.energy_scale, "force": self.force_scale, "hessian": self.hessian_scale},
            "rows": dict(self.posterior.counts), "training_structure_count": self.training_structure_count,
        }

    # -- fitting ---------------------------------------------------------------------------------------
    def structure_from_atoms(self, atoms):
        periodic = bool(np.any(atoms.pbc))
        return self.adapter.structure(atoms.numbers, atoms.positions,
                                      np.asarray(atoms.cell) if periodic else None, atoms.pbc)

    def _fit_one(self, index, atoms):
        structure = self.structure_from_atoms(atoms)
        hvp_probes, hessian_scale = None, None
        if self.hessian_scale is not None:
            if "hessian_probes" in atoms.info:
                hvp_probes = torch.as_tensor(np.asarray(atoms.info["hessian_probes"], dtype=float))
                if hvp_probes.ndim != 3 or hvp_probes.shape[1:] != (len(atoms), 3):
                    raise ValueError(f"hessian_probes must have shape [k, {len(atoms)}, 3].")
                hvp_probes = hvp_probes.to(dtype=self.dtype).to(self.device)
                hessian_scale = self.hessian_scale
            elif "hessian" in atoms.info and self.hessian_fit_probes:
                k = self.hessian_fit_probes
                hvp_probes = gaussian_probes(k, len(atoms), seed=self._probe_seed(_FIT_PROBES, index),
                                             device=self.device, dtype=self.dtype)
                # The mean of k probes' J_{Hv}ᵀ J_{Hv} estimates J_Hᵀ J_H.
                hessian_scale = self.hessian_scale * math.sqrt(k)
        request = RowRequest(energy=True, forces=self.force_scale is not None, hvp_probes=hvp_probes)
        _, rows = compute_rows(self.adapter, structure, request, self.sketch, engine=self.engine)
        self.posterior.add(rows["energy"], self.energy_scale, "energy")
        if self.force_scale is not None:
            self.posterior.add(rows["forces"], self.force_scale, "force")
        if hvp_probes is not None:
            self.posterior.add(rows["hvp"], hessian_scale, "hessian")
        self.training_structure_count += 1

    def _probe_seed(self, kind, salt):
        return np.random.SeedSequence([self.probe_seed, kind, int(salt)]).generate_state(1, np.uint64)[0]

    # -- saving ----------------------------------------------------------------------------------------
    def save_basis(self, path):
        """Atomically save the information matrix, ridge, and provenance, without model weights."""
        from alframework.tools.files import write_bytes_atomic

        buffer = BytesIO()
        np.savez(buffer, metadata=json.dumps(self.metadata), basis_id=self.basis_id,
                 information=self.posterior.information.cpu().numpy())
        write_bytes_atomic(path, buffer.getvalue())

    @classmethod
    @torch.inference_mode(False)
    def from_basis(cls, adapter, path, *, engine="auto", probe_seed=0):
        """Restore a saved fit for ``adapter``, which must hold the same model snapshot."""
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            basis = saved["basis_id"].item()
            information = saved["information"]
        if metadata.get("version") != BASIS_VERSION:
            raise ValueError("Unsupported NTK basis format. Refit the basis with this version of ALF.")
        width = metadata["sketch_dim"]
        if information.shape != (width, width) or not np.isfinite(information).all():
            raise ValueError("Invalid NTK information matrix.")
        if metadata["adapter"] != type(adapter).__name__ or metadata["model"] != adapter.digest():
            raise ValueError("The NTK basis does not match this model snapshot.")
        if tuple(metadata["states"]) != tuple(adapter.state_names):
            raise ValueError("The NTK basis was fit for different model states.")
        scales = metadata["scales"]
        estimator = cls(adapter, None, energy_scale=scales["energy"], force_scale=scales["force"],
                        hessian_scale=scales["hessian"], sketch_dim=width, seed=metadata["seed"],
                        engine=engine, probe_seed=probe_seed)
        if metadata["sketch"] != estimator.sketch.digest:
            raise ValueError("The NTK basis does not match this model's parameter sketch.")
        estimator.posterior.information = torch.from_numpy(information).to(
            dtype=estimator.dtype).to(estimator.device)
        estimator.posterior.counts.update(metadata["rows"])
        estimator.posterior.finalize(ridge=metadata["ridge"])
        estimator.training_structure_count = metadata["training_structure_count"]
        recorded = estimator.metadata
        if recorded["dtype"] != metadata["dtype"]:
            raise ValueError("Restore the NTK basis with the precision it was fit in.")
        estimator.basis_id = estimator.posterior.basis_id(recorded)
        if estimator.basis_id != basis:
            raise ValueError("Invalid NTK basis checksum.")
        return estimator

    # -- scoring ---------------------------------------------------------------------------------------
    @torch.inference_mode(False)
    def score(self, structure, *, forces=True, force_probes=0, hessian_probes=0, hvp_probes=0):
        """Predictions and raw posterior standard deviations for one :class:`Structure`.

        - ``forces``: rows for every force component, which give per-atom scores.
        - ``force_probes``, ``hessian_probes``: k Gaussian projections that estimate the RMS force and
          Hessian-element uncertainty, with about k rows each.
        - ``hvp_probes``: rows for H v, giving per-atom Hessian scores. That's 3N rows per probe.

        Outputs have a leading state axis.
        """
        n = structure.n_atoms
        k_f, k_h, k_v = (_count(name, value) for name, value in
                         (("force_probes", force_probes), ("hessian_probes", hessian_probes),
                          ("hvp_probes", hvp_probes)))

        def probes(kind, count):
            return gaussian_probes(count, n, seed=self._probe_seed(kind, n), device=self.device,
                                   dtype=self.dtype) if count else None

        bilinear = (probes(_BILINEAR_U, k_h), probes(_BILINEAR_V, k_h)) if k_h else None
        request = RowRequest(energy=True, forces=bool(forces), force_probes=probes(_FORCE_PROBES, k_f),
                             hvp_probes=probes(_HVP_PROBES, k_v), bilinear_probes=bilinear, force_values=True)
        values, rows = compute_rows(self.adapter, structure, request, self.sketch, engine=self.engine)
        variance = self.posterior.variance
        result = {"energy": values["energy"], "forces": values["forces"],
                  "energy_uncertainty_score": variance(rows["energy"]).sqrt()}
        if len(self.state_names) > 1:
            result["gap_uncertainty_score"] = variance(rows["energy"][1:] - rows["energy"][:-1]).sqrt()
        if forces:
            component = variance(rows["forces"])
            result["force_component_uncertainty_score"] = component.sqrt()
            result["force_uncertainty_score"] = component.sum(-1).sqrt()
            result["max_force_uncertainty_score"] = result["force_uncertainty_score"].amax(-1)
            result["force_rms_uncertainty"] = (component.sum((-1, -2)) / (3 * n)).sqrt()
        elif k_f:
            result["force_rms_uncertainty"] = (variance(rows["force_projection"]).mean(-1) / (3 * n)).sqrt()
        if k_h:
            # E_{u,v}[(uᵀ δH v)²] = ‖δH‖_F², so this is the RMS uncertainty of one Hessian element.
            result["hessian_rms_uncertainty"] = variance(rows["bilinear"]).mean(-1).sqrt() / (3 * n)
        if k_v:
            result["hvp_uncertainty_score"] = variance(rows["hvp"]).sum(-1).mean(1).sqrt()
        return result

    @torch.inference_mode(False)
    def predict(self, atoms, **options):
        return self.score(self.structure_from_atoms(atoms), **options)

    def predict_energy_forces(self, atoms):
        """Predictions and force-component scores, without the state axis for a one-state model."""
        result = self.predict(atoms, forces=True)
        if len(self.state_names) == 1:
            result = {key: value[0] for key, value in result.items()}
        return result

    @torch.inference_mode(False)
    def candidate_features(self, numbers, positions, pbc, cell=None, *, state=0):
        """The whitened energy row of one state, its basis ID, and the ridge, for AdaptNTK selection."""
        structure = self.adapter.structure(numbers, positions, cell, pbc)
        _, rows = compute_rows(self.adapter, structure, RowRequest(energy=True), self.sketch, engine=self.engine)
        whitened = self.posterior.whiten(rows["energy"][state])
        if not torch.isfinite(whitened).all():
            raise FloatingPointError("Nonfinite whitened energy NTK row.")
        return {"row": whitened.cpu().numpy(), "basis_id": self.basis_id, "ridge": self.ridge}


def reported_uncertainty(result, state_names, n_atoms, periodic=False):
    """ALF score names: scalars for acquisition, and atom arrays for selected frames."""
    scalar, atomic = {}, {}
    multi = len(state_names) > 1
    for index, state in enumerate(state_names):
        suffix = f"_{state}" if multi else ""
        energy = float(result["energy_uncertainty_score"][index])
        scalar[f"ntk_energy{suffix}"] = energy
        if periodic:
            scalar[f"ntk_energy_per_atom{suffix}"] = energy / n_atoms
        for key, name in (("max_force_uncertainty_score", "ntk_force_max"), ("force_rms_uncertainty", "ntk_force_rms"),
                          ("hessian_rms_uncertainty", "ntk_hessian")):
            if key in result:
                scalar[f"{name}{suffix}"] = float(result[key][index])
        for key, name in (("force_uncertainty_score", "ntk_force"), ("hvp_uncertainty_score", "ntk_hvp")):
            if key in result:
                atomic[f"{name}{suffix}"] = result[key][index].detach().cpu().numpy()
    if "gap_uncertainty_score" in result:
        for index, value in enumerate(result["gap_uncertainty_score"]):
            scalar[f"ntk_dE{index}{index + 1}"] = float(value)
    if not all(math.isfinite(value) for value in scalar.values()):
        raise FloatingPointError("Nonfinite NTK uncertainty score.")
    return scalar, atomic


class NTKASECalculator(Calculator):
    """Report one fitted estimator through ALF's ASE uncertainty contract.

    Energy and forces are those of ``selected_state``. ``scores`` are passed to :meth:`NTKEstimator.score`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, estimator, *, selected_state=0, **scores):
        super().__init__()
        self.estimator, self.selected_state, self.scores = estimator, selected_state, scores

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        result = self.estimator.predict(self.atoms, **self.scores)
        scalar, atomic = reported_uncertainty(result, self.estimator.state_names, len(self.atoms),
                                              bool(np.any(self.atoms.pbc)))
        self.results = {
            "energy": float(result["energy"][self.selected_state]),
            "forces": result["forces"][self.selected_state].cpu().numpy(),
            "alf_uncertainty": scalar,
            "alf_atomic_uncertainty": atomic,
        }


class NTKPackedEvaluator:
    """One fitted snapshot for packed MD forces, with NTK scores only at acquisition checks.

    An energy-only (AdaptNTK) estimator reports the ``energy_ntk`` payload for adaptive cohort selection.
    ``force_diagnostics`` can add the per-atom force scores of a second estimator on the same model,
    device, and precision. Any other estimator reports the scores selected by ``scores``.
    """

    n_models = 1

    def __init__(self, estimator, *, force_diagnostics=None, selected_state=0, **scores):
        self.estimator, self.force_diagnostics = estimator, force_diagnostics
        self.selected_state, self.scores = selected_state, scores
        if force_diagnostics is not None:
            if (not estimator.energy_only or force_diagnostics.force_scale is None
                    or estimator.device != force_diagnostics.device or estimator.dtype != force_diagnostics.dtype
                    or estimator.metadata["model"] != force_diagnostics.metadata["model"]):
                raise ValueError("Force diagnostics require a force-fitted estimator on the same model, "
                                 "device, and dtype as an energy-only estimator.")

    @torch.inference_mode(False)
    def evaluate(self, numbers, positions, cells, pbc, counts):
        """Return ``[1, systems]`` energies and ``[1, atoms, 3]`` forces of the selected state, unscored."""
        if not counts or min(counts) <= 0 or sum(counts) != len(numbers) or positions.shape != (len(numbers), 3):
            raise ValueError("Packed atom counts must match the nonempty input structures.")
        energies, forces = self.estimator.adapter.predict_batch(numbers, positions, cells, pbc, counts)
        state = self.selected_state
        return energies[state][None], forces[state][None]

    @torch.inference_mode(False)
    def observe_uncertainty(self, numbers, positions, cell, pbc):
        """Return scores, and any energy row or atom diagnostics, for one structure."""
        estimator = self.estimator
        periodic = bool(torch.as_tensor(pbc).any())
        if estimator.energy_only:
            payload = estimator.candidate_features(numbers, positions, pbc, cell if periodic else None,
                                                   state=self.selected_state)
            row = payload["row"].astype(np.float64, copy=False)
            report = {"energy_ntk": payload, "uncertainty": {"ntk_energy": math.sqrt(payload["ridge"] * float(row @ row))}}
            if self.force_diagnostics is not None:
                model = self.force_diagnostics
                result = model.score(model.adapter.structure(numbers, positions, cell if periodic else None, pbc))
                scalar, report["atomic_uncertainty"] = reported_uncertainty(
                    result, model.state_names, len(numbers), periodic)
                report["uncertainty"].update({k: v for k, v in scalar.items() if k.startswith("ntk_force")})
            return report
        structure = estimator.adapter.structure(numbers, positions, cell if periodic else None, pbc)
        result = estimator.score(structure, **self.scores)
        scalar, atomic = reported_uncertainty(result, estimator.state_names, len(numbers), periodic)
        return {"uncertainty": scalar, "atomic_uncertainty": atomic}
