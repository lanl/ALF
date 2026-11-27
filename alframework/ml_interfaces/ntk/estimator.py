"""Fit an NTK posterior for one frozen potential and report its uncertainty to ALF."""

from copy import deepcopy
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
from alframework.tools.excited_state_schema import gap_key_for_pair

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


def _gap_pairs(pairs, n_states):
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("gap_pairs must be a nonempty list of lower and upper state pairs.")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
           any(isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)) for state in pair)
           or not 0 <= pair[0] < pair[1] < n_states for pair in pairs):
        raise ValueError(f"Each gap_pairs entry must give two ordered states below {n_states}.")
    if len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError("gap_pairs must be distinct.")
    return pairs


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
      ``hessian_fit_probes`` fresh Gaussian probes, using E[v vᵀ] = I. ``atoms.info["hessian_states"]``
      names the states whose Hessians were supervised, by default the first state.
    - Rows of non-energy model outputs, from ``output_scales`` ``{name: scale}``, for an adapter whose
      outputs were trained, such as a scaled-coupling model. Set ``energy_scale=None`` for a model with no
      energies.

    ``fit_provenance`` optionally records JSON-serializable details about the training data and fit
    in the saved basis and its checksum. Refit when the model changes. Scores are raw posterior
    standard deviations; calibrate them against errors before using them as error estimates.
    """

    @torch.inference_mode(False)
    def __init__(self, adapter, training_geometries=(), *, energy_scale=1.0, force_scale=None,
                 hessian_scale=None, hessian_fit_probes=4, sketch_dim=64, seed=0, ridge=None,
                 ridge_factor=1e-3, engine="auto", probe_seed=0, sketch_store=None, output_scales=None,
                 fit_provenance=None):
        self.adapter = adapter
        self._fit_provenance = deepcopy(fit_provenance)
        if not adapter.state_names:
            if energy_scale not in (None, 1.0) or force_scale is not None or hessian_scale is not None:
                raise ValueError("This model predicts no energies; fit it on output_scales only.")
            energy_scale = None
        self.energy_scale = _positive("energy_scale", energy_scale)
        # Earlier version 2 bases predate this field. Preserve its presence
        # on round trip so their checksums and cohort basis IDs stay valid.
        self._record_output_scales = output_scales is not None
        self.output_scales = {str(name): _positive(f"output_scales[{name!r}]", scale)
                              for name, scale in dict(output_scales or {}).items()}
        if self.energy_scale is None and not self.output_scales:
            raise ValueError("Give energy_scale or output_scales: the fit needs rows of something the model learned.")
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
        scales = {"energy": self.energy_scale, "force": self.force_scale, "hessian": self.hessian_scale}
        if self._record_output_scales:
            scales["outputs"] = dict(self.output_scales)
        metadata = {
            "version": BASIS_VERSION, "adapter": type(self.adapter).__name__, "model": self.adapter.digest(),
            "sketch": self.sketch.digest, "seed": self.seed, "sketch_dim": self.sketch.width,
            "dtype": str(self.dtype).split(".")[-1], "ridge": self.ridge, "states": list(self.state_names),
            "scales": scales,
            "rows": dict(self.posterior.counts), "training_structure_count": self.training_structure_count,
        }
        if self._fit_provenance is not None:
            metadata["fit_provenance"] = deepcopy(self._fit_provenance)
        return metadata

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
        hessian_states = None
        if hvp_probes is not None:
            names = atoms.info.get("hessian_states", self.state_names[:1])
            unknown = set(names) - set(self.state_names)
            if unknown:
                raise ValueError(f"hessian_states {sorted(unknown)} are not model states {list(self.state_names)}.")
            hessian_states = [self.state_names.index(name) for name in names]
        request = RowRequest(energy=self.energy_scale is not None, forces=self.force_scale is not None,
                             hvp_probes=hvp_probes, outputs=tuple(self.output_scales))
        _, rows = compute_rows(self.adapter, structure, request, self.sketch, engine=self.engine)
        if self.energy_scale is not None:
            self.posterior.add(rows["energy"], self.energy_scale, "energy")
        for name, scale in self.output_scales.items():
            self.posterior.add(rows[f"output:{name}"], scale, f"output:{name}")
        if self.force_scale is not None:
            self.posterior.add(rows["forces"], self.force_scale, "force")
        if hvp_probes is not None:
            self.posterior.add(rows["hvp"][hessian_states], hessian_scale, "hessian")
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
                        engine=engine, probe_seed=probe_seed, output_scales=scales.get("outputs"),
                        fit_provenance=metadata.get("fit_provenance"))
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
    def _probes(self, kind, count, n_atoms):
        return (gaussian_probes(count, n_atoms, seed=self._probe_seed(kind, n_atoms), device=self.device,
                                dtype=self.dtype) if count else None)

    def _force_uncertainty(self, rows, n_atoms):
        if "force_projection" in rows and "forces" not in rows:
            return {"force_rms_uncertainty": (self.posterior.variance(rows["force_projection"]).mean(-1)
                                              / (3 * n_atoms)).sqrt()}
        component = self.posterior.variance(rows["forces"])
        atomic = component.sum(-1).sqrt()
        return {"force_component_uncertainty_score": component.sqrt(), "force_uncertainty_score": atomic,
                "max_force_uncertainty_score": atomic.amax(-1),
                "force_rms_uncertainty": (component.sum((-1, -2)) / (3 * n_atoms)).sqrt()}

    @torch.inference_mode(False)
    def score_forces(self, structure, *, force_probes=0, atom_indices=None, include_energy=False):
        """Score exact forces (optionally on selected atoms) or estimate the frame RMS with probes.

        With ``include_energy``, also score each state's total energy using the same basis and row request.
        """
        count = _count("force_probes", force_probes)
        n_atoms = structure.n_atoms
        if atom_indices is not None:
            atom_indices = tuple(atom_indices)
            if count:
                raise ValueError("Atom selection requires exact force scoring.")
            if (not atom_indices
                    or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                           or not 0 <= i < n_atoms for i in atom_indices)
                    or len(set(atom_indices)) != len(atom_indices)):
                raise ValueError("atom_indices must contain distinct, in-range atom indices.")
            n_atoms = len(atom_indices)
        request = RowRequest(energy=include_energy, forces=count == 0, force_atom_indices=atom_indices,
                             force_probes=self._probes(_FORCE_PROBES, count, structure.n_atoms))
        _, rows = compute_rows(self.adapter, structure, request, self.sketch, engine=self.engine)
        scores = self._force_uncertainty(rows, n_atoms)
        if include_energy:
            scores["energy_uncertainty_score"] = self.posterior.variance(rows["energy"]).sqrt()
        return scores

    @torch.inference_mode(False)
    def score(self, structure, *, forces=True, force_probes=0, hessian_probes=0, hvp_probes=0, outputs=(),
              gap_pairs=None):
        """Predictions and raw posterior standard deviations for one :class:`Structure`.

        - ``forces``: rows for every force component, which give per-atom scores.
        - ``force_probes``, ``hessian_probes``: k Gaussian projections that estimate the RMS force and
          Hessian-element uncertainty, with about k rows each.
        - ``hvp_probes``: rows for H v, giving per-atom Hessian scores. That's 3N rows per probe.
        - ``outputs``: names of smooth model outputs (see ``NTKAdapter.outputs``) to score elementwise.
        - ``gap_pairs``: ordered state pairs for ``gap`` and ``gap_uncertainty_score``. The
          default reports adjacent pairs. Explicit pairs are returned in ``gap_pairs``.

        Outputs have a leading state axis.
        """
        n = structure.n_atoms
        if gap_pairs is not None:
            gap_pairs = _gap_pairs(gap_pairs, len(self.state_names))
        k_f, k_h, k_v = (_count(name, value) for name, value in
                         (("force_probes", force_probes), ("hessian_probes", hessian_probes),
                          ("hvp_probes", hvp_probes)))

        def probes(kind, count):
            return self._probes(kind, count, n)

        variance = self.posterior.variance
        if not self.state_names:
            # A model with outputs only, such as a coupling model: no energies, forces, or Hessians.
            if k_f or k_h or k_v:
                raise ValueError("Force and Hessian scores need a model with energies.")
            values, rows = compute_rows(self.adapter, structure, RowRequest(energy=False, outputs=tuple(outputs)),
                                        self.sketch, engine=self.engine)
            result = {}
            for name in outputs:
                result[f"output_{name}"] = values[f"output:{name}"]
                result[f"output_{name}_uncertainty_score"] = variance(rows[f"output:{name}"]).sqrt()
            return result
        bilinear = (probes(_BILINEAR_U, k_h), probes(_BILINEAR_V, k_h)) if k_h else None
        request = RowRequest(energy=True, forces=bool(forces), force_probes=probes(_FORCE_PROBES, k_f),
                             hvp_probes=probes(_HVP_PROBES, k_v), bilinear_probes=bilinear, force_values=True,
                             outputs=tuple(outputs))
        values, rows = compute_rows(self.adapter, structure, request, self.sketch, engine=self.engine)
        result = {"energy": values["energy"], "forces": values["forces"],
                  "energy_uncertainty_score": variance(rows["energy"]).sqrt()}
        if len(self.state_names) > 1:
            if gap_pairs is None:
                result["gap"] = values["energy"].diff()
                gap_rows = rows["energy"][1:] - rows["energy"][:-1]
            else:
                pairs = torch.as_tensor(gap_pairs, dtype=torch.long, device=self.device)
                lower, upper = pairs.T
                result.update(gap_pairs=pairs, gap=values["energy"][upper] - values["energy"][lower])
                gap_rows = rows["energy"][upper] - rows["energy"][lower]
            result["gap_uncertainty_score"] = variance(gap_rows).sqrt()
        for name in outputs:
            result[f"output_{name}"] = values[f"output:{name}"]
            result[f"output_{name}_uncertainty_score"] = variance(rows[f"output:{name}"]).sqrt()
        if forces or k_f:
            result.update(self._force_uncertainty(rows, n))
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


def reported_uncertainty(result, state_names, n_atoms, periodic=False, *, crossing_gap_eV=0.1,
                         max_output_elements=16):
    """ALF score names: scalars for acquisition, and atom arrays for selected frames.

    A model with several states also reports ``ntk_near_crossing_<state>``. That's 1.0 when a predicted gap
    next to the state is below ``crossing_gap_eV``, where adiabatic Hessian scores grow like 1/gap. The
    gap scores (``ntk_dE01``, …) and smooth-output scores stay meaningful there. A smooth output is
    reported as ``ntk_<name>_max``, plus each element ``ntk_<name>_<i>`` when it has at most
    ``max_output_elements`` elements.
    """
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
        pairs = (result["gap_pairs"].tolist() if "gap_pairs" in result
                 else zip(range(len(state_names) - 1), range(1, len(state_names))))
        for (lower, upper), value in zip(pairs, result["gap_uncertainty_score"]):
            scalar[f"ntk_{gap_key_for_pair(lower, upper)}"] = float(value)
    if multi and "gap" in result and crossing_gap_eV is not None:
        adjacent = result["energy"].diff() if "gap_pairs" in result else result["gap"]
        gaps = adjacent.abs().detach().cpu().numpy()
        for index, state in enumerate(state_names):
            neighbours = [gaps[i] for i in (index - 1, index) if 0 <= i < len(gaps)]
            scalar[f"ntk_near_crossing_{state}"] = float(min(neighbours) < crossing_gap_eV)
    for key, value in result.items():
        if key.startswith("output_") and key.endswith("_uncertainty_score"):
            name = key[len("output_"):-len("_uncertainty_score")]
            flat = value.detach().reshape(-1).cpu().numpy()
            scalar[f"ntk_{name}_max"] = float(flat.max())
            if len(flat) <= max_output_elements:
                scalar.update({f"ntk_{name}_{i}": float(v) for i, v in enumerate(flat)})
    if not all(math.isfinite(value) for value in scalar.values()):
        raise FloatingPointError("Nonfinite NTK uncertainty score.")
    return scalar, atomic


class NTKASECalculator(Calculator):
    """Report one fitted estimator through ALF's ASE uncertainty contract.

    Energy and forces are those of ``selected_state``. ``scores`` are passed to :meth:`NTKEstimator.score`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, estimator, *, selected_state=0, crossing_gap_eV=0.1, **scores):
        super().__init__()
        self.estimator, self.selected_state, self.scores = estimator, selected_state, scores
        self.crossing_gap_eV = crossing_gap_eV

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        result = self.estimator.predict(self.atoms, **self.scores)
        scalar, atomic = reported_uncertainty(result, self.estimator.state_names, len(self.atoms),
                                              bool(np.any(self.atoms.pbc)), crossing_gap_eV=self.crossing_gap_eV)
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

    def __init__(self, estimator, *, force_diagnostics=None, selected_state=0, crossing_gap_eV=0.1, **scores):
        self.estimator, self.force_diagnostics = estimator, force_diagnostics
        self.n_states = len(estimator.state_names)
        if not 0 <= selected_state < self.n_states:
            raise ValueError("Selected state is absent from the NTK model.")
        self.selected_state, self.scores = selected_state, scores
        self.crossing_gap_eV = crossing_gap_eV
        if force_diagnostics is not None:
            if (not estimator.energy_only or force_diagnostics.force_scale is None
                    or estimator.device != force_diagnostics.device or estimator.dtype != force_diagnostics.dtype
                    or estimator.metadata["model"] != force_diagnostics.metadata["model"]):
                raise ValueError("Force diagnostics require a force-fitted estimator on the same model, "
                                 "device, and dtype as an energy-only estimator.")

    @torch.inference_mode(False)
    def evaluate(self, numbers, positions, cells, pbc, counts):
        """Return ``[1, systems]`` energies and ``[1, atoms, 3]`` forces of the selected state, unscored."""
        energies, forces = self.evaluate_states(numbers, positions, cells, pbc, counts)
        return energies[:, self.selected_state], forces[:, self.selected_state]

    def evaluate_states(self, numbers, positions, cells, pbc, counts):
        """Return unscored physical predictions with axes ``[member, state, system or atom, ...]``."""
        if not counts or min(counts) <= 0 or sum(counts) != len(numbers) or positions.shape != (len(numbers), 3):
            raise ValueError("Packed atom counts must match the nonempty input structures.")
        energies, forces = self.estimator.adapter.predict_batch(numbers, positions, cells, pbc, counts)
        return energies[None], forces[None]

    def predict_states(self, atoms_list):
        """Return unscored molecular predictions for ASE's state-aware sampler."""
        if not atoms_list or any(atoms.pbc.any() for atoms in atoms_list):
            raise ValueError("ASE state prediction requires nonperiodic molecules.")
        counts = [len(atoms) for atoms in atoms_list]
        energies, forces = self.evaluate_states(torch.as_tensor(np.concatenate([a.numbers for a in atoms_list])),
                                                torch.as_tensor(np.concatenate([a.positions for a in atoms_list])),
                                                None, (False, False, False), counts)
        return dict(state_energies=energies.cpu().numpy(),
                    state_forces=[part.cpu().numpy() for part in forces.split(counts, dim=2)])

    @torch.inference_mode(False)
    def observe_uncertainty(self, numbers, positions, cell, pbc):
        """Return scores, and any energy row or atom diagnostics, for one structure."""
        estimator = self.estimator
        periodic = bool(torch.as_tensor(pbc).any())
        if estimator.energy_only and self.n_states == 1:
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
        scalar, atomic = reported_uncertainty(result, estimator.state_names, len(numbers), periodic,
                                              crossing_gap_eV=self.crossing_gap_eV)
        return {"uncertainty": scalar, "atomic_uncertainty": atomic}

    def observe_uncertainty_batch(self, numbers, positions, cells, pbc, counts, energies, forces,
                                  *, force_atom_indices=None):
        """Score every active structure at an acquisition check, in batch order.

        This is one call per check, so the packed engine can retry structures one at a time if a batch
        fails. Each structure's rows are still computed separately.
        """
        selections = [None] * len(counts) if force_atom_indices is None else force_atom_indices
        if len(selections) != len(counts):
            raise ValueError("Supply one force atom selection per packed structure.")
        observations, start = [], 0
        for index, count in enumerate(counts):
            stop = start + count
            cell = None if cells is None else cells[index]
            options = {} if selections[index] is None else {"force_atom_indices": selections[index]}
            observations.append(self.observe_uncertainty(numbers[start:stop], positions[start:stop], cell, pbc,
                                                         **options))
            start = stop
        return observations


class GroundStateNTKPackedEvaluator(NTKPackedEvaluator):
    """Ordinary one-member dynamics with NTK acquisition checks."""

    def __init__(self, estimator, inference=None, *, force_score="projected", force_probes=8, energy_score=False):
        super().__init__(estimator)
        if force_score not in {"projected", "exact", "selected"}:
            raise ValueError("force_score must be 'projected', 'exact', or 'selected'.")
        if type(force_probes) is not int or force_probes < 1:
            raise ValueError("force_probes must be a positive integer.")
        if type(energy_score) is not bool:
            raise ValueError("energy_score must be true or false.")
        self.inference = inference
        self.force_probes = force_probes if force_score == "projected" else 0
        self.requires_force_atom_indices = force_score == "selected"
        self.energy_score = energy_score

    @torch.inference_mode(False)
    def evaluate(self, numbers, positions, cells, pbc, counts):
        if self.inference is None:
            return super().evaluate(numbers, positions, cells, pbc, counts)
        return self.inference.evaluate(numbers, positions, cells, pbc, counts)

    @torch.inference_mode(False)
    def observe_uncertainty(self, numbers, positions, cell, pbc, *, force_atom_indices=None):
        if self.requires_force_atom_indices and force_atom_indices is None:
            raise ValueError("force_score='selected' requires acquisition.force_atoms.")
        estimator = self.estimator
        structure = estimator.adapter.structure(numbers, positions, cell, pbc)
        scores = estimator.score_forces(structure, force_probes=self.force_probes,
                                         atom_indices=force_atom_indices, include_energy=self.energy_score)
        scalar, report = {}, {}
        if self.energy_score:
            scalar["ntk_energy"] = float(scores["energy_uncertainty_score"][0])
            if bool(torch.as_tensor(pbc).any()):
                scalar["ntk_energy_per_atom"] = scalar["ntk_energy"] / len(numbers)
        if self.requires_force_atom_indices:
            report["selected_force_uncertainty"] = {
                "ntk_force_selected_max": scores["force_uncertainty_score"][0].cpu().numpy()}
        else:
            scalar["ntk_force_rms"] = float(scores["force_rms_uncertainty"][0])
            if not self.force_probes:
                scalar["ntk_force_max"] = float(scores["max_force_uncertainty_score"][0])
                report["atomic_uncertainty"] = {"ntk_force": scores["force_uncertainty_score"][0].cpu().numpy()}
        if not all(math.isfinite(value) for value in scalar.values()):
            raise FloatingPointError("Nonfinite NTK uncertainty score.")
        return {"uncertainty": scalar, "atomic_uncertainty": {}, **report}
