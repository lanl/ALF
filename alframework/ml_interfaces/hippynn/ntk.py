"""Energy and force uncertainty for a fitted molecular HIPPYNN graph."""

from copy import deepcopy
import math

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from hippynn import custom_kernels


class EnergyForceNTK:
    """Fit a sketched neural tangent kernel on the model's training geometries.

    The graph takes batched atomic numbers and Cartesian positions in Angstrom
    and returns graph outputs with a single total energy in eV at index zero
    (shape [batch, 1]). Geometries are nonperiodic ASE Atoms in input order.
    Each structure contributes one energy row and all Cartesian force rows.
    The observation scales have units of eV and eV/Angstrom respectively.

    The estimator holds a CPU float64 snapshot of the graph. Fit it again when
    the model changes. Select HIPPYNN's native PyTorch kernels in the worker
    before fitting and prediction.
    """

    @torch.inference_mode(False)
    @torch.enable_grad()
    def __init__(self, energy_graph, training_geometries, *, energy_scale, force_scale,
                 sketch_dim=64, seed=0, ridge_factor=1e-3):
        if not isinstance(sketch_dim, int) or isinstance(sketch_dim, bool) or sketch_dim < 1:
            raise ValueError("sketch_dim must be a positive integer.")
        for name, value in (("energy_scale", energy_scale), ("force_scale", force_scale),
                            ("ridge_factor", ridge_factor)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        self._graph = deepcopy(energy_graph).to(device="cpu", dtype=torch.float64).eval()
        self._parameters = tuple(p for p in self._graph.parameters() if p.requires_grad)
        if not self._parameters:
            raise ValueError("The energy graph has no trainable parameters.")
        sizes = [p.numel() for p in self._parameters]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(sum(sizes), sketch_dim, generator=generator, dtype=torch.float64, device="cpu")
        projection.div_(math.sqrt(sketch_dim))
        self._projection = projection.split(sizes)
        self._sketch_dim = sketch_dim

        information = torch.zeros((sketch_dim, sketch_dim), dtype=torch.float64, device="cpu")
        self.training_structure_count = 0
        for atoms in training_geometries:
            _, _, energy_row, force_rows = self._tangents(atoms)
            rows = torch.cat((energy_row[None] / energy_scale,
                              force_rows.flatten(0, 1) / force_scale))
            information.addmm_(rows.T, rows)
            self.training_structure_count += 1
        if not self.training_structure_count:
            raise ValueError("Supply at least one training geometry.")
        self.ridge = float(ridge_factor * information.diagonal().mean())
        if not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("The NTK ridge is not finite and positive. Check model sensitivities and scales.")
        information.diagonal().add_(self.ridge)
        self._factor = torch.linalg.cholesky(information)

    def _parameter_row(self, value):
        row = torch.zeros(self._sketch_dim, dtype=torch.float64, device="cpu")
        if value.requires_grad:
            gradients = torch.autograd.grad(value, self._parameters, retain_graph=True, allow_unused=True)
            for gradient, projection in zip(gradients, self._projection):
                if gradient is not None:
                    row.addmv_(projection.T, gradient.reshape(-1))
        return row

    def _tangents(self, atoms):
        if atoms.pbc.any():
            raise ValueError("Energy and force NTK requires nonperiodic structures.")
        if not len(atoms) or np.any(atoms.numbers <= 0) or not np.isfinite(atoms.positions).all():
            raise ValueError("Supply a nonempty structure with positive atomic numbers and finite positions.")
        if custom_kernels.CUSTOM_KERNELS_ACTIVE not in (False, "pytorch"):
            raise ValueError("Select native PyTorch kernels with hippynn.custom_kernels.set_custom_kernels(False).")

        species = torch.tensor(atoms.numbers, dtype=torch.long, device="cpu")
        positions = torch.tensor(atoms.positions, dtype=torch.float64, device="cpu", requires_grad=True)
        output = self._graph(species[None], positions[None])[0]
        if output.shape != (1, 1):
            raise ValueError("The first graph output must be one energy per structure with shape [batch, 1].")
        energy = output[0, 0]
        gradient = None
        if energy.requires_grad:
            gradient, = torch.autograd.grad(energy, positions, create_graph=True, allow_unused=True)
        forces = torch.zeros_like(positions) if gradient is None else -gradient
        energy_row = self._parameter_row(energy)
        force_rows = torch.stack([self._parameter_row(f) for f in forces.flatten()])
        if not all(torch.isfinite(value).all() for value in (energy, forces, energy_row, force_rows)):
            raise FloatingPointError("The energy graph produced nonfinite predictions or NTK features.")
        return energy.detach(), forces.detach(), energy_row, force_rows.reshape(*forces.shape, self._sketch_dim)

    def _variance(self, rows):
        flat = rows.reshape(-1, self._sketch_dim)
        whitened = torch.linalg.solve_triangular(self._factor, flat.T, upper=False)
        return (self.ridge * whitened.square().sum(0)).reshape(rows.shape[:-1])

    @torch.inference_mode(False)
    @torch.enable_grad()
    def predict_energy_forces(self, atoms):
        """Return predictions and raw uncertainty scores as detached CPU tensors.

        Force component scores are [atoms, 3]. The per-atom score is the norm
        of the three component scores. Calibrate the scores against errors
        before using them to decide whether to request labels.
        """
        energy, forces, energy_row, force_rows = self._tangents(atoms)
        energy_score = self._variance(energy_row).sqrt()
        force_variance = self._variance(force_rows)
        atom_scores = force_variance.sum(1).sqrt()
        if not torch.isfinite(energy_score) or not torch.isfinite(atom_scores).all():
            raise FloatingPointError("Nonfinite NTK uncertainty score.")
        return {
            "energy": energy,
            "forces": forces,
            "energy_uncertainty_score": energy_score,
            "force_component_uncertainty_score": force_variance.sqrt(),
            "force_uncertainty_score": atom_scores,
            "max_force_uncertainty_score": atom_scores.max(),
        }


class NTKASECalculator(Calculator):
    """Expose one fitted EnergyForceNTK through ALF's ASE uncertainty contract."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        result = self.estimator.predict_energy_forces(self.atoms)
        self.results = {
            "energy": float(result["energy"]),
            "forces": result["forces"].numpy(),
            "alf_uncertainty": {
                "ntk_energy": float(result["energy_uncertainty_score"]),
                "ntk_force_max": float(result["max_force_uncertainty_score"]),
            },
            "alf_atomic_uncertainty": {"ntk_force": result["force_uncertainty_score"].numpy()},
        }
