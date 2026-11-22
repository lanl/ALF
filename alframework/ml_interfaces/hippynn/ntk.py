"""Energy and force uncertainty for a fitted molecular HIPPYNN graph."""

from copy import deepcopy
import math

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

    The estimator holds a snapshot of the graph on the requested device.
    Precision defaults to float64 on CPU and float32 on GPU. Fit it again
    when the model changes. Select HIPPYNN's native PyTorch kernels in the
    worker before fitting and prediction.
    """

    @torch.inference_mode(False)
    @torch.enable_grad()
    def __init__(self, energy_graph, training_geometries, *, energy_scale, force_scale,
                 sketch_dim=64, seed=0, ridge_factor=1e-3, device="cpu", dtype=None):
        if not isinstance(sketch_dim, int) or isinstance(sketch_dim, bool) or sketch_dim < 1:
            raise ValueError("sketch_dim must be a positive integer.")
        for name, value in (("energy_scale", energy_scale), ("force_scale", force_scale),
                            ("ridge_factor", ridge_factor)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        device = torch.device(device)
        if device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("NTK device must be CPU, CUDA (including ROCm), or MPS.")
        if dtype is None:
            dtype = torch.float64 if device.type == "cpu" else torch.float32
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("NTK precision must be torch.float32 or torch.float64.")
        if device.type == "mps" and dtype != torch.float32:
            raise ValueError("MPS NTK requires torch.float32.")
        if device.type == "mps":
            if device.index not in (None, 0) or not torch.backends.mps.is_available():
                raise RuntimeError(f"The requested NTK MPS device {device} is unavailable.")
            device = torch.device("mps")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("The requested NTK CUDA/ROCm device is unavailable.")
            ordinal = torch.cuda.current_device() if device.index is None else device.index
            if ordinal >= torch.cuda.device_count():
                raise ValueError(f"NTK device {device} is outside the visible CUDA/ROCm devices.")
            device = torch.device("cuda", ordinal)

        self.device, self.dtype = device, dtype
        self._graph = deepcopy(energy_graph).to(device=device, dtype=dtype).eval()
        self._parameters = tuple(p for p in self._graph.parameters() if p.requires_grad)
        if not self._parameters:
            raise ValueError("The energy graph has no trainable parameters.")
        sizes = [p.numel() for p in self._parameters]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(sum(sizes), sketch_dim, generator=generator, dtype=torch.float64, device="cpu")
        projection.div_(math.sqrt(sketch_dim))
        self._projection = projection.to(device=device, dtype=dtype).split(sizes)
        self._sketch_dim = sketch_dim

        information = torch.zeros((sketch_dim, sketch_dim), dtype=dtype, device=device)
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
        self._factor, status = torch.linalg.cholesky_ex(information)
        if int(status) != 0 or not torch.isfinite(self._factor).all():
            raise ValueError("Could not factor the NTK information matrix. Check scales, ridge, and precision.")

    def _parameter_row(self, value):
        row = torch.zeros(self._sketch_dim, dtype=self.dtype, device=self.device)
        if value.requires_grad:
            gradients = torch.autograd.grad(value, self._parameters, retain_graph=True, allow_unused=True)
            for gradient, projection in zip(gradients, self._projection):
                if gradient is not None:
                    row.addmv_(projection.T, gradient.reshape(-1))
        return row

    def _energy_forces(self, species, positions, *, create_graph):
        if custom_kernels.CUSTOM_KERNELS_ACTIVE not in (False, "pytorch"):
            raise ValueError("Select native PyTorch kernels with hippynn.custom_kernels.set_custom_kernels(False).")
        positions = positions.detach().clone().requires_grad_(True)
        output = self._graph(species, positions)[0]
        if output.shape != (len(species), 1):
            raise ValueError("The first graph output must be one energy per structure with shape [batch, 1].")
        gradient = None
        if output.requires_grad:
            gradient, = torch.autograd.grad(output.sum(), positions, create_graph=create_graph, allow_unused=True)
        forces = torch.zeros_like(positions) if gradient is None else -gradient
        return output[:, 0], forces

    def _tangents(self, atoms):
        return self._tensor_tangents(atoms.numbers, atoms.positions, atoms.pbc)

    def _tensor_tangents(self, numbers, positions, pbc):
        if pbc.any():
            raise ValueError("Energy and force NTK requires nonperiodic structures.")
        # Convert ASE NumPy inputs on the host first. MPS cannot accept an
        # intermediate float64 device tensor. Cloning also handles inference tensors.
        numbers = numbers if torch.is_tensor(numbers) else torch.from_numpy(numbers)
        positions = positions if torch.is_tensor(positions) else torch.from_numpy(positions)
        species = numbers.to(device=self.device, dtype=torch.long).clone()
        positions = positions.to(device=self.device, dtype=self.dtype)
        if not len(species) or (species <= 0).any() or not torch.isfinite(positions).all():
            raise ValueError("Supply a nonempty structure with positive atomic numbers and finite positions.")
        energies, batch_forces = self._energy_forces(species[None], positions[None], create_graph=True)
        energy, forces = energies[0], batch_forces[0]
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
        """Return predictions and raw uncertainty scores on the estimator device.

        Force component scores are [atoms, 3]. The per-atom score is the norm
        of the three component scores. Calibrate the scores against errors
        before using them to decide whether to request labels.
        """
        return self._score(*self._tangents(atoms))

    def _score(self, energy, forces, energy_row, force_rows):
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
        scalar, atomic = _reported_uncertainty(result)
        self.results = {
            "energy": float(result["energy"]),
            "forces": result["forces"].cpu().numpy(),
            "alf_uncertainty": scalar,
            "alf_atomic_uncertainty": atomic,
        }


def _reported_uncertainty(result):
    return {
        "ntk_energy": float(result["energy_uncertainty_score"]),
        "ntk_force_max": float(result["max_force_uncertainty_score"]),
    }, {"ntk_force": result["force_uncertainty_score"].cpu().numpy()}


class NTKPackedEvaluator:
    """Use one fitted snapshot for ordinary MD forces and observation-only NTK scores."""

    n_models = 1

    def __init__(self, estimator):
        self.estimator = estimator

    @torch.inference_mode(False)
    @torch.enable_grad()
    def evaluate(self, numbers, positions, cells, pbc, counts):
        """Return [1, systems] energies and [1, atoms, 3] forces without scoring."""
        if pbc.any():
            raise ValueError("Energy and force NTK requires nonperiodic structures.")
        if not counts or min(counts) <= 0 or sum(counts) != len(numbers) or positions.shape != (len(numbers), 3):
            raise ValueError("Packed atom counts must match the nonempty input structures.")
        device, dtype = self.estimator.device, self.estimator.dtype
        width = max(counts)
        mask = torch.arange(width, device=device)[None] < torch.as_tensor(counts, device=device)[:, None]
        species = torch.zeros((len(counts), width), dtype=torch.long, device=device)
        coordinates = torch.zeros((len(counts), width, 3), dtype=dtype, device=device)
        species[mask] = numbers.to(device=device, dtype=torch.long)
        coordinates[mask] = positions.detach().to(device=device, dtype=dtype)
        energy, forces = self.estimator._energy_forces(species, coordinates, create_graph=False)
        return energy.detach()[None], forces.detach()[mask][None]

    @torch.inference_mode(False)
    @torch.enable_grad()
    def observe_uncertainty(self, numbers, positions, cell, pbc):
        """Return the named scores and atom arrays for one observed structure."""
        result = self.estimator._score(*self.estimator._tensor_tangents(numbers, positions, pbc))
        scalar, atomic = _reported_uncertainty(result)
        return {"uncertainty": scalar, "atomic_uncertainty": atomic}
