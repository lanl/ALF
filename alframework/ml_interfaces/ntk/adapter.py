"""What a potential provides to the NTK core."""

from contextlib import nullcontext
from dataclasses import dataclass, field
import math

import numpy as np
import torch


def resolve_device_dtype(device, dtype):
    """Validate an NTK device and choose float64 on CPU and float32 on GPUs by default."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("NTK device must be CPU, CUDA (including ROCm), or MPS.")
    if dtype is None:
        dtype = torch.float64 if device.type == "cpu" else torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("NTK precision must be torch.float32 or torch.float64.")
    if device.type == "mps":
        if dtype != torch.float32:
            raise ValueError("MPS NTK requires torch.float32.")
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
    return device, dtype


@dataclass
class Structure:
    """One structure on the adapter's device: atomic numbers, positions in Å, a row-vector cell, pbc.

    ``extra`` holds whatever the adapter precomputes from the fixed geometry, such as a neighbor list.
    """

    numbers: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor
    extra: dict = field(default_factory=dict)

    @property
    def n_atoms(self):
        return len(self.numbers)


class NTKAdapter:
    """Base class for a potential seen by the NTK core.

    A subclass sets ``state_names``, ``device`` and ``dtype``, and implements :meth:`energies`. Set
    ``supports_forward`` only when :meth:`energies` works inside ``torch.func`` transforms with ``params``
    given. The reverse engine needs ``energies`` with ``params=None`` to use the live parameters from
    :meth:`live_parameters`.
    """

    state_names = ("energy",)
    supports_forward = False
    device = torch.device("cpu")
    dtype = torch.float64

    # -- parameters ------------------------------------------------------------------------------------
    def live_parameters(self):
        """``{name: parameter}`` for the trainable parameters, in a fixed order."""
        raise NotImplementedError

    def parameter_shapes(self):
        return {name: p.shape for name, p in self.live_parameters().items()}

    def parameter_values(self):
        return {name: p.detach() for name, p in self.live_parameters().items()}

    def digest(self):
        """A hash of the model snapshot that a saved basis must match."""
        raise NotImplementedError

    # -- structures ------------------------------------------------------------------------------------
    @torch.inference_mode(False)
    def structure(self, numbers, positions, cell=None, pbc=None):
        """Build a :class:`Structure` from NumPy or tensor inputs, validating them."""
        numbers = torch.as_tensor(np.asarray(numbers) if not torch.is_tensor(numbers) else numbers)
        positions = positions if torch.is_tensor(positions) else torch.from_numpy(np.asarray(positions, float))
        # MPS cannot hold an intermediate float64 tensor, so convert on the host first.
        positions = positions.detach().to(dtype=self.dtype).to(self.device).clone()
        numbers = numbers.to(device=self.device, dtype=torch.long).clone()
        pbc = torch.zeros(3, dtype=torch.bool) if pbc is None else torch.as_tensor(
            np.asarray(pbc) if not torch.is_tensor(pbc) else pbc, dtype=torch.bool)
        pbc = pbc.reshape(-1).expand(3).to(self.device) if pbc.numel() == 1 else pbc.to(self.device)
        if cell is None:
            cell = torch.zeros((3, 3))
        cell = cell if torch.is_tensor(cell) else torch.from_numpy(np.asarray(cell, float))
        cell = cell.detach().to(dtype=self.dtype).to(self.device).reshape(3, 3).clone()
        if (not len(numbers) or (numbers <= 0).any() or positions.shape != (len(numbers), 3)
                or not torch.isfinite(positions).all() or not torch.isfinite(cell).all()):
            raise ValueError("Supply a nonempty structure with positive atomic numbers and finite positions.")
        structure = Structure(numbers, positions, cell, pbc)
        self.prepare(structure)
        return structure

    def prepare(self, structure):
        """Precompute geometry-fixed inputs, such as neighbor lists, into ``structure.extra``."""

    # -- evaluation ------------------------------------------------------------------------------------
    def energies(self, structure, positions, params=None):
        """Return the per-state energies ``[n_states]`` in eV at ``positions`` ``[N, 3]``.

        ``params`` is a ``{name: tensor}`` mapping for ``torch.func.functional_call``, or None for the
        live parameters.
        """
        raise NotImplementedError

    def forward_context(self):
        """A context in which :meth:`energies` is valid inside ``torch.func`` transforms."""
        return nullcontext()

    def reverse_context(self):
        """A context in which :meth:`energies` supports reverse mode to third order."""
        return nullcontext()

    @torch.inference_mode(False)
    def predict_batch(self, numbers, positions, cells, pbc, counts):
        """Energies ``[n_states, n_systems]`` and forces ``[n_states, n_atoms, 3]`` for a packed batch.

        The default evaluates one structure at a time. Adapters can override this with a batched model call.
        """
        energies, forces = [], []
        start = 0
        with torch.inference_mode(False), torch.enable_grad(), self.reverse_context():
            for index, count in enumerate(counts):
                stop = start + count
                structure = self.structure(numbers[start:stop], positions[start:stop],
                                           None if cells is None else cells[index], pbc)
                x = structure.positions.clone().requires_grad_(True)
                energy = self.energies(structure, x)
                gradients = []
                for value in energy:
                    g = torch.autograd.grad(value, x, retain_graph=True, allow_unused=True)[0] \
                        if value.requires_grad else None
                    gradients.append(torch.zeros_like(x) if g is None else g)
                energies.append(energy.detach())
                forces.append(-torch.stack(gradients).detach())
                start = stop
        return torch.stack(energies, dim=1), torch.cat(forces, dim=1)


def gaussian_probes(count, n_atoms, *, seed, device, dtype):
    """``count`` independent standard normal probes ``[count, n_atoms, 3]``, generated on the CPU.

    The probes are not normalized, so E[v vᵀ] = I, as in Hutchinson estimation and Projected Hessian Learning.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & ((1 << 63) - 1))
    probes = torch.randn(count, n_atoms, 3, generator=generator, dtype=torch.float64)
    return probes.to(dtype=dtype).to(device)


def check_finite(name, *values):
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError(f"The model produced nonfinite {name}.")


def as_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite NTK uncertainty score.")
    return value
