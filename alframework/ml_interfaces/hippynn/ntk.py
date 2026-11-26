"""HIPPYNN graphs for ALF's NTK uncertainty, molecular or periodic, with one or more energy states.

The graph takes ``(species, positions)``, or ``(species, positions, cell)`` with a ``CellNode``, in batch
form. Each output is a ``[batch, 1]`` total energy in eV, one per state. NTK rows use HIPPYNN's
plain-PyTorch message-passing kernels (``_pytorch_raw``) during scoring. These support ``torch.func``
and derivatives of any order. The previously active kernels, Triton for example, are restored afterwards,
so MD forces keep using them.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import warnings

import numpy as np
import torch
from torch.func import functional_call

from hippynn import custom_kernels
from hippynn.graphs import inputs as hippynn_inputs

from alframework.ml_interfaces.ntk.adapter import NTKAdapter, resolve_device_dtype
from alframework.ml_interfaces.ntk.estimator import (NTKASECalculator, NTKEstimator, NTKPackedEvaluator,
                                                     read_basis_metadata)

# NTKASECalculator and NTKPackedEvaluator are imported for convenience; they are documented in the core.
__all__ = ["HippynnNTKAdapter", "EnergyForceNTK", "EnergyNTK"]

SCORING_KERNELS = "_pytorch_raw"


@contextmanager
def scoring_kernels(name=SCORING_KERNELS):
    """Select HIPPYNN's message-passing kernels for a block, then restore the previous ones."""
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    if previous == name:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using non-standard custom kernel implementation")
        custom_kernels.set_custom_kernels(name)
    try:
        yield
    finally:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using non-standard custom kernel implementation")
            custom_kernels.set_custom_kernels(previous)


class _Functional(torch.nn.Module):
    """``GraphModule.forward`` without its ``requires_grad_`` on inputs, which ``torch.func`` forbids."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, *values):
        graph = self.graph
        computed = dict(zip(graph.input_nodes, values))
        for node in graph.forward_output_list:
            computed[node] = graph.get_module(node)(*(computed[parent] for parent in node.parents))
        return tuple(computed[node] for node in graph.nodes_to_compute)


def _graph_digest(graph):
    """Hash the graph snapshot and trainable parameter order used by the sketch."""
    digest = hashlib.sha256()
    for name, module in graph.named_modules():
        digest.update(repr((name, type(module).__module__, type(module).__qualname__)).encode())
    for name, parameter in graph.named_parameters():
        digest.update(repr((name, parameter.requires_grad)).encode())
    for name, value in graph.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(repr((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class HippynnNTKAdapter(NTKAdapter):
    """A frozen copy of a HIPPYNN energy graph on the scoring device and precision."""

    supports_forward = True

    @torch.inference_mode(False)
    def __init__(self, energy_graph, *, device="cpu", dtype=None, state_names=None):
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        graph = deepcopy(energy_graph).to(device=self.device, dtype=self.dtype).eval()
        roles = []
        for node in graph.input_nodes:
            if isinstance(node, hippynn_inputs.SpeciesNode):
                roles.append("species")
            elif isinstance(node, hippynn_inputs.PositionsNode):
                roles.append("positions")
            elif isinstance(node, hippynn_inputs.CellNode):
                roles.append("cell")
            else:
                raise ValueError(f"Unsupported HIPPYNN graph input {node.name!r}; use species, positions, and cell.")
        if sorted(roles) not in (["positions", "species"], ["cell", "positions", "species"]):
            raise ValueError("The graph must take species and positions, and optionally a cell.")
        self.roles, self.periodic = tuple(roles), "cell" in roles
        n_states = len(graph.nodes_to_compute)
        if state_names is None:
            state_names = ("energy",) if n_states == 1 else tuple(f"E{i}" for i in range(n_states))
        if len(state_names) != n_states:
            raise ValueError(f"The graph has {n_states} energy outputs but {len(state_names)} state names.")
        self.state_names = tuple(state_names)
        self.graph = graph
        self._functional = _Functional(graph)
        self._parameters = {name: p for name, p in graph.named_parameters() if p.requires_grad}
        if not self._parameters:
            raise ValueError("The energy graph has no trainable parameters.")
        self._digest = _graph_digest(graph)

    def live_parameters(self):
        return self._parameters

    def digest(self):
        return self._digest

    def prepare(self, structure):
        periodic = bool(structure.pbc.any())
        if periodic and not self.periodic:
            raise ValueError("This HIPPYNN graph takes no cell, so it scores nonperiodic structures only.")
        if self.periodic and not bool(structure.pbc.all()):
            raise ValueError("A periodic HIPPYNN graph needs fully periodic cells.")

    def _arguments(self, species, positions, cells):
        values = {"species": species, "positions": positions, "cell": cells}
        return tuple(values[role] for role in self.roles)

    def _stack_outputs(self, outputs, batch):
        for output in outputs:
            if output.shape != (batch, 1):
                raise ValueError("Each graph output must be one energy per structure with shape [batch, 1].")
        return torch.stack([output[:, 0] for output in outputs])  # [S, batch]

    def energies(self, structure, positions, params=None):
        arguments = self._arguments(structure.numbers[None], positions[None], structure.cell[None])
        if params is None:
            outputs = self._functional(*arguments)
        else:
            outputs = functional_call(self._functional, {f"graph.{k}": v for k, v in params.items()}, arguments)
        return self._stack_outputs(outputs, 1)[:, 0]

    def forward_context(self):
        return scoring_kernels()

    def reverse_context(self):
        return scoring_kernels()

    @torch.inference_mode(False)
    def predict_batch(self, numbers, positions, cells, pbc, counts):
        """Padded batch energies and forces with the active kernels (Triton, for example)."""
        device, dtype = self.device, self.dtype
        pbc = torch.as_tensor(pbc)
        if bool(pbc.any()) != self.periodic or (self.periodic and not bool(pbc.all())):
            raise ValueError("This HIPPYNN graph needs " + ("fully periodic" if self.periodic else "nonperiodic")
                             + " structures.")
        width = max(counts)
        mask = torch.arange(width, device=device)[None] < torch.as_tensor(counts, device=device)[:, None]
        species = torch.zeros((len(counts), width), dtype=torch.long, device=device)
        coordinates = torch.zeros((len(counts), width, 3), dtype=dtype, device=device)
        species[mask] = numbers.to(device=device, dtype=torch.long)
        coordinates[mask] = positions.detach().to(device=device, dtype=dtype).clone()
        cell = None if not self.periodic else torch.as_tensor(cells).to(device=device, dtype=dtype).clone()
        with torch.inference_mode(False), torch.enable_grad():
            coordinates.requires_grad_(True)
            energies = self._stack_outputs(self._functional(*self._arguments(species, coordinates, cell)), len(counts))
            forces = []
            for state in energies:
                gradient, = torch.autograd.grad(state.sum(), coordinates, retain_graph=True, allow_unused=True)
                forces.append(torch.zeros_like(coordinates[mask]) if gradient is None else -gradient[mask])
        return energies.detach(), torch.stack(forces).detach()


class EnergyForceNTK(NTKEstimator):
    """Fit a sketched NTK posterior on a HIPPYNN graph's energy and force training geometries.

    Observation scales are in eV, eV/Å, and eV/Å² (see :class:`NTKEstimator`). The graph snapshot is
    copied to ``device``, in float64 on CPU and float32 on GPU by default. Refit the estimator when the
    model changes.
    """

    def __init__(self, energy_graph, training_geometries, *, energy_scale, force_scale, hessian_scale=None,
                 hessian_fit_probes=4, sketch_dim=64, seed=0, ridge=None, ridge_factor=1e-3, device="cpu",
                 dtype=None, engine="auto", probe_seed=0, state_names=None):
        super().__init__(HippynnNTKAdapter(energy_graph, device=device, dtype=dtype, state_names=state_names),
                         training_geometries, energy_scale=energy_scale, force_scale=force_scale,
                         hessian_scale=hessian_scale, hessian_fit_probes=hessian_fit_probes,
                         sketch_dim=sketch_dim, seed=seed, ridge=ridge, ridge_factor=ridge_factor,
                         engine=engine, probe_seed=probe_seed)


class EnergyNTK(NTKEstimator):
    """The AdaptNTK basis: energy-gradient rows of a HIPPYNN graph's original training geometries.

    Fit once per checkpoint, save it with :meth:`save_basis`, and restore it in workers with
    :meth:`from_basis` on the same graph.
    """

    def __init__(self, energy_graph, training_geometries, *, sketch_dim=64, seed=0, ridge=None,
                 ridge_factor=1e-3, device="cpu", dtype=None, engine="auto", state_names=None):
        super().__init__(HippynnNTKAdapter(energy_graph, device=device, dtype=dtype, state_names=state_names),
                         training_geometries, sketch_dim=sketch_dim, seed=seed, ridge=ridge,
                         ridge_factor=ridge_factor, engine=engine)

    @classmethod
    def from_basis(cls, energy_graph, path, *, device="cpu", engine="auto", state_names=None):
        """Restore a saved basis for a worker without reading training data."""
        dtype = getattr(torch, read_basis_metadata(path)["dtype"])
        adapter = HippynnNTKAdapter(energy_graph, device=device, dtype=dtype, state_names=state_names)
        return NTKEstimator.from_basis(adapter, path, engine=engine)
