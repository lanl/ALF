"""HIPPYNN graphs for ALF's NTK uncertainty, molecular or periodic, with one or more energy states.

The graph takes ``(species, positions)``, or ``(species, positions, cell)`` with a ``CellNode``, in batch
form. Each output is a ``[batch, 1]`` total energy in eV, one per state. NTK rows use HIPPYNN's
plain-PyTorch message-passing kernels (``_pytorch_raw``) during scoring. These support ``torch.func``
and derivatives of any order. The previously active kernels, Triton for example, are restored afterwards,
so MD forces keep using them.
"""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.func import functional_call

from hippynn import custom_kernels
from hippynn.graphs import inputs as hippynn_inputs

from alframework.ml_interfaces.ntk.adapter import NTKAdapter, resolve_device_dtype, select_parameters
from alframework.ml_interfaces.ntk.estimator import (GroundStateNTKPackedEvaluator, NTKASECalculator,
                                                     NTKEstimator, NTKPackedEvaluator,
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
    def __init__(self, energy_graph, *, device="cpu", dtype=None, state_names=None, parameters="trainable",
                 energy_scales=None):
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
        # The graph's outputs are in label units; dividing by these gives eV (see properties_list).
        scales = [1.0] * n_states if energy_scales is None else [float(s) for s in energy_scales]
        if len(scales) != n_states or not all(math.isfinite(s) and s > 0 for s in scales):
            raise ValueError("energy_scales needs one finite positive factor per state.")
        self.energy_scales = torch.tensor(scales, dtype=self.dtype, device=self.device)
        self.graph = graph
        self._functional = _Functional(graph)
        try:
            self._parameters = select_parameters(graph, parameters)
        except ValueError:
            raise ValueError("The energy graph has no trainable parameters.") from None
        self._digest = _graph_digest(graph) + ("" if energy_scales is None else repr(scales))

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
        return torch.stack([output[:, 0] for output in outputs]) / self.energy_scales[:, None]  # [S, batch], eV

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


# -- ALF's excited-state HIPPYNN models --------------------------------------------------------------------

NTK_BASIS_FILE = "ntk_basis.npz"


def excited_state_ntk_adapter(member_directory, properties_list, *, device="cpu", dtype=None, parameters="trainable"):
    """An NTK adapter for one member of an ALF excited-state HIPPYNN ensemble.

    The graph keeps the member's state energy outputs (``E0``, ``E1``, … in ``properties_list``), and the
    adapter converts them to eV with the properties' conversion factors.
    """
    from hippynn.graphs import GraphModule

    from alframework.ml_interfaces.hippynn.interface import (_load_member, _target,
                                                             validate_excited_state_training_properties)

    table = validate_excited_state_training_properties(properties_list)
    model = _load_member(str(Path(member_directory).resolve()), "cpu")
    graph = GraphModule(model.input_nodes, [_target(model, row["energy_db_name"]) for row in table])
    scales = []
    for row in table:
        energy, force = (float(properties_list[row[key]][2]) if len(properties_list[row[key]]) > 2 else 1.0
                         for key in ("energy_key", "force_key"))
        if not math.isclose(energy, force):
            raise ValueError("NTK forces are energy derivatives, so each state's energy and force conversion "
                             "factors must be equal.")
        scales.append(energy)
    return HippynnNTKAdapter(graph, device=device, dtype=dtype, parameters=parameters,
                             state_names=tuple(row["energy_key"] for row in table), energy_scales=scales)


def training_shards(model_directory):
    """The frozen training inventory recorded with an ensemble (``label_sources.json``)."""
    record = Path(model_directory) / "label_sources.json"
    if record.is_file():
        shards = json.loads(record.read_text()).get("training_shards")
        if shards:
            return shards
    raise ValueError(f"{model_directory} records no training shards; pass training_shards explicitly.")


def _member_training_split(member_directory):
    path = member_directory / "training_input_manifest.json"
    if not path.is_file():
        raise ValueError(f"{path} is required to fit or verify NTK on the member's exact training split.")
    manifest = json.loads(path.read_text())
    files, frames, splits = (manifest.get(key) for key in ("files", "frames", "splits"))
    indices = splits.get("train") if isinstance(splits, dict) else None
    if not files or not frames or not indices:
        raise ValueError(f"{path} must record the training shards, frame locations, and training indices.")
    excluded = set(splits.get("valid", ())) | set(splits.get("test", ())) | set(manifest.get("filtered_indices", ()))
    if (any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
            or len(set(indices)) != len(indices) or excluded.intersection(indices)):
        raise ValueError(f"{path} has invalid training indices or includes held-out or filtered frames.")
    return files, frames, sorted(indices)


def _matching_basis(adapter, path, expected):
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return None
    with stream:
        metadata = read_basis_metadata(stream)
        if all(metadata.get(key) == value for key, value in expected.items()):
            # Check and restore the same file even if a worker replaces the path concurrently.
            stream.seek(0)
            return NTKEstimator.from_basis(adapter, stream)
    return None


def _split_geometries(adapter, properties_list, shards, files, frames, indices, *, read_workers):
    from ase import Atoms

    from alframework.ml_interfaces.hippynn.interface import load_excited_state_h5_arrays

    keys = {role: node.db_name for role, node in zip(adapter.roles, adapter.graph.input_nodes)}
    arrays, summary = load_excited_state_h5_arrays(
        None, properties_list, coordinates_key=keys["positions"], species_key=keys["species"],
        shards=shards, read_workers=read_workers)
    if [entry["sha256"] for entry in summary["files"]] != [entry["sha256"] for entry in files]:
        raise ValueError("NTK training shards do not match the member's training manifest.")
    locations = {entry["path"]: shard for entry, shard in zip(files, shards)}
    expected = [(locations[frame["file"]], frame["group"], frame["frame"]) for frame in frames]
    observed = [(frame["file"], frame["group"], frame["frame"]) for frame in summary["frames"]]
    if observed != expected:
        raise ValueError("NTK training frame order does not match the member's training manifest.")
    numbers, coordinates = arrays[keys["species"]], arrays[keys["positions"]]
    return (Atoms(numbers=numbers[i], positions=coordinates[i]) for i in indices)


def load_hippynn_excited_state_ntk_estimator(model_directory, properties_list, *, device="cpu", dtype=None,
                                             member=0, shards=None, energy_scale=0.05, force_scale=0.1,
                                             sketch_dim=64, seed=0, ridge_factor=1e-3, parameters="trainable",
                                             basis_file=NTK_BASIS_FILE, h5_read_workers=None):
    """Fit, or restore, the NTK posterior of one excited-state ensemble member.

    The member's training manifest identifies the geometries to use. Supply ``shards`` in manifest order
    to relocate byte-identical files. A cached basis is reused for the same model, training split, and
    fit settings. Set ``force_scale`` and ``energy_scale`` from the model's validation errors.
    """
    members = sorted(Path(model_directory).glob("model-*/"))
    if not 0 <= member < len(members):
        raise ValueError(f"{model_directory} has {len(members)} members, not a member {member}.")
    directory = members[member]
    files, frames, indices = _member_training_split(directory)
    checksums = [entry["sha256"] for entry in files]
    if shards is None:
        shards = [entry["path"] for entry in files]
    else:
        from alframework.tools.files import cached_sha256

        shards = [str(Path(shard).resolve()) for shard in shards]
        if [cached_sha256(shard) for shard in shards] != checksums:
            raise ValueError("Explicit NTK training shards must match the member's training manifest in order.")
    adapter = excited_state_ntk_adapter(directory, properties_list, device=device, dtype=dtype, parameters=parameters)
    split = {"files": files, "frames": [frames[i] for i in indices], "indices": indices}
    provenance = {"training_split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
                  "ridge_factor": float(ridge_factor)}
    energy_scale = None if energy_scale is None else float(energy_scale)
    force_scale = None if force_scale is None else float(force_scale)
    expected = {"model": adapter.digest(), "dtype": str(adapter.dtype).split(".")[-1],
                "sketch_dim": sketch_dim, "seed": int(seed), "fit_provenance": provenance,
                "training_structure_count": len(indices),
                "scales": {"energy": energy_scale, "force": force_scale, "hessian": None}}
    path = directory / basis_file
    estimator = _matching_basis(adapter, path, expected)
    if estimator is not None:
        return estimator
    # Workers may all load the same newly published checkpoint at once.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        estimator = _matching_basis(adapter, path, expected)
        if estimator is None:
            geometries = _split_geometries(adapter, properties_list, shards, files, frames, indices,
                                          read_workers=h5_read_workers)
            estimator = NTKEstimator(adapter, geometries, energy_scale=energy_scale, force_scale=force_scale,
                                     sketch_dim=sketch_dim, seed=seed, ridge_factor=ridge_factor,
                                     fit_provenance=provenance)
            estimator.save_basis(path)
    return estimator


def load_hippynn_excited_state_ntk(model_directory, device="cpu", *, properties_list, selected_state=0,
                                   scores=None, crossing_gap_eV=0.1, **fit):
    """``batched_calculator`` loader: one excited-state member with NTK scores for MD.

    ALF's excited-state sampler chooses the propagation state. ``selected_state`` sets the state
    for direct use of the evaluator. ``scores`` are the options of ``NTKEstimator.score``. Pass
    ``properties_list`` and any fit settings through ``batched_calculator_options``.
    """
    from alframework.ml_interfaces.ntk.estimator import NTKPackedEvaluator

    estimator = load_hippynn_excited_state_ntk_estimator(model_directory, properties_list, device=device, **fit)
    return NTKPackedEvaluator(estimator, selected_state=selected_state, crossing_gap_eV=crossing_gap_eV,
                              **(scores or {}))


# -- Graphs whose outputs are not energies, such as a coupling model on a frozen energy model -----------

class HippynnOutputsNTKAdapter(NTKAdapter):
    """A HIPPYNN graph scored on its outputs, for example a scaled-coupling model (Model B).

    The graph takes species and positions, plus any other input nodes. ``extra_inputs(structure)`` returns
    those inputs for one structure as ``{db_name: tensor}`` with a leading batch axis of 1, for example
    gradient differences and gaps from a frozen energy model (see :func:`frozen_model_inputs`). The
    outputs are ``nodes_to_compute``, named by their ``db_name`` (or ``name``), and are scored with
    ``NTKEstimator.score(outputs=...)``. They are fitted with ``NTKEstimator(output_scales=...)``. The
    adapter has no states: it predicts no energies, forces, or Hessians.
    """

    state_names = ()

    @torch.inference_mode(False)
    def __init__(self, graph, *, extra_inputs=None, device="cpu", dtype=None, parameters="trainable",
                 supports_forward=False):
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        graph = deepcopy(graph).to(device=self.device, dtype=self.dtype).eval()
        roles = []
        for node in graph.input_nodes:
            if isinstance(node, hippynn_inputs.SpeciesNode):
                roles.append("species")
            elif isinstance(node, hippynn_inputs.PositionsNode):
                roles.append("positions")
            else:
                roles.append(f"extra:{node.db_name or node.name}")
        if "species" not in roles or "positions" not in roles:
            raise ValueError("The graph must take species and positions.")
        self.extra_names = tuple(role[len("extra:"):] for role in roles if role.startswith("extra:"))
        if self.extra_names and extra_inputs is None:
            raise ValueError(f"The graph also takes {list(self.extra_names)}; give extra_inputs to supply them.")
        self.roles, self.extra_inputs = tuple(roles), extra_inputs
        self.output_names = tuple(node.db_name or node.name for node in graph.nodes_to_compute)
        self.supports_forward = bool(supports_forward)
        self.graph = graph
        self._functional = _Functional(graph)
        try:
            self._parameters = select_parameters(graph, parameters)
        except ValueError:
            raise ValueError("The graph has no trainable parameters.") from None
        self._digest = _graph_digest(graph)

    def live_parameters(self):
        return self._parameters

    def digest(self):
        return self._digest

    def prepare(self, structure):
        if bool(structure.pbc.any()):
            raise ValueError("HippynnOutputsNTKAdapter scores nonperiodic structures only.")
        if self.extra_names:
            values = self.extra_inputs(structure)
            missing = set(self.extra_names) - set(values)
            if missing:
                raise ValueError(f"extra_inputs gave no {sorted(missing)}.")
            structure.extra["inputs"] = {name: torch.as_tensor(values[name]).to(self.device).detach().clone()
                                         for name in self.extra_names}

    def energies(self, structure, positions, params=None):
        return positions.new_zeros(0)

    def outputs(self, structure, positions, params=None, names=()):
        unknown = set(names) - set(self.output_names)
        if unknown:
            raise ValueError(f"The graph has no outputs {sorted(unknown)}; it has {list(self.output_names)}.")
        values = {"species": structure.numbers[None], "positions": positions[None]}
        for name in self.extra_names:
            tensor = structure.extra["inputs"][name]
            values[f"extra:{name}"] = tensor.to(positions.dtype) if tensor.is_floating_point() else tensor
        arguments = tuple(values[role] for role in self.roles)
        if params is None:
            outputs = self._functional(*arguments)
        else:
            outputs = functional_call(self._functional, {f"graph.{k}": v for k, v in params.items()}, arguments)
        return {name: output[0] for name, output in zip(self.output_names, outputs) if name in names}

    def forward_context(self):
        return scoring_kernels()

    def reverse_context(self):
        return scoring_kernels()


def frozen_model_inputs(model_adapter, names):
    """``extra_inputs`` built from a frozen multi-state energy model's predictions.

    ``names`` maps each extra input's ``db_name`` to ``("GDV", i, j)`` for the gradient difference F_i − F_j,
    flattened to ``[1, 3N]``, or to ``("GAP", i, j)`` for E_j − E_i, as ``[1]``. States are indices into
    ``model_adapter.state_names``.
    """
    for kind, i, j in names.values():
        if kind not in ("GDV", "GAP"):
            raise ValueError(f"Input kinds are GDV or GAP, not {kind!r}.")

    def inputs(structure):
        energies, forces = model_adapter.predict_batch(structure.numbers, structure.positions, None, structure.pbc,
                                                       [structure.n_atoms])
        values = {}
        for name, (kind, i, j) in names.items():
            if kind == "GDV":
                values[name] = (forces[i] - forces[j]).reshape(1, -1)
            else:
                values[name] = (energies[j] - energies[i]).reshape(1)
        return values

    return inputs
