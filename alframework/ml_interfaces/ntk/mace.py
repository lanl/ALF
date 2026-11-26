"""MACE models for ALF's NTK uncertainty, molecular or periodic.

MACE's forward pass calls ``positions.requires_grad_()``, which ``torch.func`` transforms forbid. So MACE
rows use batched reverse mode, falling back to one backward pass per output where needed. The neighbor
list is built once per scored geometry, and its fixed edges and shifts are differentiated through MACE's
smooth cutoff.
"""

from copy import deepcopy
import hashlib

import numpy as np
import torch
from ase import Atoms
from torch.func import functional_call

from alframework.ml_interfaces.ntk.adapter import NTKAdapter, resolve_device_dtype, select_parameters


def _mace():
    try:
        from mace import data, tools
        from mace.tools import torch_geometric
    except ImportError as exc:
        raise ImportError("MACE NTK uncertainty requires mace-torch.") from exc
    return data, tools, torch_geometric


def _model_digest(model):
    digest = hashlib.sha256(repr((type(model).__module__, type(model).__qualname__)).encode())
    for name, parameter in model.named_parameters():
        digest.update(repr((name, parameter.requires_grad)).encode())
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(repr((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class MACENTKAdapter(NTKAdapter):
    """A frozen copy of a MACE model on the scoring device and precision.

    ``head`` selects the output head of a multi-head model. The default is the model's first head.
    ``parameters`` chooses what the posterior covers (see :func:`~alframework.ml_interfaces.ntk.adapter.
    select_parameters`). The default keeps the checkpoint's trainable parameters, so a model fine-tuned
    with frozen layers or LoRA is linearized only in what was trained. A model with several energies
    names them ``E0, E1, …`` unless ``state_names`` is given.
    """

    supports_forward = False

    @torch.inference_mode(False)
    def __init__(self, model, *, device="cpu", dtype=None, head=None, parameters="trainable", state_names=None):
        data, tools, _ = _mace()
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        self.model = deepcopy(model).to(device=self.device, dtype=self.dtype).eval()
        self._parameters = select_parameters(self.model, parameters)
        n_states = int(getattr(self.model, "n_energies", 1) or 1)
        if state_names is None:
            state_names = ("energy",) if n_states == 1 else tuple(f"E{i}" for i in range(n_states))
        if len(state_names) != n_states:
            raise ValueError(f"The model predicts {n_states} energies but {len(state_names)} state names were given.")
        self.state_names = tuple(state_names)
        self.z_table = tools.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.cutoff = float(self.model.r_max)
        heads = list(getattr(self.model, "heads", None) or [])
        if head is not None and head not in heads:
            raise ValueError(f"MACE head {head!r} is not one of {heads}.")
        self.heads = heads or None
        self.head = head if head is not None else (heads[0] if heads else None)
        self._digest = _model_digest(self.model)

    @classmethod
    def from_checkpoint(cls, path, **options):
        """Load a whole-model MACE checkpoint, as written by MACE's training script."""
        _mace()  # MACE allowlists e3nn's constants for torch.load before e3nn is imported.
        return cls(torch.load(path, map_location="cpu", weights_only=False), **options)

    def live_parameters(self):
        return self._parameters

    def digest(self):
        return self._digest

    def _graph(self, items):
        _, _, torch_geometric = _mace()
        batch = torch_geometric.batch.Batch.from_data_list(items).to(self.device)
        graph = batch.to_dict()
        return {key: value.to(self.dtype) if torch.is_tensor(value) and value.is_floating_point() else value
                for key, value in graph.items()}

    def _item(self, numbers, positions, cell, pbc):
        data, _, _ = _mace()
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        config = data.config_from_atoms(atoms)
        options = {"heads": self.heads} if self.heads else {}
        if self.heads:
            config.head = self.head
        return data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.cutoff, **options)

    def prepare(self, structure):
        missing = set(structure.numbers.tolist()) - set(self.z_table.zs)
        if missing:
            raise ValueError(f"The MACE model has no species {sorted(missing)}.")
        cell = structure.cell.cpu().numpy() if bool(structure.pbc.any()) else None
        structure.extra["mace"] = self._graph([self._item(
            structure.numbers.cpu().numpy(), structure.positions.cpu().double().numpy(), cell,
            structure.pbc.cpu().numpy())])

    def energies(self, structure, positions, params=None):
        graph = dict(structure.extra["mace"])
        graph["positions"] = positions
        options = {"compute_force": False, "training": False}
        if params is None:
            output = self.model(graph, **options)
        else:
            output = functional_call(self.model, params, (graph,), options)
        energy = output["energy"].reshape(-1)
        if len(energy) != len(self.state_names):
            raise ValueError(f"The model returned {len(energy)} energies for {len(self.state_names)} states.")
        return energy

    @torch.inference_mode(False)
    def predict_batch(self, numbers, positions, cells, pbc, counts):
        pbc = torch.as_tensor(pbc).cpu().numpy()
        items, start = [], 0
        numbers, positions = numbers.cpu().numpy(), positions.detach().cpu().double().numpy()
        for index, count in enumerate(counts):
            stop = start + count
            cell = None if cells is None or not pbc.any() else torch.as_tensor(cells[index]).cpu().numpy()
            items.append(self._item(numbers[start:stop], positions[start:stop], cell, pbc))
            start = stop
        graph = self._graph(items)
        with torch.inference_mode(False), torch.enable_grad():
            output = self.model(graph, compute_force=True, training=False)
        energies, forces = output["energy"].detach(), output["forces"].detach()
        if energies.ndim == 1:
            return energies[None], forces[None]
        # Several states: energies [systems, S] and forces [atoms, S, 3].
        return energies.T.contiguous(), forces.permute(1, 0, 2).contiguous()
