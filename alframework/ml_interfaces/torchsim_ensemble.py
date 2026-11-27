"""Evaluate an ensemble of TorchSim models on a whole batch, for ALF's batched sampling engines.

Fine-tuned MACE and UMA copies are both TorchSim models, so their ``batched_calculator`` loaders
build their models and hand them to :class:`TorchSimBatchedEnsemble`. Nothing imports Torch or
TorchSim until an ensemble is built.
"""


def packed_state(numbers, positions, cells, pbc, counts, *, device, dtype, charge=None, spin=None):
    """Build the same TorchSim input for ordinary and differentiable model evaluation."""
    import torch
    import torch_sim as ts

    counts = torch.as_tensor(counts, device=device)
    n_systems = len(counts)
    extras = {name: torch.full((n_systems,), int(value), device=device)
              for name, value in (('charge', charge), ('spin', spin)) if value is not None}
    positions = positions.detach().to(device, dtype)
    return ts.SimState(positions=positions, masses=torch.ones(len(positions), device=device, dtype=dtype),
                       cell=cells.detach().to(device, dtype).transpose(-1, -2),
                       pbc=torch.as_tensor(pbc, device=device, dtype=torch.bool),
                       atomic_numbers=numbers.to(device, torch.long),
                       system_idx=torch.repeat_interleave(torch.arange(n_systems, device=device), counts,
                                                          output_size=len(numbers)),
                       _system_extras=extras)


class TorchSimBatchedEnsemble:
    """Implements ``evaluate(numbers, positions, cells, pbc, counts)`` for the batched engines.

    ``cells`` are row vectors, as in ASE; TorchSim's are column vectors. The result is
    ``(energies [models, systems], forces [models, atoms, 3])`` in eV and eV/Å. ``charge`` and
    ``spin``, when given, apply to every system, because the engines don't pass them per structure.
    """

    def __init__(self, models, device='cuda:0', dtype='float32', charge=None, spin=None, source='this directory'):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        # One copy is allowed; a run whose acquisition reads the spread refuses it (acquisition.require_members).
        if not models:
            raise ValueError(f'No fine-tuned copies in {source}.')
        self._members = list(models)
        self.n_models = len(self._members)
        self.charge, self.spin = charge, spin

    def state(self, numbers, positions, cells, pbc, counts):
        return packed_state(numbers, positions, cells, pbc, counts, device=self.device, dtype=self.dtype,
                            charge=self.charge, spin=self.spin)

    def evaluate(self, numbers, positions, cells, pbc, counts):
        torch = self._torch
        state = self.state(numbers, positions, cells, pbc, counts)
        energies, forces = [], []
        for model in self._members:
            output = model(state.clone())
            energies.append(output['energy'].detach().reshape(state.n_systems).to(self.dtype))
            forces.append(output['forces'].detach().reshape(len(state.positions), 3).to(self.dtype))
        return torch.stack(energies), torch.stack(forces)
