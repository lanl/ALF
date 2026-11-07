"""Evaluate an ensemble of TorchSim models on a whole batch, for ALF's batched sampling engines.

Fine-tuned MACE and UMA copies are both TorchSim models, so their ``batched_calculator`` loaders
build their models and hand them to :class:`TorchSimBatchedEnsemble`. Nothing imports Torch or
TorchSim until an ensemble is built.
"""


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
        if len(models) < 2:
            raise ValueError(f'An uncertainty ensemble requires at least two fine-tuned copies in {source}.')
        self._members = list(models)
        self.n_models = len(self._members)
        self.charge, self.spin = charge, spin

    def state(self, numbers, positions, cells, pbc, counts):
        import torch_sim as ts
        torch = self._torch
        counts = torch.as_tensor(counts, device=self.device)
        n_systems = len(counts)
        extras = {}
        if self.charge is not None:
            extras['charge'] = torch.full((n_systems,), int(self.charge), device=self.device)
        if self.spin is not None:
            extras['spin'] = torch.full((n_systems,), int(self.spin), device=self.device)
        positions = positions.detach().to(self.device, self.dtype)
        return ts.SimState(positions=positions, masses=torch.ones(len(positions), device=self.device, dtype=self.dtype),
                           cell=cells.detach().to(self.device, self.dtype).transpose(-1, -2),
                           pbc=torch.as_tensor([bool(value) for value in pbc], device=self.device),
                           atomic_numbers=numbers.to(self.device, torch.long),
                           system_idx=torch.repeat_interleave(torch.arange(n_systems, device=self.device), counts),
                           _system_extras=extras)

    def evaluate(self, numbers, positions, cells, pbc, counts):
        torch = self._torch
        state = self.state(numbers, positions, cells, pbc, counts)
        energies, forces = [], []
        for model in self._members:
            output = model(state.clone())
            energies.append(output['energy'].detach().reshape(state.n_systems).to(self.dtype))
            forces.append(output['forces'].detach().reshape(len(state.positions), 3).to(self.dtype))
        return torch.stack(energies), torch.stack(forces)
