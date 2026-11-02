"""TorchSim engine for batched multi-trajectory MLMD (``driver.engine: torchsim``).

:class:`TorchSimEngine` implements the engine contract of
:class:`alframework.samplers.batched_mlmd.ASEEngine`, so
``run_mlmd_trajectories`` drives it unchanged: the same checks, schedules,
stopping, acquisition, and cohorts. All active trajectories are packed into one
TorchSim ``MDState`` and every MD step makes one batched ensemble evaluation.

Integration is Langevin BAOAB (Leimkuhler and Matthews) built from TorchSim's
``momentum_step`` and ``position_step``. The stochastic step draws each
trajectory's noise from that trajectory's own random stream (seeded from its
molecule ID, as in the ASE engine). With exact inference a trajectory's result
therefore does not depend on which other trajectories share its task.
Finished trajectories are removed from the packed state rather than frozen.
As in ASE's ``Langevin`` (``fixcm=True``) the noise carries no net momentum,
and the friction is the legacy MLMD value of 0.02 inverse ASE time units.

Units are those of ASE: Angstrom, eV, amu, and ASE time; ``dt`` is in fs.

Forces used for propagation are the ensemble mean plus the optional well
potential and UDD bias (``MLMD_calculator_options``), with the formulas of
``MLMD_calculator``. Screening reads the unbiased per-member predictions.

Inference comes from a batched evaluator (``evaluate(numbers, positions,
cells, pbc, counts) -> (energies [models, systems], forces [models, atoms,
3])``, e.g. ``hippynn_interface.load_hippynn_batched_ensemble``), selected
by the sampler's ``batched_calculator``. Without one, the ``ase_calculator``
ensemble is evaluated one structure at a time.
"""

import numpy as np
from ase import units
from ase.io.trajectory import Trajectory

from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator
from alframework.samplers.batched_mlmd import minimum_distance

KB_EV_PER_K = 8.617333262145e-5
CALCULATOR_OPTIONS = ("well_params", "udd_bias_weight", "udd_bias_eps", "debug_print")
WELL_PARAMS = ("r_start", "force", "origin", "zero_properties", "mass_weighted")


def require_torchsim():
    """Import TorchSim, with an actionable error when the optional engine is missing."""
    try:
        import torch
        from torch_sim.integrators.md import MDState, momentum_step, position_step
    except ImportError as exc:
        raise ImportError("driver.engine 'torchsim' requires torch and torch-sim-atomistic (Python 3.12+) in the "
                          "sampler worker environment; install ALF's 'torchsim' extra.") from exc
    return torch, MDState, momentum_step, position_step


class ASECalculatorBatch:
    """Batched-evaluator adapter for a list of ASE calculators (sequential)."""

    def __init__(self, calculators):
        self.evaluator = ASEEnsembleEvaluator(calculators)
        self.n_models = self.evaluator.n_models

    def evaluate(self, numbers, positions, cells, pbc, counts):
        import torch
        from ase import Atoms
        numbers, positions, cells = numbers.cpu().numpy(), positions.detach().cpu().numpy(), cells.cpu().numpy()
        offsets = np.cumsum([0, *counts])
        atoms_list = [Atoms(numbers=numbers[start:end], positions=positions[start:end], cell=cell,
                            pbc=np.asarray(pbc.cpu() if hasattr(pbc, "cpu") else pbc))
                      for start, end, cell in zip(offsets[:-1], offsets[1:], cells)]
        prediction = self.evaluator.predict(atoms_list)
        return (torch.as_tensor(prediction["energies"]),
                torch.as_tensor(np.concatenate(prediction["forces"], axis=1)))


class TorchSimEngine:
    """Packed Langevin dynamics for several trajectories on one device."""

    def __init__(self, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, friction=0.02,
                 device="cpu"):
        torch, MDState, self._momentum_step, self._position_step = require_torchsim()
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.evaluator = models if hasattr(models, "evaluate") else ASECalculatorBatch(list(models))
        self.n_models = self.evaluator.n_models
        self._atoms = list(atoms_list)
        if len({tuple(bool(v) for v in atoms.pbc) for atoms in self._atoms}) > 1:
            raise ValueError("A TorchSim batch must share one periodic boundary condition.")
        if any(atoms.constraints for atoms in self._atoms):
            raise NotImplementedError("The TorchSim engine does not support ASE constraints.")
        self._options = _calculator_options(calculator_options)
        self._rngs = list(rngs)
        self._temperatures = [float(value) for value in temperatures_K]
        self._dt = float(dt) * units.fs
        self._friction = float(friction)
        self._writers = {}
        self.steps = [0] * len(self._atoms)

        def tensor(values, dtype=self.dtype):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        counts = [len(atoms) for atoms in self._atoms]
        self.state = MDState(
            positions=tensor(np.concatenate([atoms.get_positions() for atoms in self._atoms])),
            masses=tensor(np.concatenate([atoms.get_masses() for atoms in self._atoms])),
            # TorchSim stores lattice vectors as columns; ASE as rows.
            cell=tensor(np.stack([atoms.cell.array.T for atoms in self._atoms])),
            pbc=torch.as_tensor(self._atoms[0].pbc, device=self.device),
            atomic_numbers=tensor(np.concatenate([atoms.numbers for atoms in self._atoms]), torch.long),
            system_idx=torch.repeat_interleave(torch.arange(len(counts), device=self.device),
                                               torch.as_tensor(counts, device=self.device)),
            momenta=tensor(np.concatenate([atoms.get_momenta() for atoms in self._atoms])),
            energy=torch.zeros(len(counts), dtype=self.dtype, device=self.device),
            forces=torch.zeros((sum(counts), 3), dtype=self.dtype, device=self.device))
        self._rows = list(range(len(self._atoms)))
        self._layout()
        self._host = None
        self._evaluate()

    # -- packed layout ---------------------------------------------------------------------------

    def _layout(self):
        counts = [len(self._atoms[index]) for index in self._rows]
        offsets = np.cumsum([0, *counts])
        self._counts = counts
        self._slices = {index: slice(int(start), int(end))
                        for index, start, end in zip(self._rows, offsets[:-1], offsets[1:])}
        self._position = {index: row for row, index in enumerate(self._rows)}

    def _drop(self, indices):
        """Copy members back to their Atoms, then remove them from the packed state."""
        for index in indices:
            self._sync(index)
        keep = [row for row, index in enumerate(self._rows) if index not in set(indices)]
        self._rows = [self._rows[row] for row in keep]
        if self._rows:
            selection = self._torch.as_tensor(keep, device=self.device)
            atom_mask = self._torch.isin(self.state.system_idx, selection)
            energies, forces = self._members
            self.state = self.state[keep]
            self._members = (energies[:, selection], forces[:, atom_mask])
        else:
            self.state, self._members = None, None
        self._host = None
        self._layout()

    def _host_arrays(self):
        if self._host is None:
            energies, forces = self._members
            self._host = {"energies": energies.cpu().numpy(), "forces": forces.cpu().numpy(),
                          "positions": self.state.positions.cpu().numpy(),
                          "momenta": self.state.momenta.cpu().numpy(),
                          "cell": self.state.cell.transpose(-1, -2).cpu().numpy()}
        return self._host

    def _sync(self, index):
        if index not in self._position:
            return
        host, atoms = self._host_arrays(), self._atoms[index]
        selection = self._slices[index]
        atoms.set_cell(host["cell"][self._position[index]], scale_atoms=False)
        atoms.set_positions(host["positions"][selection])
        atoms.set_momenta(host["momenta"][selection])

    # -- forces ----------------------------------------------------------------------------------

    def _evaluate(self):
        torch, state = self._torch, self.state
        energies, forces = self.evaluator.evaluate(state.atomic_numbers, state.positions,
                                                   state.cell.transpose(-1, -2), state.pbc, self._counts)
        energies = energies.to(self.device, self.dtype)
        forces = forces.to(self.device, self.dtype)
        self._members = (energies, forces)
        self._host = None
        self._stale = False
        energy, force = energies.mean(0), forces.mean(0)
        well = self._options["well_params"]
        if well is not None:
            well_energy, well_force = self._well(well)
            energy, force = energy + well_energy, force + well_force
        weight = self._options["udd_bias_weight"]
        if weight is not None:
            spread = energies.std(0, unbiased=False)
            deviations = energies - energies.mean(0)
            bias = (deviations[:, state.system_idx, None] * (forces - forces.mean(0))).sum(0)
            scale = torch.where(spread > self._options["udd_bias_eps"], weight / (self.n_models * spread),
                                torch.zeros_like(spread))
            energy = energy + weight * spread
            force = force + bias * scale[state.system_idx, None]
        state.energy, state.forces = energy, force
        bad_atoms = (~torch.isfinite(force)).any(1).to(self.dtype)
        bad = ~torch.isfinite(energy) | (torch.zeros_like(energy).index_add_(0, state.system_idx, bad_atoms) > 0)
        return [self._rows[row] for row in torch.nonzero(bad).flatten().tolist()]

    def _well(self, params):
        torch, state = self._torch, self.state
        relative = state.positions - torch.as_tensor(params.get("origin", [0, 0, 0]), dtype=self.dtype,
                                                     device=self.device)
        weights = state.masses if params.get("mass_weighted", True) else torch.ones_like(state.masses)
        distance = relative.norm(dim=1)
        depth = (distance - params["r_start"]).clamp(min=0)
        outside = depth > 0
        energy = torch.zeros_like(state.energy).index_add_(0, state.system_idx, weights * depth * params["force"])
        unit = relative / (distance + 1e-4 * (~outside & (distance - params["r_start"] < 0)))[:, None]
        force = -(unit * (outside * weights * params["force"])[:, None])
        return energy, force

    # -- integration -----------------------------------------------------------------------------

    def _noise(self):
        """Standard normals for every active atom, each trajectory from its own stream."""
        draws = [self._rngs[index].standard_normal(size=(len(self._atoms[index]), 3)) for index in self._rows]
        return self._torch.as_tensor(np.concatenate(draws), dtype=self.dtype, device=self.device)

    def _ou(self):
        """Ornstein-Uhlenbeck momentum update; the kick carries no net momentum per system (fixcm)."""
        torch, state = self._torch, self.state
        kT = torch.as_tensor([self._temperatures[index] * KB_EV_PER_K for index in self._rows],
                             dtype=self.dtype, device=self.device)
        c1 = float(np.exp(-self._friction * self._dt))
        kick = torch.sqrt(kT * (1 - c1 * c1))[state.system_idx, None] * torch.sqrt(state.masses)[:, None] \
            * self._noise()
        counts = torch.as_tensor(self._counts, dtype=self.dtype, device=self.device)
        net = torch.zeros((len(self._rows), 3), dtype=self.dtype, device=self.device).index_add_(
            0, state.system_idx, kick)
        factor = torch.where(counts > 1, torch.sqrt(counts / (counts - 1).clamp(min=1)), torch.ones_like(counts))
        kick = (kick - (net / counts[:, None])[state.system_idx]) * factor[state.system_idx, None]
        state.momenta = c1 * state.momenta + kick

    def _baoab(self):
        half = self._dt / 2
        self._momentum_step(self.state, half)
        self._position_step(self.state, half)
        self._ou()
        self._position_step(self.state, half)
        bad = self._evaluate()
        self._momentum_step(self.state, half)
        return bad

    # -- engine contract -------------------------------------------------------------------------

    def step(self, indices, n):
        failures = {}
        retired = [index for index in self._rows if index not in set(indices)]
        if retired:
            self._drop(retired)
        if not self._rows:
            return failures
        try:
            if self._stale:
                bad = self._evaluate()
                if bad:
                    failures.update({index: FloatingPointError("non-finite ensemble prediction") for index in bad})
                    self._drop(bad)
            for _ in range(n):
                if not self._rows:
                    break
                bad = self._baoab()
                for index in self._rows:
                    self.steps[index] += 1
                if bad:
                    failures.update({index: FloatingPointError("non-finite ensemble prediction") for index in bad})
                    self._drop(bad)
                self._write_trajectories()
        except Exception as exc:
            failures.update({index: exc for index in self._rows})
            self._drop(list(self._rows))
        return failures

    def observe(self, index):
        if self._stale:
            self._evaluate()
        host = self._host_arrays()
        selection = self._slices[index]
        prediction = {"energies": host["energies"][:, [self._position[index]]].astype(float),
                      "forces": [host["forces"][:, selection].astype(float)]}
        return {"prediction": prediction, "min_distance": minimum_distance(self.atoms(index))}

    def atoms(self, index):
        self._sync(index)
        return self._atoms[index]

    def set_temperature(self, index, temperature_K):
        self._temperatures[index] = float(temperature_K)

    def scale_cell(self, index, scalar):
        """Scale one system's cell and positions; forces are refreshed before the next step."""
        selection, row = self._slices[index], self._position[index]
        self.state.positions[selection] *= scalar
        self.state.cell[row] *= scalar
        self._host = None
        self._stale = True

    def finish(self, index):
        if index in self._position:
            self._drop([index])
        writer = self._writers.pop(index, None)
        if writer is not None:
            writer[0].close()

    def attach_trajectory(self, index, path, interval):
        """Write the initial frame, then every ``interval`` steps (positions and momenta only)."""
        writer = Trajectory(path, mode="w")
        writer.write(self.atoms(index))
        self._writers[index] = (writer, int(interval))

    def _write_trajectories(self):
        for index, (writer, interval) in self._writers.items():
            if index in self._position and self.steps[index] % interval == 0:
                writer.write(self.atoms(index))


def _calculator_options(options):
    options = dict(options or {})
    unknown = set(options) - set(CALCULATOR_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown MLMD_calculator_options for the TorchSim engine: {sorted(unknown)}.")
    well = options.get("well_params")
    if well is not None:
        unknown = set(well) - set(WELL_PARAMS)
        if unknown:
            raise ValueError(f"Unknown well_params: {sorted(unknown)}.")
    return {"well_params": None if well is None else dict(well),
            "udd_bias_weight": options.get("udd_bias_weight"),
            "udd_bias_eps": float(options.get("udd_bias_eps", 1e-12))}
