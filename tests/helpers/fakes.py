import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class FakeASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        atoms=None,
        directory=None,
        command=None,
        fake_results=None,
        fake_converged=True,
        fake_expected_command=None,
        fake_expected_properties=None,
        **kwargs,
    ):
        super().__init__()
        self.init_atoms = atoms
        self.directory = directory
        self.command = command
        self.kwargs = kwargs
        self.fake_results = fake_results
        self.converged = fake_converged
        self.fake_expected_command = fake_expected_command
        self.fake_expected_properties = fake_expected_properties
        self.calculate_calls = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calculate_calls.append((atoms, list(properties)))
        if self.fake_expected_command is not None:
            assert self.command == self.fake_expected_command
        if self.fake_expected_properties is not None:
            assert list(properties) == list(self.fake_expected_properties)

        if self.fake_results is None:
            self.results = {"energy": -1.25, "forces": np.ones((len(atoms), 3))}
        else:
            self.results = copy.deepcopy(self.fake_results)


class FixedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy, forces):
        super().__init__()
        self.energy = energy
        self.forces = np.array(forces, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.energy
        self.results["forces"] = self.forces.copy()


class FakeUncertaintyCalculator(Calculator):
    implemented_properties = ["energy_stdev", "forces_stdev_mean", "forces_stdev_max"]

    def calculate(self, atoms=None, properties=("energy_stdev",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy_stdev": 0.0,
            "forces_stdev_mean": 0.0,
            "forces_stdev_max": 0.0,
        }


class FakeLangevin:
    def __init__(self, atoms, timestep, friction, temperature_K):
        self.atoms = atoms
        self.timestep = timestep
        self.friction = friction
        self.temperature_K = temperature_K
        self.run_calls = []

    def run(self, steps):
        self.run_calls.append(steps)

    def set_temperature(self, temperature_K):
        self.temperature_K = temperature_K


class FakeTask:
    def __init__(self, status, result=None, done=True, running=False, exception=None):
        self._status = status
        self._result = result
        self._done = done
        self._running = running
        self._exception = exception

    def done(self):
        return self._done

    def running(self):
        return self._running

    def task_status(self):
        return self._status

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


def fake_training_pool(member_log=lambda index: "Training complete\n", result=lambda params: None):
    """Build a ``multiprocessing.Pool`` stand-in that writes member logs instead of training.

    ``member_log(index)`` gives each member's log text, or ``None`` to write no log.
    ``result(params)`` gives the value the worker would have returned. Returns the
    pool class and a namespace recording how the code under test used it.
    """
    record = SimpleNamespace(processes=None, func=None, params_list=None, calls=[])

    class FakePool:
        def __init__(self, processes):
            record.processes = processes

        def map(self, func, params_list):
            record.func, record.params_list = func, params_list
            for index, params in enumerate(params_list):
                member = Path(params["model_dir"])
                member.mkdir(parents=True)
                log = member_log(index)
                if log is not None:
                    (member / "training_log.txt").write_text(log)
            return [result(params) for params in params_list]

        def close(self):
            record.calls.append("close")

        def join(self):
            record.calls.append("join")

    return FakePool, record


def import_neurochem_with_fake_backend(monkeypatch):
    """Import ``neurochem_interface`` against stand-ins, since its backend can't be installed in CI."""
    fake_ase_interface = ModuleType("ase_interface")
    fake_ase_interface.aniensloader = object()
    fake_ase_interface.ANIENS = lambda model: ("ANIENS", model)
    fake_ase_interface.batchedensemblemolecule = lambda *args: ("batchedensemblemolecule", args)

    monkeypatch.setitem(sys.modules, "anitraintools", ModuleType("anitraintools"))
    monkeypatch.setitem(sys.modules, "ase_interface", fake_ase_interface)
    monkeypatch.delitem(sys.modules, "alframework.ml_interfaces.neurochem_interface", raising=False)

    return importlib.import_module("alframework.ml_interfaces.neurochem_interface")


class PairSpringCalculator(Calculator):
    """All-pairs harmonic springs; ensembles differing in ``k`` give bounded MD with real uncertainty."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k, d0=1.2):
        super().__init__()
        self.k = k
        self.d0 = d0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        energy, forces = 0.0, np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                vector = positions[i] - positions[j]
                distance = np.linalg.norm(vector)
                energy += self.k * (distance - self.d0) ** 2
                force = -2 * self.k * (distance - self.d0) * vector / distance
                forces[i] += force
                forces[j] -= force
        self.results = {"energy": energy, "forces": forces}


def spring_ensemble(spread=1.0):
    return [PairSpringCalculator(3.5 + spread * delta) for delta in (-1.5, -0.5, 0.5, 1.5)]


def load_spring_ensemble(model_directory, device="cpu"):
    """Stand-in for an ``ase_calculator`` ensemble loader."""
    return spring_ensemble()


class QuadraticStateEnsemble:
    """Analytic multi-state forces with correlated member energy errors."""

    n_models = 2
    n_states = 3

    def evaluate_states(self, numbers, positions, cells, pbc, counts):
        import torch

        x = torch.as_tensor(positions)
        k = x.new_tensor([[0.5, 1.0, 1.5], [0.8, 1.2, 2.0]])
        system = torch.repeat_interleave(
            torch.arange(len(counts), device=x.device), torch.as_tensor(counts, device=x.device)
        )
        squared = x.new_zeros(len(counts)).index_add_(0, system, (x * x).sum(1))
        offsets = x.new_tensor([0.0, 10.0])[:, None, None]
        return k[:, :, None] * squared + offsets, -2 * k[:, :, None, None] * x[None, None]

    def predict_states(self, atoms_list):
        import torch

        counts = [len(a) for a in atoms_list]
        e, f = self.evaluate_states(
            None, torch.tensor(np.concatenate([a.positions for a in atoms_list])), None, None, counts
        )
        offsets = np.cumsum([0, *counts])
        return dict(
            state_energies=e.numpy(), state_forces=[f[:, :, lo:hi].numpy() for lo, hi in zip(offsets[:-1], offsets[1:])]
        )
