import copy

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class FakeASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms=None, directory=None, command=None, fake_results=None,
                 fake_converged=True, fake_expected_command=None, fake_expected_properties=None, **kwargs):
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


class FakeTask:
    def __init__(self, status, result=None, done=True, running=False):
        self._status = status
        self._result = result
        self._done = done
        self._running = running

    def done(self):
        return self._done

    def running(self):
        return self._running

    def task_status(self):
        return self._status

    def result(self):
        return self._result
