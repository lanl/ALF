"""Teachers for tests: a real TorchSim model, an ASE calculator, and a scripted stand-in."""

import numpy as np

from alframework.qm_interfaces.teacher_interface import AseTeacher, TorchSimTeacher
from tests.helpers.fakes import PairSpringCalculator

LOADS = []


def load_lennard_jones_teacher(settings, device):
    """A real TorchSim model, so batching is exercised without MACE or fairchem."""
    import torch
    from torch_sim.models.lennard_jones import LennardJonesModel

    LOADS.append(settings["name"])
    model = LennardJonesModel(sigma=1.5, epsilon=0.05, cutoff=5.0, device=torch.device(device),
                              dtype=torch.float64, compute_forces=True)
    return TorchSimTeacher(model, "torch-sim-atomistic", settings["options"].get("elements"))


def load_spring_teacher(settings, device):
    """An ASE calculator teacher that needs no torch."""
    LOADS.append(settings["name"])
    return AseTeacher(PairSpringCalculator(3.5), "ase", settings["options"].get("elements"),
                      settings["options"].get("charge_spin", False))


class ScriptedTeacher:
    """Answers from a rule, and records each call, to test failures and grouping."""

    package = "tests"

    def __init__(self, options):
        self.options, self.calls, self.infos = options, [], []
        self.supported_numbers = None if options.get("elements") is None else set(options["elements"])
        self.supports_charge_spin = options.get("charge_spin", False)

    def evaluate(self, atoms_list):
        self.calls.append([len(atoms) for atoms in atoms_list])
        self.infos.extend(dict(atoms.info) for atoms in atoms_list)
        if any(len(atoms) == self.options.get("fail_on_size") for atoms in atoms_list):
            raise RuntimeError("unsupported structure")
        energies = [self.options.get("energy", -1.0) * len(atoms) for atoms in atoms_list]
        forces = [np.full((len(atoms), 3), self.options.get("force", 0.1)) for atoms in atoms_list]
        return energies, forces


def load_scripted_teacher(settings, device):
    LOADS.append(settings["name"])
    return ScriptedTeacher(settings["options"])


def load_broken_teacher(settings, device):
    raise RuntimeError("checkpoint is corrupt")


class StatefulTeacher(ScriptedTeacher):
    """Like a MACE model, it keeps state between calls, so overlapping calls corrupt each other."""

    def __init__(self, options):
        super().__init__(options)
        self.busy = False

    def evaluate(self, atoms_list):
        import time

        if self.busy:
            raise AttributeError("model state was changed by another call")
        self.busy = True
        try:
            time.sleep(0.005)
            return super().evaluate(atoms_list)
        finally:
            self.busy = False


def load_stateful_teacher(settings, device):
    LOADS.append(settings["name"])
    return StatefulTeacher(settings["options"])
