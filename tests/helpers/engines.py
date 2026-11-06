"""Shared pieces for tests that run on every packed dynamics engine.

Nothing here imports torch at module level, so test files using it still run in
the torch-free CI job. An engine's packages are only needed once a test asks for it.
"""

import os
from functools import partial
from importlib.metadata import PackageNotFoundError, version

import pytest
from ase.calculators.calculator import Calculator, all_changes

DEVICE = os.environ.get("ALF_TEST_ENGINE_DEVICE", "cpu")


def require_engine(name):
    """Skip the calling test unless this packed engine is installed at a supported version."""
    if name == "torchsim":
        pytest.importorskip("torch_sim")
    elif name == "alchemi":
        from alframework.samplers.alchemi_engine import SUPPORTED_VERSIONS

        try:
            installed = version("nvalchemi-toolkit")
        except PackageNotFoundError:
            pytest.skip("nvalchemi-toolkit is not installed")
        if installed not in SUPPORTED_VERSIONS:
            pytest.skip(f"nvalchemi-toolkit {installed} is not one of {SUPPORTED_VERSIONS}")


def engine_class(name):
    if name == "torchsim":
        from alframework.samplers.torchsim_engine import TorchSimEngine

        return partial(TorchSimEngine, device=DEVICE)
    from alframework.samplers.alchemi_engine import AlchemiEngine

    return partial(AlchemiEngine, device=DEVICE)


class Tether(Calculator):
    """Harmonic tether to the origin, for thermostat checks."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k):
        super().__init__()
        self.k = k

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.get_positions()
        self.results = {"energy": 0.5 * self.k * (x**2).sum(), "forces": -self.k * x}
