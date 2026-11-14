"""Fixtures shared across the test suite."""

import pytest
from ase import Atoms

from tests.helpers.engines import require_engine
from tests.helpers.fakes import FixedCalculator


@pytest.fixture
def h2_atoms():
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def water_atoms():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]])


@pytest.fixture
def fixed_calculator_factory():
    return FixedCalculator


@pytest.fixture(params=["torchsim", "alchemi"])
def engine_name(request):
    """Run the test once per packed engine, skipping any engine that isn't installed."""
    require_engine(request.param)
    return request.param


@pytest.fixture(scope="session")
def trained_ensemble(tmp_path_factory):
    """A real two-member HIPPYNN water ensemble, trained on CPU once per session.

    Parity tests compare code paths on the same trained model; they do not need
    an accurate potential, so no external checkpoint is required.
    """
    pytest.importorskip("torch")
    pytest.importorskip("hippynn")
    from tests.helpers.hippynn import train_cpu_ensemble

    directory = tmp_path_factory.mktemp("trained") / "ensemble"
    train_cpu_ensemble(directory)
    return directory


@pytest.fixture(scope="session")
def trained_periodic_ensemble(tmp_path_factory):
    """Periodic-trained vector HIPPYNN members, including a real cell input and periodic pair finder."""
    pytest.importorskip("torch")
    pytest.importorskip("hippynn")
    from tests.helpers.hippynn import train_cpu_ensemble

    directory = tmp_path_factory.mktemp("trained-periodic") / "ensemble"
    train_cpu_ensemble(directory, periodic=True)
    return directory


@pytest.fixture(params=["trained_ensemble", "trained_periodic_ensemble"])
def neighbor_ensemble(request):
    return request.getfixturevalue(request.param)
