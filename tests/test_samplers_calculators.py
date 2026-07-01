import importlib

import numpy as np
from ase import Atoms

from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator, Well_Potential
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import FakeLangevin, FakeUncertaintyCalculator
from tests.helpers.interface_checks import check_ase_calculator_interface, check_energy_force_results


def test_well_potential():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    calc = Well_Potential(r_start=1.0, force=2.0, origin=[0.0, 0.0, 0.0], mass_weighted=False)

    calc.calculate(atoms, properties=["energy", "forces"])

    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])


def test_mlmd_calculator_reports_ensemble_means_and_uncertainties(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator([model_a, model_b])

    check_ase_calculator_interface(calc, atoms, ["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"])
    check_energy_force_results(calc.results, len(atoms))
    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["forces_stdev_max"] == 1.0
    assert calc.results["forces_stdev_mean"] == 1.0 / 3.0


def test_mlmd_sampling_rescales_cell_for_density_fluctuation(monkeypatch):
    mlmd_sampling_module = importlib.import_module("alframework.samplers.mlmd_sampling")

    monkeypatch.setattr(mlmd_sampling_module, "Langevin", FakeLangevin)

    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    molecule = MoleculesObject(atoms, "h2-density")
    mass = np.sum(atoms.get_masses())
    initial_volume = atoms.get_volume()
    density_unit_conversion = 1.66054e-24 / 1.0e-24
    starting_density_value = density_unit_conversion * mass / initial_volume

    result = mlmd_sampling_module.mlmd_sampling(
        molecule,
        FakeUncertaintyCalculator(),
        dt=1000.0,
        maxt=2.0,
        Escut=1.0,
        Fscut=1.0,
        Ncheck=1,
        Tamp=0.0,
        Tper=1.0,
        Tsrt=300.0,
        Tend=300.0,
        Ramp=0.2,
        Rper=2.0,
        Rend=2.0,
        distcut=0.1,
    )

    expected_final_density = (1.0e-24 / 1.66054e-24) * annealing_schedule(
        t=1.0,
        tmax=2.0,
        amp=0.2,
        per=2.0,
        srt=starting_density_value,
        end=2.0,
    )

    assert result.get_atoms() is None
    assert len(result.get_metadata()["denss"]) == 2
    assert np.isclose(result.get_metadata()["denss"][-1], expected_final_density)
    assert np.isclose(result.get_metadata()["cell"].volume, mass / expected_final_density)
