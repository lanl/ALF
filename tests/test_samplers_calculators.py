import numpy as np
from ase import Atoms

from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator, Well_Potential
from tests.helpers.interface_checks import check_ase_calculator_interface, check_energy_force_results


def test_well_potential_applies_restoring_force_outside_radius():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    calc = Well_Potential(r_start=1.0, force=2.0, origin=[0.0, 0.0, 0.0], mass_weighted=False)

    calc.calculate(atoms, properties=["energy", "forces"])

    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])


def test_well_potential_zero_properties_do_not_duplicate_core_properties():
    calc = Well_Potential(
        r_start=1.0,
        force=1.0,
        zero_properties=["energy", "forces", "potential_energy", "stress"],
    )

    assert "stress" in calc.implemented_properties
    assert calc.implemented_properties.count("energy") == 1
    assert calc.implemented_properties.count("forces") == 1


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
