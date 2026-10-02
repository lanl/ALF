import importlib

import numpy as np
import pytest
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


def test_well_potential_defaults_to_mass_weighted_forces():
    atoms = Atoms("HeH", positions=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    calc = Well_Potential(r_start=1.0, force=1.0, origin=[0.0, 0.0, 0.0])

    calc.calculate(atoms, properties=["energy", "forces"])

    masses = atoms.get_masses()
    np.testing.assert_allclose(calc.results["forces"][:, 0], -masses)


def test_well_potential_zero_properties_excludes_core_properties():
    atoms = Atoms("H", positions=[[2.0, 0.0, 0.0]])
    calc = Well_Potential(
        r_start=1.0,
        force=2.0,
        origin=[0.0, 0.0, 0.0],
        zero_properties=["energy", "potential_energy", "forces", "stress"],
    )

    calc.calculate(atoms, properties=["energy", "forces", "stress"])

    assert calc.zero_properties == ["stress"]
    assert "energy" in calc.implemented_properties
    assert "potential_energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties
    assert "stress" in calc.implemented_properties
    assert calc.results["energy"] == 2.0 * atoms.get_masses()[0]
    np.testing.assert_allclose(calc.results["forces"], [[-2.0 * atoms.get_masses()[0], 0.0, 0.0]])
    assert calc.results["stress"] == 0


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


def test_mlmd_calculator_single_model_has_zero_uncertainty(fixed_calculator_factory):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    model = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    calc = MLMD_calculator([model])

    calc.calculate(atoms, ["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"])

    assert calc.results["energy"] == 1.5
    np.testing.assert_allclose(calc.results["forces"], [[0.1, 0.2, 0.3]])
    assert calc.results["energy_stdev"] == 0.0
    assert calc.results["forces_stdev_mean"] == 0.0
    assert calc.results["forces_stdev_max"] == 0.0


def test_mlmd_calculator_without_udd(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator([model_a, model_b])

    calc.calculate(atoms, ["energy", "forces", "energy_stdev"])

    assert "E_en_bias" not in calc.results
    assert "F_en_bias" not in calc.results
    assert calc.results["energy"] == 2.0
    np.testing.assert_allclose(calc.results["forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])


def test_mlmd_calculator_applies_udd_energy_and_force_bias(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator([model_a, model_b], udd_bias_weight=0.5)

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["unbiased_energy"] == 2.0
    np.testing.assert_allclose(calc.results["unbiased_forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert calc.results["E_en_bias"] == 0.5
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert calc.results["energy"] == 2.5
    np.testing.assert_allclose(calc.results["forces"], [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])


def test_mlmd_calculator_udd_zero_disagreement_has_no_bias(fixed_calculator_factory):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    model_a = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    model_b = fixed_calculator_factory(energy=1.5, forces=[[0.1, 0.2, 0.3]])
    calc = MLMD_calculator([model_a, model_b], udd_bias_weight=0.5)

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 0.0
    assert calc.results["E_en_bias"] == 0.0
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.0, 0.0, 0.0]])
    assert calc.results["energy"] == calc.results["unbiased_energy"]
    np.testing.assert_allclose(calc.results["forces"], calc.results["unbiased_forces"])


def test_mlmd_calculator_udd_ignores_well_potential(fixed_calculator_factory):
    atoms = Atoms("H2", positions=[[0.2, 0.0, 0.0], [0.0, 0.0, 0.7]])
    model_a = fixed_calculator_factory(energy=1.0, forces=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    model_b = fixed_calculator_factory(energy=3.0, forces=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    calc = MLMD_calculator(
        [model_a, model_b],
        well_params={"r_start": 0.0, "force": 10.0, "origin": [0.0, 0.0, 0.0], "mass_weighted": False},
        udd_bias_weight=0.5,
    )

    calc.calculate(atoms, ["energy", "forces"])

    assert calc.results["energy_stdev"] == 1.0
    assert calc.results["E_en_bias"] == 0.5
    np.testing.assert_allclose(calc.results["F_en_bias"], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    assert calc.results["unbiased_energy"] > 2.0


def test_mlmd_sampling_updates_density_and_cell_volume(monkeypatch):
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


def test_udd_sampling_task_requires_bias_weight(h2_atoms):
    udd_sampling_module = importlib.import_module("alframework.samplers.udd_sampling")
    molecule = MoleculesObject(h2_atoms, "h2-udd")

    with pytest.raises(ValueError, match="udd_bias_weight"):
        error = udd_sampling_module.simple_udd_sampling_task.func(
            molecule,
            {"MLMD_calculator_options": {"well_params": None}},
            "models/model-{:04d}",
            0,
            1,
        )
        error.reraise()


def test_udd_sampling_task_injects_bias_and_delegates(monkeypatch, h2_atoms):
    udd_sampling_module = importlib.import_module("alframework.samplers.udd_sampling")
    molecule = MoleculesObject(h2_atoms, "h2-udd")
    captured = {}

    def fake_mlmd_task(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
        captured["molecule_object"] = molecule_object
        captured["sampler_config"] = sampler_config
        captured["model_path"] = model_path
        captured["current_model_id"] = current_model_id
        captured["gpus_per_node"] = gpus_per_node
        return molecule_object

    monkeypatch.setattr(udd_sampling_module.simple_mlmd_sampling_task, "func", fake_mlmd_task)

    original_config = {
        "udd_bias_weight": 0.45,
        "MLMD_calculator_options": {"well_params": None},
        "dt": 0.25,
    }
    result = udd_sampling_module.simple_udd_sampling_task.func(
        molecule,
        original_config,
        "models/model-{:04d}",
        7,
        4,
    )

    assert result is molecule
    assert captured["molecule_object"] is molecule
    assert captured["model_path"] == "models/model-{:04d}"
    assert captured["current_model_id"] == 7
    assert captured["gpus_per_node"] == 4
    assert captured["sampler_config"]["dt"] == 0.25
    assert captured["sampler_config"]["MLMD_calculator_options"] == {
        "well_params": None,
        "udd_bias_weight": 0.45,
    }
    assert "udd_bias_weight" not in captured["sampler_config"]
    assert original_config["MLMD_calculator_options"] == {"well_params": None}
