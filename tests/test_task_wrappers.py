import numpy as np

from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import FakeASECalculator
from tests.helpers.interface_checks import (
    check_energy_force_results,
    check_molecule_result,
    check_scratch_directory,
    check_task_convergence,
)


def test_ase_calculator_task_uses_mocked_calculator(tmp_path, monkeypatch, h2_atoms):
    from alframework.qm_interfaces import ase_calculator_interface

    monkeypatch.setattr(
        ase_calculator_interface,
        "load_module_from_config",
        lambda config, field: FakeASECalculator,
    )

    task_func = getattr(ase_calculator_interface.ase_calculator_task, "func", ase_calculator_interface.ase_calculator_task)
    molecule = MoleculesObject(h2_atoms, "h2")
    result = task_func(
        molecule,
        {
            "QM_run_command": "fake-qm",
            "ASE_calculator": "fake.module.Calculator",
            "label": "kept",
            "fake_expected_command": "fake-qm",
            "fake_expected_properties": ["energy", "forces"],
            "fake_results": {"energy": -1.25, "forces": np.ones((len(h2_atoms), 3))},
        },
        str(tmp_path),
        {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]},
    )

    check_molecule_result(result, natoms=2, required_properties=["energy", "forces"])
    check_task_convergence(result, True)
    assert result.get_results()["energy"] == -1.25
    np.testing.assert_allclose(result.get_results()["forces"], np.ones((2, 3)))
    check_scratch_directory(tmp_path, "h2")


def test_orca_double_task_averages_runs_and_checks_thresholds(tmp_path, monkeypatch, h2_atoms):
    from alframework.qm_interfaces import orca5_interface

    responses = iter(
        [
            {"energy": -1.0, "forces": np.zeros((2, 3)), "converged": True},
            {"energy": -1.2, "forces": np.ones((2, 3)) * 0.2, "converged": True},
        ]
    )

    def fake_single_point(self, molecule, properties=None):
        return next(responses)

    monkeypatch.setattr(orca5_interface.orcaGenerator, "single_point", fake_single_point)

    task_func = getattr(orca5_interface.orca_double_calculator_task, "func", orca5_interface.orca_double_calculator_task)
    result = task_func(
        MoleculesObject(h2_atoms, "h2"),
        {
            "ncpu": 1,
            "orca_env_file": None,
            "QM_run_command": "orca",
            "orcasimpleinput": "HF",
            "orcablocks": "",
            "Ediff": 0.25,
            "Fdiff": 0.25,
        },
        str(tmp_path),
        {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]},
    )

    check_molecule_result(result, natoms=2, required_properties=["energy", "forces"])
    check_task_convergence(result, True)
    assert result.get_results()["energy"] == -1.1
    np.testing.assert_allclose(result.get_results()["forces"], np.ones((2, 3)) * 0.1)


def test_qchem_task_uses_mocked_single_point(tmp_path, monkeypatch, water_atoms):
    from alframework.qm_interfaces import qchem_DFT_interface

    expected = {"energy": -10.0, "forces": np.zeros((3, 3)), "converged": True}

    def fake_single_point(self, molecule, charge=0, mult=1, prefix="qchem", properties=None):
        assert charge == 0
        assert mult == 1
        assert prefix == "qchem"
        return expected

    monkeypatch.setattr(qchem_DFT_interface.qchemGenerator, "single_point", fake_single_point)
    task_func = getattr(qchem_DFT_interface.qchem_dft_calculator_task, "func", qchem_DFT_interface.qchem_dft_calculator_task)

    result = task_func(
        MoleculesObject(water_atoms, "water"),
        ncpu=1,
        qchem_env_file=None,
        QM_run_command="qchem",
        rem="JOBTYPE FORCE",
        qchemblocks="",
        QM_scratch_dir=str(tmp_path),
        properties_list={"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]},
    )

    check_molecule_result(result, natoms=3, required_properties=["energy", "forces"])
    check_task_convergence(result, True)
    check_energy_force_results(result.get_results(), natoms=3)
    assert result.get_results() == expected
