"""The ASE VASP task, with a stand-in for ASE's VASP calculator."""

from pathlib import Path

import numpy as np
import pytest

from alframework.tools.molecules import MoleculesObject


@pytest.mark.parametrize("matches", [0, 1, 2])
def test_vasp_labels_converge_only_on_one_ediff_message_and_use_the_molecule_s_kpoints(
        tmp_path, monkeypatch, h2_atoms, matches):
    import ase
    import ase.calculators.vasp

    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task

    calls = []

    class ControlledVasp:
        def __init__(self, directory, command, **options):
            self.directory = Path(directory)
            calls.append((command, options))

        def calculate(self, atoms, properties):
            assert properties == ["energy", "forces"]
            self.directory.mkdir(parents=True)
            (self.directory / "OUTCAR").write_text("aborting loop because EDIFF is reached\n" * matches)
            self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}

    monkeypatch.setattr(ase.calculators.vasp, "Vasp", ControlledVasp)
    molecule = MoleculesObject(h2_atoms, "h2")
    molecule.update_metadata({"kpoints": [2, 2, 2]})
    config = {"QM_run_command": "controlled-vasp", "input": {"encut": 400, "kpts": [1, 1, 1]}}
    result = VASP_ase_calculator_task.func(molecule, config, str(tmp_path), {"energy": [], "forces": []})
    assert result is molecule and result.check_convergence() is (matches == 1)
    assert calls == [("controlled-vasp", {"encut": 400, "kpts": [2, 2, 2]})]
    assert config["input"]["kpts"] == [1, 1, 1], "the configuration is not changed"
    if matches == 1:
        # The source is the configuration, the same for every label, not the molecule's own k-points.
        assert result.get_metadata()["label_source"] == {
            "labeler": "VASP_ase_calculator_task", "calculator": "ase.calculators.vasp.Vasp",
            "ase_version": ase.__version__, "input": {"encut": 400, "kpts": [1, 1, 1]}}
    else:
        assert "label_source" not in result.get_metadata()
