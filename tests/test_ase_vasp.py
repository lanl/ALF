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


def test_slab_mesh_uses_reciprocal_lengths_and_keeps_one_point_through_vacuum(h2_atoms):
    from alframework.qm_interfaces.ase.vasp import _spacing_kpoints

    h2_atoms.pbc = True
    h2_atoms.set_cell([10, 10, 12, 90, 90, 60])
    assert _spacing_kpoints(h2_atoms, 'kpoint_spacing_A_inv', 0.18, ['c']) == [5, 5, 1]
    assert _spacing_kpoints(h2_atoms, 'kpoint_spacing_A_inv', 0.18, ['a', 'b']) == [1, 1, 3]
    h2_atoms.cell[2, 2] = 35
    assert _spacing_kpoints(h2_atoms, 'kpoint_spacing_A_inv', 0.18, ['c']) == [5, 5, 1]
    h2_atoms.cell[0] *= 2
    assert _spacing_kpoints(h2_atoms, 'kpoint_spacing_A_inv', 0.18, ['c']) == [3, 5, 1]


@pytest.mark.parametrize('setting', ['kpoint_spacing_A_inv', 'slab_kspacing_A_inv'])
@pytest.mark.parametrize('spacing, options', [
    (0, {}), (True, {}), (float('inf'), {}),
    (0.18, {'kpts': [4, 4, 1]}), (0.18, {'kspacing': 0.18}), (0.18, {'gamma': False}),
])
def test_spacing_rejects_invalid_or_conflicting_settings_at_startup(setting, spacing, options):
    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task
    from alframework.tools.capabilities import validate_task_config

    with pytest.raises(ValueError, match=setting):
        validate_task_config(VASP_ase_calculator_task, {setting: spacing, 'input': options}, {})


def test_vasp_spacing_options_are_mutually_exclusive():
    from alframework.qm_interfaces.ase.vasp import validate_vasp_config

    with pytest.raises(ValueError, match='kpoint_spacing_A_inv cannot be combined with slab_kspacing_A_inv'):
        validate_vasp_config({'kpoint_spacing_A_inv': 0.18, 'slab_kspacing_A_inv': 0.18, 'input': {}})


@pytest.mark.parametrize('enabled, options', [
    ('true', {}), (True, {'idipol': 3}), (True, {'ldipol': False}),
    (True, {'dipol': [0.5, 0.5, 0.5]}), (True, {'nelect': 50}), (True, {'charge': -1}),
])
def test_automatic_slab_dipole_rejects_conflicting_settings(enabled, options):
    from alframework.qm_interfaces.ase.vasp import validate_vasp_config

    with pytest.raises(ValueError, match='slab_dipole_correction'):
        validate_vasp_config({'slab_dipole_correction': enabled, 'input': options})


@pytest.mark.parametrize('cell, pbc, metadata, match', [
    ([4, 4, 0], True, {}, 'full periodic cell'),
    ([4, 4, 4], [True, True, False], {}, 'full periodic cell'),
    ([4, 4, 4], True, {'vacuum_axes': 'c'}, 'vacuum_axes'),
    ([4, 4, 4], True, {'vacuum_axes': ['z']}, 'vacuum_axes'),
    ([4, 4, 4], True, {'vacuum_axes': ['c', 'c']}, 'vacuum_axes'),
    ([4, 4, 4], True, {'kpoints': [1, 1, 1]}, 'metadata kpoints'),
])
def test_adaptive_spacing_rejects_invalid_structure_inputs(h2_atoms, tmp_path, cell, pbc, metadata, match):
    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task

    h2_atoms.set_cell(cell)
    h2_atoms.pbc = pbc
    molecule = MoleculesObject(h2_atoms, 'h2')
    molecule.update_metadata(metadata)
    config = {'QM_run_command': 'controlled-vasp', 'kpoint_spacing_A_inv': 0.18, 'input': {}}
    with pytest.raises(ValueError, match=match):
        result = VASP_ase_calculator_task.func(molecule, config, str(tmp_path), {})
        result.reraise()
