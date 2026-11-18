"""Exercise the legacy RDKit task through the actual MoleculesObject packer."""
from functools import wraps

import pytest
from ase import Atoms

pytest.importorskip('rdkit')
from alframework.builders import rdkit_condensed_phase as rdkit_builder
from alframework.tools.molecules_class import MoleculesObject


@pytest.mark.parametrize('solvents', [['O'], {'O': 0.25, 'C': 0.75}])
def test_task_preserves_solvent_specification_and_returns_object(monkeypatch, solvents):
    real_packer = rdkit_builder.condensed_phase_builder
    observed = {}

    @wraps(real_packer)
    def pack(**kwargs):
        observed.update(kwargs)
        return real_packer(**kwargs)

    monkeypatch.setattr(rdkit_builder, 'condensed_phase_builder', pack)
    monkeypatch.setattr(rdkit_builder, 'ASEfromSMILES', lambda smiles: Atoms('H2O',
        positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0]]))
    options = [['O']]
    # No duplicate solvent entry in builder_config: the explicit task argument must survive.
    molecule = rdkit_builder.rdkit_condensed_phase_builder_task.func(
        'rdkit-0', {'min_dist': 0.5, 'max_patience': 10}, [[6, 6]]*3,
        [0.1, 0.1], options, solvents)
    assert isinstance(molecule, MoleculesObject)
    assert molecule.get_moleculeid() == 'rdkit-0'
    assert molecule.get_atoms().pbc.all()
    assert len(molecule) == 3
    assert molecule.get_metadata()['actual_density'] >= 0.1
    assert observed['solvent_molecules'] == solvents
    assert set(observed['molecule_library']) == set(['O'] + list(solvents))
    assert options == [['O']]


def test_task_can_fill_a_solvent_only_box(monkeypatch):
    monkeypatch.setattr(rdkit_builder, 'ASEfromSMILES', lambda smiles: Atoms('H2O',
        positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0]]))
    molecule = rdkit_builder.rdkit_condensed_phase_builder_task.func(
        'solvent-only', {'min_dist': 0.5}, [[6, 6]]*3, [0.1, 0.1], [[]], {'O': 1.0})
    assert isinstance(molecule, MoleculesObject)
    assert molecule.get_atoms().get_chemical_formula() == 'H2O'
    assert molecule.get_metadata()['actual_density'] >= 0.1


def test_task_builds_real_rdkit_geometry_and_can_reuse_solutes():
    options = [['O']]
    for name in ['first', 'second']:
        molecule = rdkit_builder.rdkit_condensed_phase_builder_task.func(
            name, {'min_dist': 0.5, 'max_patience': 10}, [[6, 6]]*3,
            [0.1, 0.1], options, {'O': 1.0})
        assert molecule.get_moleculeid() == name
        assert molecule.get_atoms().get_chemical_formula() == 'H2O'
        assert molecule.get_atoms().pbc.all()
    assert options == [['O']]
