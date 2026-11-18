"""Defect counts and species stay correct when substitutions and vacancies are combined."""
import random

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

from alframework.builders import defect_crystal_builder as defect
from alframework.tools.molecules import MoleculesObject


@pytest.fixture
def structure(monkeypatch):
    atoms = Atoms('Na3Cl', positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
                  cell=[10]*3, pbc=True)
    atoms.set_masses([25, 26, 27, 37])
    monkeypatch.setattr(defect.ase.io, 'read', lambda *args, **kwargs: atoms.copy())
    return atoms


@pytest.mark.parametrize('seed', range(10))
def test_multiple_vacancies_remove_only_host_atoms(structure, seed):
    random.seed(seed)
    result = defect.builder('unused.cfg', nVac=2, tMain='Na')
    assert sorted(result.get_chemical_symbols()) == ['Cl', 'Na']
    assert result.pbc.all()
    np.testing.assert_array_equal(result.cell, structure.cell)
    chlorine = list(result.symbols).index('Cl')
    np.testing.assert_array_equal(result.positions[chlorine], structure.positions[3])
    assert result.get_masses()[chlorine] == 37


@pytest.mark.parametrize('seed', range(10))
def test_replacements_use_distinct_sites_and_vacancies_use_remaining_hosts(structure, seed):
    random.seed(seed)
    result = defect.builder('unused.cfg', tRep=['K', 'Li'], nVac=1, tMain='Na')
    assert sorted(result.get_chemical_symbols()) == ['Cl', 'K', 'Li']
    for atom in result:
        expected = 37 if atom.symbol == 'Cl' else atomic_masses[atomic_numbers[atom.symbol]]
        assert atom.mass == expected


@pytest.mark.parametrize('kwargs', [
    {'nVac': 4}, {'tRep': ['K']*4}, {'tRep': ['K', 'Li'], 'nVac': 2},
    {'nVac': 1, 'tMain': 'F'},
])
def test_impossible_defect_counts_raise_instead_of_retrying_forever(structure, kwargs):
    with pytest.raises(ValueError, match='only .* available'):
        defect.builder('unused.cfg', **kwargs)


def test_all_hosts_can_be_removed_without_losing_other_species(structure):
    result = defect.builder('unused.cfg', nVac=3)
    assert result.get_chemical_symbols() == ['Cl']


def test_no_defects_preserve_geometry_and_masses(structure):
    result = defect.builder('unused.cfg')
    assert result == structure
    np.testing.assert_array_equal(result.get_masses(), structure.get_masses())


def test_task_wraps_multiple_vacancy_result_with_metadata(tmp_path, monkeypatch):
    path = tmp_path/'host.cfg'
    write(path, Atoms('Na3Cl', positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
                      cell=[10]*3, pbc=True))
    # Choose maxVac while retaining the task's real CFG loader and object wrapper.
    monkeypatch.setattr(defect.r, 'choice', lambda values: values[-1])
    result = defect.defect_cell_builder_task.func(
        'defect-0', str(tmp_path)+'/', maxVac=2, cScaleRange=[1, 1], maxAt=0)
    assert isinstance(result, MoleculesObject)
    assert result.get_moleculeid() == 'defect-0'
    assert sorted(result.get_atoms().get_chemical_symbols()) == ['Cl', 'Na']
    assert result.get_metadata()['nVac'] == 2
    assert result.get_metadata()['StructFile'] == str(path)


def test_cell_rattle_keeps_fractional_coordinates_in_a_skewed_cell(monkeypatch):
    atoms = Atoms('Na2', scaled_positions=[[.1, .2, .3], [.6, .7, .8]],
                  cell=[[4, .5, .1], [0, 5, .3], [0, 0, 6]], pbc=True)
    monkeypatch.setattr(defect.ase.io, 'read', lambda *args, **kwargs: atoms.copy())
    monkeypatch.setattr(defect.np.random, 'normal', lambda **kwargs: .1)
    result = defect.builder('unused.cfg', CRat=.1)
    np.testing.assert_allclose(result.get_scaled_positions(), atoms.get_scaled_positions())
    assert not np.allclose(result.cell, atoms.cell)
    off_diagonal = np.triu_indices(3, 1)
    np.testing.assert_array_equal(result.cell.array[off_diagonal], atoms.cell.array[off_diagonal])
