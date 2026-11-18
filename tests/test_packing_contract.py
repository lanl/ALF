"""Condensed-phase packing places every required solute within the atom budget, without changing its inputs."""

import pytest
from ase import Atoms

from alframework.builders import builders
from alframework.tools.molecules import MoleculesObject

LIBRARY = {"C": Atoms("C"), "O": Atoms("O"), "N": Atoms("N"), "water": Atoms("H2O")}


def pack(solutes, **options):
    """Pack into a 20 Angstrom box whose target density the first fragment already exceeds."""
    box = MoleculesObject(Atoms(cell=[20] * 3, pbc=True), "test")
    return builders.condensed_phase_builder(box, LIBRARY, solute_molecules=solutes, density=1e-6, shake=0,
                                            **options)


@pytest.mark.parametrize("names", [["C", "O"], ["C", "O", "N"]])
def test_every_required_solute_is_placed_past_the_target_density_and_an_early_stop(names):
    original = names.copy()
    result = pack(names, early_stop_probability=1)
    assert sorted(result.atoms.get_chemical_symbols()) == sorted(names)
    assert names == original, "the solute list is not consumed"


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_a_required_solute_larger_than_the_atom_budget_fails(limit):
    with pytest.raises(ValueError, match="Required solute"):
        pack(["water"], max_atoms=limit)


def test_a_solute_that_exactly_fills_the_budget_fits():
    assert len(pack(["water"], max_atoms=3)) == 3


def test_optional_solvent_stops_at_the_budget_instead_of_failing():
    assert len(pack([], solvent_molecules=["water"], max_atoms=1)) == 0


def test_a_required_solute_that_cannot_be_placed_fails_promptly():
    with pytest.raises(RuntimeError, match="required solutes"):
        pack(["C", "O"], min_dist=100, max_patience=2)


def test_the_packing_tasks_leave_their_solute_options_unchanged(monkeypatch):
    monkeypatch.setattr(builders, "readMolFiles", lambda directory: ({"C": Atoms("C"), "O": Atoms("O")}, []))
    options = [["C", "O"]]
    arguments = dict(builder_config={"shake": 0}, molecule_library_dir="unused", cell_range=[[20, 20]] * 3,
                     solute_molecule_options=options, Rrange=[1e-6, 1e-6])
    for name in ["first", "second"]:
        assert len(builders.simple_condensed_phase_builder_task.func(name, **arguments)) == 2
    assert [len(m) for m in builders.simple_multi_condensed_phase_builder_task.func(["a", "b"], **arguments)] == [2, 2]
    assert options == [["C", "O"]]
