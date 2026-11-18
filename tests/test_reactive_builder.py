"""The reactive builder's rattle setting: a boolean, or the strings true and false."""

import pytest
from ase import Atoms
from parsl.app.errors import RemoteExceptionWrapper

from alframework.builders import reactive_builder as reactive

LIBRARY = dict(N_reactions=1, molecule_library_dir="unused", r_files="r", ts_files="ts", p_files="p")


@pytest.mark.parametrize("flag,rattled", [(False, False), ("false", False), (True, True), (" True ", True)])
def test_rattle_follows_the_flag_and_the_reaction_metadata_is_kept(monkeypatch, flag, rattled):
    monkeypatch.setattr(reactive.ase.io, "read", lambda *args, **kwargs: Atoms("H"))
    calls = []
    monkeypatch.setattr(Atoms, "rattle", lambda self: calls.append(self))
    result = reactive.load_reactive_task.func("reaction-0", dict(LIBRARY), rattle=flag)
    if isinstance(result, RemoteExceptionWrapper):
        result.reraise()
    assert len(calls) == (3 if rattled else 0), "reactant, transition state and product"
    assert result.get_moleculeid() == "reaction-0"
    assert {"reactant", "ts", "product", "index_mol"} <= set(result.metadata)


@pytest.mark.parametrize("flag", [None, 1, "sometimes", ""])
def test_any_other_rattle_value_fails_before_reading_files(flag):
    result = reactive.load_reactive_task.func("reaction-0", {}, rattle=flag)
    with pytest.raises(ValueError, match="rattle"):
        result.reraise()
