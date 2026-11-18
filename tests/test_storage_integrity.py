"""Periodicity survives storage and replay, and a shard's formula group never mixes periodic and nonperiodic frames."""

import h5py
import numpy as np
import pytest

from alframework.builders.external_geometry import build_external_geometries
from alframework.ml_interfaces.fine_tuning import shard_atoms
from alframework.tools.label_validation import screen_labels
from alframework.tools.restarts import HeldLabels
from alframework.tools.data_shards import store_current_data
from tests.helpers.labels import PROPERTIES, labeled_h2


def replay(path):
    return build_external_geometries(["new-0"], dict(sources=[str(path)], selection_mode="sequential"))[0]


@pytest.mark.parametrize("periodic", [False, True])
def test_a_stored_frame_replays_and_trains_with_its_periodicity(tmp_path, periodic):
    path = tmp_path / "frames.h5"
    original = labeled_h2("original", periodic)
    store_current_data(str(path), [original], PROPERTIES)
    replayed = replay(path)
    np.testing.assert_array_equal(replayed.atoms.pbc, original.atoms.pbc)
    np.testing.assert_array_equal(shard_atoms([path])[0].pbc, original.atoms.pbc)
    np.testing.assert_allclose(replayed.atoms.positions, original.atoms.positions)
    if periodic:
        np.testing.assert_allclose(replayed.atoms.cell, original.atoms.cell)


@pytest.mark.parametrize("flags", [[False] * 3, [True, False, True]])
def test_explicit_pbc_flags_in_a_source_file_win_over_its_cell(tmp_path, flags):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("H")
        group["species"] = np.array(["H"], dtype="S")
        group["coordinates"] = np.zeros((1, 1, 3))
        group["cell"] = [np.eye(3) * 8]
        group["pbc"] = [flags]
    np.testing.assert_array_equal(replay(path).atoms.pbc, flags)


@pytest.mark.parametrize("first,second", [(False, True), (True, False)])
def test_a_label_that_would_mix_periodicity_in_its_formula_group_is_held_unconverged(tmp_path, first, second):
    held = HeldLabels(tmp_path / "held")
    held.hold([labeled_h2("first", first)])
    conflicting = labeled_h2("second", second)
    held.hold([conflicting])
    assert not conflicting.check_convergence()
    assert "cannot mix periodic and nonperiodic" in conflicting.metadata["labeling_error"]
    assert held.successful() == 2 and held.converged() == 1
    assert store_current_data(str(tmp_path / "next.h5"), held.iter_results(held.pieces()), PROPERTIES) == 1


def test_the_periodicity_of_held_labels_is_remembered_after_a_restart(tmp_path):
    HeldLabels(tmp_path / "held").hold([labeled_h2("before the restart", True)])
    restarted = HeldLabels(tmp_path / "held")
    conflicting = labeled_h2("after the restart", False)
    restarted.hold([conflicting])
    assert not conflicting.check_convergence() and restarted.converged() == 1


def test_after_a_store_the_next_shard_can_hold_the_other_periodicity(tmp_path):
    held = HeldLabels(tmp_path / "held")
    held.hold([labeled_h2("periodic", True)])
    store_current_data(str(tmp_path / "data-0000.h5"), held.iter_results(held.pieces()), PROPERTIES)
    held.discard(held.pieces())
    nonperiodic = labeled_h2("nonperiodic", False)
    held.hold([nonperiodic])
    assert nonperiodic.check_convergence()


def test_a_partly_periodic_label_is_rejected_before_it_is_held():
    slab = labeled_h2("slab", [True, False, True])
    screen_labels([slab], PROPERTIES)
    assert not slab.check_convergence() and "Partly periodic" in slab.metadata["labeling_error"]


@pytest.mark.parametrize("flags", [(False, True), ([True, False, True],)])
def test_storing_an_unstorable_periodicity_directly_fails_and_keeps_earlier_shards(tmp_path, flags):
    earlier = tmp_path / "earlier.h5"
    store_current_data(str(earlier), [labeled_h2("earlier", True)], PROPERTIES)
    before = earlier.read_bytes()
    path = tmp_path / "next.h5"
    with pytest.raises(ValueError, match="periodic"):
        store_current_data(str(path), [labeled_h2(str(i), flag) for i, flag in enumerate(flags)], PROPERTIES)
    assert not path.exists() and not (tmp_path / "next.h5.partial").exists()
    assert earlier.read_bytes() == before


def test_storing_a_dataset_with_a_row_missing_fails_and_writes_nothing(tmp_path, monkeypatch):
    from alframework.tools import data_shards

    original = data_shards.store_pair_index

    def drop_a_row(group, system, properties):
        original(group, system, properties)
        group["energy"] = group["energy"][:-1]

    monkeypatch.setattr(data_shards, "store_pair_index", drop_a_row)
    with pytest.raises(ValueError, match="not aligned"):
        store_current_data(str(tmp_path / "rows.h5"), [labeled_h2("a"), labeled_h2("b")], PROPERTIES)
    assert not (tmp_path / "rows.h5").exists()
