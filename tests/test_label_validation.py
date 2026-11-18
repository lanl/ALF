"""Invalid QM labels are marked unconverged before they're counted, and can't be stored."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from parsl import python_app

from alframework.tools.label_validation import screen_labels
from alframework.tools.restarts import HeldLabels
from alframework.tools.data_shards import store_current_data
from alframework.tools.label_validation import system_checker
from tests.helpers.driver import MODEL_PATTERN, run_alf_driver
from tests.helpers.labels import PROPERTIES, labeled_h2


def corrupt(molecule, properties, damage):
    """Damage one part of a label; ``unit`` makes the energy overflow only once its multiplier is applied."""
    results = molecule.qm_results
    if damage in ("nan energy", "inf energy"):
        results["energy"] = np.nan if damage.startswith("nan") else np.inf
    elif damage in ("nan force", "inf force"):
        results["forces"][0, 0] = np.nan if damage.startswith("nan") else np.inf
    elif damage == "positions":
        molecule.atoms.positions[0, 0] = np.inf
    elif damage == "cell":
        molecule.atoms.cell[0, 0] = np.nan
    elif damage == "missing forces":
        del results["forces"]
    elif damage == "forces for one atom":
        results["forces"] = np.zeros((1, 3))
    elif damage == "two force components":
        results["forces"] = np.zeros((2, 2))
    elif damage == "energy array":
        results["energy"] = np.zeros(2)
    elif damage == "unit":
        results["energy"] = 1e308
        properties["energy"] = ["energy", "system", 1e308]


DAMAGE = ["nan energy", "inf energy", "nan force", "inf force", "positions", "cell", "missing forces",
          "forces for one atom", "two force components", "energy array", "unit"]


@pytest.mark.parametrize("damage", DAMAGE)
def test_an_invalid_label_is_held_unconverged_and_its_sibling_is_stored(tmp_path, damage):
    good, bad = labeled_h2("good"), labeled_h2("bad")
    properties = dict(PROPERTIES)
    corrupt(bad, properties, damage)
    screen_labels([good, bad], properties)
    assert good.check_convergence() and not bad.check_convergence()
    assert bad.metadata["labeling_error"].startswith("Invalid label:")
    held = HeldLabels(tmp_path / "held")
    held.hold([good, bad])
    assert held.successful() == 2 and held.converged() == 1
    assert store_current_data(str(tmp_path / "valid.h5"), held.iter_results(held.pieces()), properties) == 1


@pytest.mark.parametrize("damage", DAMAGE)
def test_storing_an_invalid_label_directly_fails_and_writes_nothing(tmp_path, damage):
    bad = labeled_h2("bad")
    properties = dict(PROPERTIES)
    corrupt(bad, properties, damage)
    with pytest.raises(ValueError):
        store_current_data(str(tmp_path / "invalid.h5"), [bad], properties)
    assert not (tmp_path / "invalid.h5").exists()


def test_numpy_scalars_and_array_valued_system_properties_are_valid():
    molecule = labeled_h2("valid")
    molecule.qm_results.update(energy=np.float64(-1), dipole=np.ones(3), quadrupole=np.eye(3))
    properties = dict(PROPERTIES, dipole=["dipole", "system", 1], quadrupole=["quadrupole", "system", 1])
    screen_labels([molecule], properties)
    assert molecule.check_convergence() and system_checker(molecule)


def test_an_unconverged_result_is_kept_without_checking_it():
    molecule = labeled_h2("failed", converged=False)
    molecule.update_atoms(None)
    molecule.qm_results.clear()
    assert screen_labels([molecule], PROPERTIES) == [molecule]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.array([np.nan])])
def test_the_builder_check_rejects_nonfinite_results(value):
    molecule = labeled_h2("bad")
    molecule.qm_results["energy"] = value
    assert not system_checker(molecule, kill_on_fail=False, print_error=False)


def test_a_finite_bootstrap_replaces_an_invalid_label_and_trains_only_on_valid_data(tmp_path, monkeypatch):
    trained = []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return labeled_h2(moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        if molecule_object.get_moleculeid().endswith("0000000000"):
            molecule_object.qm_results["energy"] = np.nan
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id, h5_dir):
        for shard in Path(h5_dir).glob("*.h5"):
            with h5py.File(shard) as handle:
                assert all(np.isfinite(group["energy"][()]).all() for group in handle.values())
        trained.append(current_training_id)
        (tmp_path / MODEL_PATTERN.format(current_training_id)).mkdir(parents=True)
        return [True], current_training_id

    run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=sampler, qm=qm, ml=ml,
                   bootstrap_set=2, finite_bootstrap=True, bootstrap_spares=1, parallel_samplers=2,
                   target_queued_QM=2, save_h5_threshold=0, maximum_stored_labels=2, poll_interval=0.01)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["lifetime_stored_labels"] == 2 and trained == [0]
