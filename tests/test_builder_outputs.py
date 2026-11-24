"""Builders return one MoleculesObject or a list of them, checked the same way wherever the coordinator harvests."""

import pytest
from ase import Atoms
from parsl import python_app

from alframework.tools.builder_outputs import builder_outputs
from alframework.tools.molecules import MoleculesObject
from tests.helpers.driver import MODEL_PATTERN, run_alf_driver


def test_one_structure_or_a_list_comes_back_as_the_same_objects():
    molecule = MoleculesObject(Atoms("H"), "id")
    assert builder_outputs(molecule, "builder") == [molecule]
    assert builder_outputs([molecule], "builder")[0] is molecule
    assert builder_outputs([], "builder") == []


def coordinator_builders(returned):
    """Apps for a coordinator run whose builder returns ``returned(moleculeid)``; nothing after it may run."""

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return returned(moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        raise AssertionError("must not reach sampling")

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        raise AssertionError("must not reach labeling")

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        return [True], current_training_id

    return dict(builder=builder, sampler=sampler, qm=qm, ml=ml)


UNSUPPORTED = {
    "tuple": lambda moleculeid: (MoleculesObject(Atoms("H"), moleculeid),),
    "legacy record": lambda moleculeid: [{"moleculeid": moleculeid}, Atoms("H"), {}],
    "atoms": lambda moleculeid: Atoms("H"),
    "none": lambda moleculeid: None,
}


@pytest.mark.parametrize("phase", ["bootstrap", "active learning", "--test_builder"])
@pytest.mark.parametrize("returned", UNSUPPORTED, ids=str)
def test_an_unsupported_builder_return_stops_the_coordinator_naming_the_builder(tmp_path, monkeypatch,
                                                                                phase, returned):
    if phase == "active learning":
        (tmp_path / MODEL_PATTERN.format(0)).mkdir(parents=True)
    with pytest.raises(TypeError, match="Builder mock.builder.*MoleculesObject"):
        run_alf_driver(tmp_path, monkeypatch, **coordinator_builders(UNSUPPORTED[returned]),
                       bootstrap_set=1, parallel_samplers=1, target_queued_QM=1, save_h5_threshold=0,
                       poll_interval=0.01, arguments=["--test_builder"] if phase == "--test_builder" else [])


def test_test_builder_refuses_a_builder_that_returns_no_structures(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="returned no structures for --test_builder"):
        run_alf_driver(tmp_path, monkeypatch, **coordinator_builders(lambda moleculeid: []),
                       bootstrap_set=1, parallel_samplers=1, target_queued_QM=1, save_h5_threshold=0,
                       poll_interval=0.01, arguments=["--test_builder"])


def test_a_builder_gets_one_id_unless_it_takes_a_list():
    from alframework.tools.builder_outputs import builder_width, builds_batches

    def one(moleculeid, builder_config):
        pass

    def many(moleculeids, builder_config):
        pass

    assert not builds_batches(one) and builds_batches(python_app(executors=["x"])(many))
    assert builder_width(one, {"maximum_builder_structures": 8}) == 1
    assert builder_width(many, {"maximum_builder_structures": 8}) == 8
    assert builder_width(many, {}) == 1
    for bad in (0, True, 2.0, "4"):
        with pytest.raises(ValueError, match="maximum_builder_structures must be a positive integer"):
            builder_width(one, {"maximum_builder_structures": bad})


def test_a_single_id_builder_gets_every_id_once_whatever_maximum_builder_structures_says(tmp_path, monkeypatch):
    """The bootstrap reserves one ID per task for a builder that takes one, so none is skipped or overcounted."""
    import numpy as np
    from tests.helpers.driver import stored_ids
    from tests.helpers.molecules import SPRING_WATER

    calls, training = [], []

    @python_app(executors=["finite_executor"])
    def builder(moleculeid):
        calls.append(moleculeid)
        return MoleculesObject(SPRING_WATER.copy(), moleculeid)

    @python_app(executors=["finite_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["finite_executor"])
    def label(molecule_objects):
        for molecule in molecule_objects:
            molecule.store_results(dict(energy=-1.0, forces=np.zeros((3, 3))))
            molecule.set_converged_flag(True)
        return molecule_objects

    @python_app(executors=["finite_executor"])
    def train(current_training_id):
        training.extend(stored_ids(tmp_path / "h5"))
        return [True] * 4, current_training_id

    run_alf_driver(tmp_path, monkeypatch, builder=builder, sampler=sampler, qm=label, ml=train,
                   executors=("finite_executor",),
                   qm_config={"batching": {"mode": "native", "batch_size": 4, "partial_policy": "timeout",
                                           "max_wait_seconds": 0.001}},
                   stop_when=lambda status: status["current_model_id"] >= 0, bootstrap_set=5, finite_bootstrap=True,
                   maximum_builder_structures=4, parallel_samplers=8, target_queued_QM=8, save_h5_threshold=4,
                   gpus_per_node=0)
    assert sorted(name for name in calls if name.startswith("mol-boot-")) == [f"mol-boot-{i:010d}" for i in range(5)]
    assert len(training) == 5
    learning = sorted(int(name.rsplit("-", 1)[1]) for name in calls if not name.startswith("mol-boot-"))
    assert learning and learning == list(range(learning[0], learning[0] + len(learning)))
