"""Builders return one MoleculesObject or a list of them, checked the same way wherever the coordinator harvests."""

import pytest
from ase import Atoms
from parsl import python_app

from alframework.tools.builder_outputs import builder_outputs
from alframework.tools.molecules_class import MoleculesObject
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
