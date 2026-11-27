"""When a run retrains: the ``retraining`` block, counting labels or sampled structures.

A round that counts sampled structures retrains once each of them is accounted for, labeled or not: trajectories
rejected for ``distcut`` or by a topology screen, ones that never triggered, unselected cohort members and failed
sampler tasks all count, so a round doesn't wait for more sampling to make up the frames it didn't label.
"""
import json

import h5py
import pytest
from ase import Atoms

from alframework.tools.molecules import MoleculesObject
from alframework.tools.retraining import (SAMPLED_WITHOUT_LABEL, describe, ready_to_retrain, resolved,
                                          retraining_settings)
from alframework.tools.trajectory_summary import trajectory_row
from tests.helpers.driver import run_alf_driver
from tests.helpers.driver import LOOP, restart_tasks

ROUND = {key: value for key, value in LOOP.items() if key != 'save_h5_threshold'}


@pytest.mark.parametrize('master, expected', [
    ({'retraining': {'after': 2048}}, {'after': 2048, 'counts': 'labels'}),
    ({'retraining': {'after': 2048, 'counts': 'sampled'}}, {'after': 2048, 'counts': 'sampled'}),
    ({'save_h5_threshold': 2047}, {'after': 2048, 'counts': 'labels'}),
    ({'save_h5_threshold': 2.5}, {'after': 3, 'counts': 'labels'}),  # more than 2.5 labels is at least 3
])
def test_the_block_and_the_older_threshold_give_the_same_settings(master, expected):
    assert retraining_settings(master) == expected


@pytest.mark.parametrize('master, message', [
    ({'retraining': {'after': 8}, 'save_h5_threshold': 7}, 'not both'),
    ({'retraining': [8]}, 'must be a dictionary'),
    ({'retraining': {'after': 0}}, 'positive integer'),
    ({'retraining': {'after': True}}, 'positive integer'),
    ({'retraining': {'after': 8, 'counts': 'trajectories'}}, "one of \\['labels', 'sampled'\\]"),
    ({'retraining': {'after': 8, 'every': 2}}, "Unknown retraining keys \\['every'\\]"),
    ({'save_h5_threshold': '7'}, 'must be a number'),
    ({}, 'Set retraining.after'),
])
def test_mistakes_are_refused(master, message):
    with pytest.raises(ValueError, match=message):
        retraining_settings(master)


def test_the_test_modes_need_no_retraining_setting():
    assert retraining_settings({}, required=False) is None


def test_labels_count_converged_labels_and_sampled_counts_every_resolved_structure():
    labels, sampled = {'after': 4, 'counts': 'labels'}, {'after': 4, 'counts': 'sampled'}
    # Two converged labels, one unconverged, and two trajectories that ended without a frame.
    counts = dict(converged=2, finished=3, sampled_without_label=2)
    assert resolved(labels, **counts) == 2 and not ready_to_retrain(labels, **counts)
    assert resolved(sampled, **counts) == 5 and ready_to_retrain(sampled, **counts)
    assert ready_to_retrain(labels, converged=4, finished=4, sampled_without_label=0)


def test_a_round_without_a_new_label_does_not_retrain():
    sampled = {'after': 4, 'counts': 'sampled'}
    assert not ready_to_retrain(sampled, converged=0, finished=2, sampled_without_label=10)


def test_the_startup_report_names_the_unit():
    assert '2,048 new labels' in describe({'after': 2048, 'counts': 'labels'})
    assert '2,048 sampled structures' in describe({'after': 2048, 'counts': 'sampled'})


def test_the_sampler_coordinator_counts_structures_that_ended_without_a_frame():
    from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task
    from alframework.tools.sampler_coordinator import SamplerCoordinator

    coordinator = SamplerCoordinator(simple_mlmd_sampling_task, 'mlmd', {'dt': 0.5})
    frame = MoleculesObject(Atoms('H'), 'labeled')
    rejected, completed = MoleculesObject(Atoms('H'), 'rejected'), MoleculesObject(Atoms('H'), 'completed')
    rejected.update_atoms(None)
    completed.update_atoms(None)
    assert coordinator.harvest([frame, rejected, completed]) == [frame]
    # Two harvested without a frame, and three members of sampler tasks that failed.
    assert coordinator.take_without_label(3) == 5
    assert coordinator.take_without_label() == 0, 'each structure is counted once'


def test_a_finished_cohort_counts_its_members_that_were_not_selected():
    from tests.test_cohorts import cohort_config, replay, sampled

    results = sampled(12, cohort_config(12, 3))
    frames, pool = replay(results, cohort_size=12, k=3, width=4, order='in_order')
    assert 0 < len(frames) <= 3
    # With its selected frames' labels, the cohort accounts for all twelve trajectories.
    assert pool.take_unselected() + len(frames) == 12
    assert pool.take_unselected() == 0


def test_the_summary_names_distcut_as_a_rejection_reason():
    assert trajectory_row('mol-0000-0000000001', {'trajectory_outcome': 'rejected', 'distcrit': True})['reason'] == \
        'distcut'
    screened = {'trajectory_outcome': 'rejected', 'screen_reject_reason': 'topology_changed', 'distcrit': True}
    assert trajectory_row('mol-0000-0000000002', screened)['reason'] == 'topology_changed'
    assert trajectory_row('mol-0000-0000000003', {'trajectory_outcome': 'eligible', 'distcrit': False})['reason'] is None


ENDINGS = {
    'distcut': {'trajectory_outcome': 'rejected', 'distcrit': True},
    'topology': {'trajectory_outcome': 'rejected', 'screen_reject_reason': 'topology_changed'},
    'not_triggered': {'trajectory_outcome': 'completed'},
}


def rejecting_sampler(rejects, ending):
    """A sampler that ends the trajectories ``rejects(index)`` picks without a frame, the way ``ending`` says."""
    from parsl import python_app

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        index = int(molecule_object.get_moleculeid().rsplit('-', 1)[1])
        if rejects(index):
            molecule_object.update_metadata(dict(ending))
            molecule_object.update_atoms(None)
        return molecule_object

    return sampler


def shard_labels(tmp_path, number):
    with h5py.File(tmp_path / "h5" / f"data-{number:04d}.h5") as handle:
        return sum(int(group['_id'].shape[0]) for group in handle.values())


@pytest.mark.parametrize('ending', sorted(ENDINGS))
def test_a_round_of_sampled_structures_retrains_once_each_is_resolved(tmp_path, monkeypatch, capsys, ending):
    tasks = dict(restart_tasks(tmp_path, []), sampler=rejecting_sampler(lambda index: index % 2, ENDINGS[ending]))
    run_alf_driver(tmp_path, monkeypatch, **tasks, **ROUND, retraining={'after': 4, 'counts': 'sampled'})
    assert 'retraining: once 4 sampled structures are resolved' in capsys.readouterr().out
    # The first shard is the bootstrap; the second, the round. Half its structures ended without a label.
    assert 1 <= shard_labels(tmp_path, 1) < 4


def test_counting_labels_waits_for_that_many_labels(tmp_path, monkeypatch):
    tasks = dict(restart_tasks(tmp_path, []), sampler=rejecting_sampler(lambda index: index % 2, ENDINGS['distcut']))
    run_alf_driver(tmp_path, monkeypatch, **tasks, **ROUND, retraining={'after': 4})
    assert shard_labels(tmp_path, 1) >= 4


def test_rejections_alone_never_retrain_and_their_count_survives_a_restart(tmp_path, monkeypatch):
    sampled = dict(ROUND, retraining={'after': 4, 'counts': 'sampled'})
    every = rejecting_sampler(lambda index: True, ENDINGS['distcut'])
    run_alf_driver(tmp_path, monkeypatch, **dict(restart_tasks(tmp_path, []), sampler=every), **sampled,
                   stop_when=lambda status: status.get(SAMPLED_WITHOUT_LABEL, 0) >= 8)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status['current_model_id'] == 0 and not (tmp_path / "h5" / "data-0001.h5").exists(), \
        'a round of only rejections retrained on unchanged data'
    assert status[SAMPLED_WITHOUT_LABEL] >= 4

    # After a restart the round's rejections still count: the first new label or two complete it.
    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **sampled)
    assert 1 <= shard_labels(tmp_path, 1) < 4
    assert json.loads((tmp_path / "status.json").read_text())['current_model_id'] >= 1
