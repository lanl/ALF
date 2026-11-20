"""The sizing block: what each task holds and how many labels a round adds, with the settings that follow."""
import pytest

from alframework.tools.sizing import apply_sizing, derive_sizing, describe, sizing_block

BLOCK = {'trajectories_per_sampler_task': 256, 'structures_per_qm_task': 128, 'labels_per_round': 16384}


def test_the_block_fills_in_the_widths_and_queue_limits():
    derived = derive_sizing(BLOCK, sampler_slots=64, builder_batches=True)
    master, sampler, qm = {}, {'batching': {'partial_policy': 'timeout', 'max_wait_seconds': 30}}, {}
    apply_sizing(derived, master, sampler, qm)
    assert master == {'parallel_samplers': 64 * 256, 'maximum_builder_structures': 256, 'target_queued_QM': 16384,
                      'save_h5_threshold': 16383}
    assert sampler == {'batching': {'partial_policy': 'timeout', 'max_wait_seconds': 30, 'batch_size': 256}}
    assert qm == {'batching': {'batch_size': 128}}
    assert describe(BLOCK, derived) == (
        'Sizing: sampler batching.batch_size 256, maximum_builder_structures 256, parallel_samplers 16,384 '
        '(64 sampler tasks at once), QM batching.batch_size 128, target_queued_QM 16,384, save_h5_threshold 16,383 '
        '(retrain once 16,384 new labels are held)')


def test_each_key_fills_in_only_its_own_settings():
    master, sampler, qm = {}, {}, {}
    apply_sizing(derive_sizing({'labels_per_round': 100}), master, sampler, qm)
    assert master == {'target_queued_QM': 100, 'save_h5_threshold': 99} and sampler == {} and qm == {}
    assert derive_sizing(None) == {'master': {}, 'sampler': {}, 'QM': {}}


def test_a_builder_without_moleculeids_keeps_one_structure_per_task():
    derived = derive_sizing({'trajectories_per_sampler_task': 32}, sampler_slots=4, builder_batches=False)
    assert derived['master'] == {'parallel_samplers': 128}


def test_the_sampler_slots_can_be_given_and_must_be_known():
    assert derive_sizing({'trajectories_per_sampler_task': 32, 'sampler_tasks_at_once': 2},
                         sampler_slots=64)['master']['parallel_samplers'] == 64
    with pytest.raises(ValueError, match='set sizing.sampler_tasks_at_once'):
        derive_sizing({'trajectories_per_sampler_task': 32}, sampler_slots=None)
    with pytest.raises(ValueError, match='needs trajectories_per_sampler_task too'):
        sizing_block({'sizing': {'sampler_tasks_at_once': 2}})


def test_a_sampler_that_takes_one_structure_cannot_be_given_a_batch():
    with pytest.raises(ValueError, match='needs a sampler task that takes molecule_objects'):
        derive_sizing({'trajectories_per_sampler_task': 32}, sampler_slots=4, sampler_batches=False)


def test_settings_given_in_their_own_place_must_agree():
    derived = derive_sizing(BLOCK, sampler_slots=64, builder_batches=True)
    apply_sizing(derived, {'target_queued_QM': 16384}, {'batching': {'batch_size': 256}}, {})
    with pytest.raises(ValueError, match='target_queued_QM is 100 in the master configuration but sizing makes it '
                                         '16384'):
        apply_sizing(derived, {'target_queued_QM': 100}, {}, {})
    with pytest.raises(ValueError, match='batching.batch_size is 32 in the sampler configuration but sizing'):
        apply_sizing(derived, {}, {'batching': {'batch_size': 32}}, {})


@pytest.mark.parametrize('block, message', [
    ([], 'must be a dictionary'), ({'labels': 1}, "unknown settings \\['labels'\\]"),
    ({'labels_per_round': 0}, 'positive integer'), ({'labels_per_round': True}, 'positive integer')])
def test_mistakes_in_the_block_are_refused(block, message):
    with pytest.raises(ValueError, match=message):
        sizing_block({'sizing': block})


def test_a_reload_gets_the_derived_values_again():
    derived = derive_sizing({'labels_per_round': 8, 'structures_per_qm_task': 4})
    for _ in range(2):  # the coordinator's startup, then a reload of unedited files
        master, qm = {}, {'batching': {'partial_policy': 'timeout'}}
        apply_sizing(derived, master, {}, qm)
        assert master['save_h5_threshold'] == 7 and qm['batching'] == {'partial_policy': 'timeout', 'batch_size': 4}


def test_the_coordinator_runs_sized_by_the_block_as_by_hand(tmp_path, monkeypatch, capsys):
    """The same run, sized by the block or with each setting written out, labels the same bootstrap."""
    import h5py
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    loop = {key: value for key, value in LOOP.items() if key not in ('target_queued_QM', 'save_h5_threshold')}
    first = {}
    for arm in ('by_hand', 'sized'):
        run_dir = tmp_path / arm
        run_dir.mkdir()
        if arm == 'by_hand':
            settings = dict(loop, target_queued_QM=4, save_h5_threshold=3)
            qm = {'batching': {'mode': 'sequential', 'batch_size': 2}}
        else:
            settings = dict(loop, sizing={'labels_per_round': 4, 'structures_per_qm_task': 2})
            qm = {}
        with monkeypatch.context() as patch:
            run_alf_driver(run_dir, patch, **restart_tasks(run_dir, []), **settings, finite_bootstrap=True,
                           qm_config=qm)
        with h5py.File(run_dir / 'h5' / 'data-0000.h5') as handle:
            first[arm] = sorted(value.decode() for group in handle.values() for value in group['_id'][()])
    out = capsys.readouterr().out
    assert 'Sizing: QM batching.batch_size 2, target_queued_QM 4, save_h5_threshold 3 (retrain once 4 new labels' in out
    assert first['sized'] == first['by_hand'] and len(first['sized']) == LOOP['bootstrap_set']
