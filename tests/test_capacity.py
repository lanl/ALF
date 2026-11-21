"""The startup capacity report: what a run keeps in flight against what its Parsl executors can run."""
from types import SimpleNamespace

import pytest
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import AprunLauncher, MpiExecLauncher, SimpleLauncher, SrunLauncher
from parsl.providers import SlurmProvider

from alframework.tools.capacity import capacity_report, executor_slots, pools_per_node, task_slots


def _htex(label, workers, nodes, blocks):
    return HighThroughputExecutor(label=label, max_workers_per_node=workers, available_accelerators=workers,
                                  provider=SlurmProvider(nodes_per_block=nodes, max_blocks=blocks))


def test_slots_come_from_workers_nodes_and_blocks():
    assert executor_slots(_htex('a', 4, 3, 2)) == 24
    assert executor_slots(ThreadPoolExecutor(label='t', max_threads=3)) == 3
    assert executor_slots(SimpleNamespace(label='x')) is None
    executors = {'a': _htex('a', 4, 3, 2), 'b': _htex('b', 4, 1, 1), 'x': SimpleNamespace()}
    assert task_slots(['a', 'b'], executors) == 28
    assert task_slots(['a', 'x'], executors) is None, 'one unknown executor makes the total unknown'
    assert task_slots(['missing'], executors) is None


def _pooled(launcher, workers=1, nodes=128):
    return HighThroughputExecutor(label='p', max_workers_per_node=workers, available_accelerators=workers,
                                  provider=SlurmProvider(nodes_per_block=nodes, max_blocks=1, launcher=launcher))


@pytest.mark.parametrize('launcher, pools', [
    # One pool per GPU, as a per-GPU layout starts them: Parsl counts one per node.
    (SrunLauncher(overrides='--nodes=128 --ntasks 1024 --ntasks-per-node=8 -c 7 --gpus-per-task=1'), 8),
    (SrunLauncher(overrides='--ntasks-per-node 4'), 4),
    (SrunLauncher(overrides='--ntasks 1 --ntasks 256'), 2),  # srun keeps the last --ntasks; 256 over 128 nodes
    (SrunLauncher(overrides='-n 100'), 1),  # doesn't divide over the nodes, so not counted
    (SrunLauncher(overrides='--ntasks-per-node=1 -c 56'), 1),
    (SrunLauncher(), 1),
    (MpiExecLauncher(overrides='-ppn 8 --depth 7'), 8),
    (AprunLauncher(overrides='-N 6'), 6),
    (SimpleLauncher(), 1),
])
def test_a_launcher_that_starts_several_pools_per_node_is_counted(launcher, pools):
    assert pools_per_node(_pooled(launcher).provider) == pools
    assert executor_slots(_pooled(launcher)) == 128 * pools


def test_another_launcher_can_say_how_many_pools_it_starts():
    launcher = SimpleLauncher()
    launcher.pools_per_node = 8
    assert executor_slots(_pooled(launcher)) == 1024
    launcher.pools_per_node = True
    assert executor_slots(_pooled(launcher)) == 128, 'only a positive integer counts'


def _report(master, **overrides):
    executors = {'alf_sampler_executor': _htex('alf_sampler_executor', 4, 4, 1),
                 'alf_QM_executor': _htex('alf_QM_executor', 4, 2, 1),
                 'alf_ML_executor': _htex('alf_ML_executor', 1, 4, 1)}
    arguments = dict(builder_labels=['alf_sampler_executor'], builder_width=256, builder_batches=True,
                     sampler_labels=['alf_sampler_executor'], sampler_width=256,
                     qm_labels=['alf_QM_executor'], qm_width=256, ml_labels=['alf_ML_executor'], n_models=4)
    arguments.update(overrides)
    return capacity_report(master, executors, **arguments)


def test_a_run_sized_to_its_executors_gets_no_warnings():
    lines, warnings = _report(dict(parallel_samplers=4096, target_queued_QM=2048, save_h5_threshold=4095))
    assert warnings == []
    assert lines == ['sampling: 16 tasks at once of up to 256 structures = 4,096; parallel_samplers 4,096',
                     'builders: 256 structures per task, 16 tasks to fill sampling',
                     'QM: 8 tasks at once of up to 256 structures = 2,048; target_queued_QM 2,048',
                     'retraining: once more than 4,095 new labels are held',
                     'ML: 4 ensemble members, 4 trained at once']


@pytest.mark.parametrize('master, overrides, expected', [
    (dict(parallel_samplers=1024), {}, 'fills 4 of 16 sampler task slots, so 12 stay idle'),
    (dict(parallel_samplers=4100), {}, 'not a multiple of the sampler batch size 256'),
    (dict(parallel_samplers=4096), dict(builder_width=100), 'fill sampler batches of 256 unevenly'),
    (dict(parallel_samplers=4096, target_queued_QM=512), {}, 'fills 2 of 8 QM task slots, so 6 stay idle'),
    (dict(parallel_samplers=4096), dict(n_models=8), '8 ensemble members but 4 ML task slots'),
    (dict(parallel_samplers=4096), dict(n_models=4, member_nodes=2), '4 ensemble members but 2 ML task slots'),
], ids=['idle_samplers', 'partial_batches', 'builder_width', 'idle_qm', 'members_wait', 'multinode_members'])
def test_each_mismatch_is_named(master, overrides, expected):
    _, warnings = _report(master, **overrides)
    assert any(expected in warning for warning in warnings), warnings


def test_unknown_executors_and_unbatched_stages_are_described_without_guessing():
    lines, warnings = capacity_report(
        dict(parallel_samplers=10), {'s': SimpleNamespace()}, builder_labels=['s'], builder_width=1,
        builder_batches=False, sampler_labels=['s'], sampler_width=1, qm_labels=['s'], qm_width=1)
    assert lines[0] == 'sampling: an unknown number of tasks at once of up to 1 structures; parallel_samplers 10'
    assert lines[1] == 'builders: one structure per task'
    assert warnings == []


def test_the_coordinator_prints_the_report_at_startup(tmp_path, monkeypatch, capsys):
    from tests.helpers.driver import run_alf_driver
    from tests.test_restarts import LOOP, restart_tasks

    run_alf_driver(tmp_path, monkeypatch, **restart_tasks(tmp_path, []), **LOOP)
    out = capsys.readouterr().out
    assert 'Capacity:\n  sampling: 2 tasks at once of up to 1 structures = 2; parallel_samplers 2' in out
    assert '  QM: 2 tasks at once of up to 1 structures = 2; target_queued_QM 2' in out
