"""Functional checks for the opt-in campaign's audit and acceptance rules."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

CAMPAIGN = Path(__file__).resolve().parents[1] / 'examples/simple_water_campaign'
# The example is intentionally importable by worker PYTHONPATH, not installed as a package.
sys.path.insert(0, str(CAMPAIGN))
import campaign_support as support
import campaign_tasks as tasks
spec = importlib.util.spec_from_file_location('water_campaign_runner', CAMPAIGN / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def profile():
    return support.read_json(CAMPAIGN / 'profile.json')


def write_h5(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('H_2_O_1')
        group['_id'] = np.array(ids, dtype='S')


def members():
    return [dict(member=f'model-{i:02d}', complete=True, epochs=2,
                 counts=dict(retained=40, train=33, valid=4, test=3)) for i in range(4)]


def accepted_case(tmp_path, profile, qm_width=4, width=4):
    case = support.Case('torchsim', width, qm_width)
    run = support.prepare_run(tmp_path / 'case', case, profile, support.darwin_site())
    boot = [f'mol-boot-{i:010d}' for i in range(32)]
    new = [f'mol-0000-{i:010d}' for i in range(8)]
    write_h5(run / 'h5store/data-0000.h5', boot)
    write_h5(run / 'h5store/data-0001.h5', new)
    support.atomic_json(run / 'status.json', dict(current_model_id=1))
    support.atomic_json(run / 'training/model-0001.json',
                        dict(model_id=1, input=dict(ids=boot+new), members=members()))
    support.event('builder_end', run, ids=boot+new)
    support.event('sampler_end', run, ids=new, selected_ids=new, width=width)
    for name in new:
        support.event('qm_end', run, id=name, converged=True)
    support.event('task_done', run, name='label_sequentially', width=qm_width, success=True)
    return run, boot, new


def test_matrix_and_immutable_configurations(tmp_path, profile):
    cases = support.matrix()
    assert len(cases) == len({c.name for c in cases}) == 12
    for case in cases:
        run = support.prepare_run(tmp_path / case.name, case, profile, support.darwin_site())
        master = support.read_json(run / 'master_config.json')
        sampler = support.read_json(run / 'mlmd_config.json')
        qm = support.read_json(run / 'orca_config.json')
        ml = support.read_json(run / 'hippynn_config.json')
        assert master['maximum_builder_structures'] == master['parallel_samplers'] == 4
        assert 'maximum_completed_QM' not in master
        assert ml['n_models'] == 4 and ml['controller_options']['max_epochs'] == 1000
        assert (sampler['Escut'], sampler['Fscut'], sampler['distcut']) == (20, .3, .7)
        assert sampler['maxt'] == .1
        assert 'PBE def2-SVP' in qm['orcasimpleinput'] and qm['ncpu'] == 8
        assert ('batching' in qm) == (case.qm_width == 4)
        assert ('batching' in sampler) == (case.width == 4)
        assert ('driver' in sampler) == (case.engine != 'ase' or case.width == 4)
        support.prepare_run(run, case, profile, support.darwin_site())
        sampler['Fscut'] = .001
        support.atomic_json(run / 'mlmd_config.json', sampler)
        with pytest.raises(ValueError, match='Immutable'):
            support.prepare_run(run, case, profile, support.darwin_site())


def test_completion_requires_new_lineage_and_real_batch_execution(tmp_path, profile):
    run, boot, new = accepted_case(tmp_path, profile)
    assert support.completion(run)['qualifying_ids'] == new
    training_path = run / 'training/model-0001.json'
    training = support.read_json(training_path)
    training['input']['ids'] = boot  # A late bootstrap-only retraining is insufficient.
    support.atomic_json(training_path, training)
    assert support.completion(run) is None
    training['input']['ids'] = boot + new[:7]
    support.atomic_json(training_path, training)
    assert support.completion(run) is None
    training['input']['ids'] = boot + new
    training['members'][0]['epochs'] = 0
    support.atomic_json(training_path, training)
    assert support.completion(run) is None
    training['members'][0]['epochs'] = 2
    support.atomic_json(training_path, training)
    for path in (run / 'events').glob('*.json'):
        if support.read_json(path)['kind'] == 'task_done':
            path.unlink()
    support.event('task_done', run, name='label_sequentially', width=1, success=True)
    assert support.completion(run) is None  # Configured width four is not evidence.


def test_partial_status_and_failed_labels_cannot_pass(tmp_path, profile):
    run, _, _ = accepted_case(tmp_path, profile)
    (run / 'status.json').write_text('{"current_model_id":')
    assert support.completion(run) is None
    support.atomic_json(run / 'status.json', dict(current_model_id=0))
    assert support.completion(run) is None
    support.atomic_json(run / 'status.json', dict(current_model_id=1))
    for path in (run / 'events').glob('*.json'):
        rec = support.read_json(path)
        if rec['kind'] == 'qm_end':
            rec['converged'] = False
            support.atomic_json(path, rec)
            break
    assert support.completion(run) is None


def test_seed_cases_are_independent_and_hash_checked(tmp_path, profile):
    seed = tmp_path / 'seed'
    data = seed / 'h5store/data-0000.h5'
    write_h5(data, ['mol-boot-0'])
    model = seed / 'models/model-0000/model-00/best_model.pt'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'weights')
    hashes = {str(p.relative_to(seed)): support.fingerprint(p) for p in (data, model)}
    support.atomic_json(seed / 'seed.json', dict(files_sha256=hashes, ids=['mol-boot-0']))
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    support.seed_case(seed, first)
    support.seed_case(seed, second)
    (first / model.relative_to(seed)).write_bytes(b'new weights')
    assert (second / model.relative_to(seed)).read_bytes() == model.read_bytes() == b'weights'
    assert not (first / 'status.json').exists()
    with pytest.raises(ValueError, match='already seeded'):
        support.seed_case(seed, first)
    model.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Seed changed'):
        support.seed_case(seed, tmp_path / 'third')


def test_actual_submission_widths_include_packed_qm(h2_atoms):
    from alframework.tools.labeling import label_sequentially
    from alframework.tools.molecules_class import MoleculesObject
    molecules = [MoleculesObject(h2_atoms.copy(), str(i)) for i in range(4)]
    assert support.submitted_width(label_sequentially, (molecules, None, {}), {}) == (4, ['0', '1', '2', '3'])
    assert support.submitted_width(tasks.build_water.func, (), dict(moleculeids=['a','b'], builder_config={},
        molecule_library_dir='/', cell_range=[], solute_molecule_options=[], Rrange=[])) == (2, ['a', 'b'])


def test_cleanup_cancels_only_journalled_jobs(tmp_path):
    support.event('job_submitted', tmp_path, job_id='123')
    support.event('job_submitted', tmp_path, job_id='124')
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        # An unrelated active job must neither be cancelled nor prevent cleanup.
        return SimpleNamespace(returncode=0, stdout='999\n' if command[0] == 'squeue' else '', stderr='')

    assert support.cleanup_jobs(tmp_path, execute=execute) == ['123', '124']
    assert commands[0] == ['scancel', '123', '124']
    assert '999' not in commands[0]
    support.event('job_submitted', tmp_path, job_id='123;scancel --user=someone')
    with pytest.raises(ValueError, match='Invalid owned'):
        support.cleanup_jobs(tmp_path, execute=execute)


def test_real_parsl_submission_audit(tmp_path, monkeypatch, h2_atoms):
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from alframework.tools.labeling import label_sequentially
    from alframework.tools.molecules_class import MoleculesObject
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='qm', max_threads=1)],
                            run_dir=str(tmp_path / 'runinfo')))
    runner.audit_submissions(dfk, tmp_path)

    def label(molecule_object):
        return molecule_object

    packed = python_app(executors=['qm'])(label_sequentially)
    try:
        result = packed([MoleculesObject(h2_atoms.copy(), str(i)) for i in range(4)], label, {}).result()
        assert len(result) == 4
    finally:
        dfk.cleanup()
        parsl.clear()
    records = support.events(tmp_path)
    assert any(r['kind'] == 'task_done' and r['width'] == 4 and r['success'] for r in records)


def test_deadline_stops_only_its_process_and_verifies_cleanup(tmp_path, monkeypatch, profile):
    run = support.prepare_run(tmp_path / 'run', None, profile, support.darwin_site())
    (run / 'events').mkdir(exist_ok=True)
    class Process:
        pid = 54321
        returncode = None
        def poll(self):
            return self.returncode
        def send_signal(self, sig):
            self.signal = sig
        def wait(self, timeout):
            self.returncode = 0
    process = Process()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, 'cleanup_jobs', lambda path: [])
    with pytest.raises(TimeoutError):
        runner.launch(run, 'coordinator', deadline=0)
    assert process.signal == runner.signal.SIGINT
    assert support.read_json(run / 'cleanup.json')['verified']


def test_real_multi_builder_adapter_preserves_ids_and_geometry(tmp_path, monkeypatch, profile):
    from alframework.tools.tools import load_config_file
    run = support.prepare_run(tmp_path / 'builder', None, profile, support.darwin_site())
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    builder = load_config_file(run / 'builder_config.json', str(run) + '/')
    result = tasks.build_water.func(['a', 'b', 'c', 'd'], builder,
        builder['molecule_library_dir'], builder['cell_range'], builder['solute_molecule_options'], builder['Rrange'])
    result = tasks.unwrap(result)
    assert [m.get_moleculeid() for m in result] == ['a', 'b', 'c', 'd']
    record = next(r for r in support.events(run) if r['kind'] == 'builder_end')
    assert record['width'] == 4 and all(all(a['pbc']) for a in record['structures'])


def test_member_reports_require_actual_epochs_and_capture_retention(tmp_path):
    member = tmp_path / 'model-00'
    member.mkdir()
    for name in ['best_checkpoint.pt', 'best_model.pt', 'experiment_structure.pt']:
        (member / name).write_bytes(b'checkpoint')
    log = 'Array Shapes After Cleaning\n| coordinates | float32 | (40, 30, 3) |\n'
    for split, count in [('train', 33), ('valid', 4), ('test', 3)]:
        log += f'Arrays for split: {split}\n| coordinates | float32 | ({count}, 30, 3) |\n'
    (member / 'training_log.txt').write_text(log + 'Epoch 0:\nEpoch 1:\nTraining complete.\n')
    report, = support.member_reports(tmp_path)
    assert report['epochs'] == 2 and report['complete']
    assert report['counts'] == dict(retained=40, train=33, valid=4, test=3)


def test_storage_validation_checks_units_sorting_and_cells(tmp_path, profile):
    run = support.prepare_run(tmp_path / 'storage', None, profile, support.darwin_site())
    structure = dict(numbers=[8, 1, 1], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                     cell=(np.eye(3)*10).tolist(), pbc=[True]*3)
    properties = dict(energy=-76., forces=[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                      dipole=[1., 2., 3.], quadrupole=np.eye(3).tolist())
    support.event('qm_end', run, id='water', converged=True, structure=structure, properties=properties)
    path = run / 'h5store/data-0000.h5'
    path.parent.mkdir()
    master = support.read_json(run / 'master_config.json')
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('water')
        group['_id'] = np.array(['water'], dtype='S')
        group['species'] = np.array(['H','H','O'], dtype='S')
        group['coordinates'] = [np.array(structure['positions'])[[1,2,0]]]
        group['cell'] = [structure['cell']]
        for key, (name, kind, factor) in master['properties_list'].items():
            value = np.array(properties[key])
            if kind == 'atomic':
                value = value[[1,2,0]]
            group[name] = [value*factor]
    assert support.validate_storage(run) == 1
    with h5py.File(path, 'a') as handle:
        handle['water/forces'][0,0,0] = 999.
    with pytest.raises(AssertionError):
        support.validate_storage(run)


def test_alchemi_adapter_uses_qualified_interpreter_and_data_only(tmp_path, monkeypatch, profile, h2_atoms):
    import pickle
    from alframework.tools.molecules_class import MoleculesObject
    run = support.prepare_run(tmp_path / 'alchemi', support.Case('alchemi', 4, 4), profile, support.darwin_site())
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    monkeypatch.delenv('ALF_CAMPAIGN_BACKEND', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
    molecules = [MoleculesObject(h2_atoms.copy(), 'a')]

    def execute(command, env, **kwargs):
        assert command[0] == support.darwin_site()['alchemi_python']
        assert env['ALF_CAMPAIGN_BACKEND'] == '1'
        assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-a,GPU-b'
        directory = Path(command[-1])
        with (directory / 'input.pkl').open('rb') as handle:
            request = pickle.load(handle)
        assert request['function'] == 'sample_many'
        assert request['args'][0][0].get_moleculeid() == 'a'
        with (directory / 'output.pkl').open('wb') as handle:
            pickle.dump(request['args'][0], handle, protocol=4)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', execute)
    delegated, output = tasks.alchemi_backend('sample_many', (molecules, {}, '/models/{}', 0, 4))
    assert delegated and output[0].get_moleculeid() == 'a'
    monkeypatch.setenv('ALF_CAMPAIGN_BACKEND', '1')
    assert tasks.alchemi_backend('sample_many', ()) == (False, None)


def test_data_pickle_is_readable_in_alchemi_python(tmp_path, h2_atoms):
    import pickle
    from alframework.tools.molecules_class import MoleculesObject
    python = Path(support.darwin_site()['alchemi_python'])
    if not python.is_file():
        pytest.skip('qualified ALCHEMI Python environment unavailable')
    path = tmp_path / 'transfer.pkl'
    with path.open('wb') as handle:
        pickle.dump([MoleculesObject(h2_atoms, 'transfer')], handle, protocol=4)
    code = ('import pickle,sys; a=pickle.load(open(sys.argv[1],"rb")); '
            'assert a[0].get_moleculeid()=="transfer"; assert len(a[0])==2')
    import os
    result = subprocess.run([str(python), '-B', '-c', code, str(path)],
        env=dict(os.environ, PYTHONPATH=str(support.SOURCE)), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_resource_profile_exposes_mpi_slots_without_mixing_python_versions(tmp_path, monkeypatch, profile):
    import parsl.addresses
    import parsl.config
    import parsl.executors
    import parsl.providers
    run = support.prepare_run(tmp_path / 'resources', support.Case('alchemi', 4, 4), profile, support.darwin_site())
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    class Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(parsl.providers, 'SlurmProvider', Provider)
    monkeypatch.setattr(parsl.executors, 'HighThroughputExecutor', lambda **kwargs: kwargs)
    monkeypatch.setattr(parsl.config, 'Config', lambda **kwargs: kwargs)
    monkeypatch.setattr(parsl.addresses, 'address_by_hostname', lambda: '127.0.0.1')
    spec = importlib.util.spec_from_file_location('campaign_resource_test', CAMPAIGN / 'campaign_resources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executors = {e['label']: e for e in module.config['executors']}
    qm = executors['alf_QM_executor']
    request = qm['provider'].kwargs
    assert request['cores_per_node'] is None  # Avoid multiplying sixteen tasks by sixteen CPUs each.
    assert '--ntasks-per-node=16' in request['scheduler_options']
    assert '--cpus-per-task=1' in request['scheduler_options']
    assert qm['max_workers_per_node'] == 2 and qm['cores_per_worker'] == 8
    assert qm['cpu_affinity'] == 'block'
    transport = executors['alf_sampler_executor']['interchange_launch_cmd'][0]
    assert transport == support.darwin_site()['base_python']
    assert all(e['provider'].kwargs['max_blocks'] == 1 for e in executors.values())


def test_retry_preserves_original_bootstrap_and_overall_budget(tmp_path, monkeypatch, profile):
    profile = dict(profile, stage_seconds=100, case_seconds=30, cleanup_seconds=5)
    support.atomic_json(tmp_path / 'profile.json', profile)
    support.atomic_json(tmp_path / 'site.json', {})
    clock = [20.]
    calls = []
    monkeypatch.setattr(runner.time, 'time', lambda: clock[0])

    def launch(run, operation, deadline, **kwargs):
        calls.append((operation, deadline))
        clock[0] += 1
        return {'model_id': 0}

    monkeypatch.setattr(runner, 'launch', launch)
    monkeypatch.setattr(runner, 'freeze_seed', lambda *args: None)
    monkeypatch.setattr(runner, 'seed_case', lambda *args: None)
    monkeypatch.setattr(runner, 'case_metrics', lambda *args: {})
    assert runner.suite(tmp_path, selected='ase-w1-q1', budget_start=0.)
    assert calls[:2] == [('preflight', 95.), ('coordinator', 95.)]
    assert support.read_json(tmp_path / 'execution.json')['overall_deadline'] == 400.
    assert all(deadline <= 400. for _, deadline in calls)
    assert support.read_json(tmp_path / 'result.json')['complete_matrix'] is False
