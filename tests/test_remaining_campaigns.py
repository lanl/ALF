"""Acceptance regressions for the remaining real-backend campaign harness."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/simple_water_campaign'))
import campaign_support as support
import campaign_tasks as tasks
from campaign_evidence import campaign_accepted, failure_target, replay_cohorts, sampling_audit
from tests.test_water_campaign import members, write_h5


def profile(campaign):
    return dict(support.read_json(support.HERE / 'profile.json'),
                **support.read_json(support.HERE / 'campaigns.json')[campaign])


def test_campaign_configs_and_molecular_units(tmp_path):
    assert [len(support.matrix(c)) for c in ('core', 'cohort', 'failures', 'pyseqm', 'mi250')] == [12, 12, 3, 2, 4]
    for campaign in ('cohort', 'failures', 'pyseqm', 'mi250'):
        cases = support.matrix(campaign)
        assert len({c.name for c in cases}) == len(cases)
        for case in cases:
            run = support.prepare_run(tmp_path / campaign / case.name, case, profile(campaign), support.darwin_site())
            sampler = support.read_json(run / 'mlmd_config.json')
            master = support.read_json(run / 'master_config.json')
            assert (sampler['Escut'], sampler['Fscut'], sampler['distcut'], sampler['maxt']) == (20, .3, .7, .1)
            if campaign == 'cohort':
                acquisition = sampler['acquisition']
                assert acquisition['trajectory_policy'] == case.policy
                assert acquisition['selection'] == dict(mode='cohort_top_k', cohort_size=8, k=4)
                if case.ranking == 'summed':
                    assert acquisition['weights'] == dict(energy=1., force_rms=1.)
            if campaign == 'pyseqm':
                assert set(master['properties_list']) == {'energy', 'forces'}
                assert all(v[-1] == 1 for v in master['properties_list'].values())
                assert support.read_json(run / 'orca_config.json')['batching']['mode'] == 'native'
                assert support.read_json(run / 'hippynn_config.json')['network_params']['possible_species'] == [0, 1, 6, 8]


def test_molecular_builder_is_reproducible_heterogeneous_and_nonperiodic(tmp_path, monkeypatch):
    run = support.prepare_run(tmp_path, None, profile('pyseqm'), support.darwin_site())
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    config = support.read_json(run / 'builder_config.json')
    ids = [f'mol-boot-{i:010d}' for i in range(4)]
    first = tasks.unwrap(tasks.build_molecules.func(ids, config))
    second = tasks.unwrap(tasks.build_molecules.func(ids, config))
    from alframework.samplers.geometry import minimum_distance
    assert [len(m) for m in first] == [3, 6, 3, 6]
    for a, b in zip(first, second):
        assert not a.get_atoms().pbc.any() and minimum_distance(a.get_atoms()) >= .7
        np.testing.assert_array_equal(a.get_atoms().positions, b.get_atoms().positions)
    assert not np.array_equal(first[0].get_atoms().positions, first[2].get_atoms().positions)


def cohort_case(run, policy='continue'):
    from ase import Atoms
    from alframework.tools.cohorts import CohortPool
    from alframework.tools.molecules_class import MoleculesObject
    from tests.test_cohorts import candidate, returned
    case = support.Case('ase', 4, 4, 'cohort', policy=policy)
    support.prepare_run(run, case, profile('cohort'), support.darwin_site())
    selected_ids, parents = [], []
    for cohort_index in range(2):
        pool = CohortPool(dict(cohort_size=8, k=4))
        results = []
        for index in range(8):
            name = f'mol-0000-{8*cohort_index+index:010d}'
            parent = MoleculesObject(Atoms('H'), name)
            pool.admit(parent, model_id=0)
            metadata = dict(parent.get_metadata())
            # Each pool has its own counter in this fixture; assign unique audit IDs.
            metadata['cohort_id'] = f'cohort-0000-{cohort_index:06d}'
            support.event('cohort_admitted', run, id=name, metadata=metadata, pinned_model_id=0, current_model_id=0)
            candidates = [candidate(name, step, float(index*2+step)) for step in (1, 2)]
            for c in candidates:
                support.event('candidate_created', run, candidate=c)
            result = returned(name, 'eligible', candidates)
            pool.record(result)
            support.event('cohort_terminal', run, id=name, metadata=result.get_metadata())
            results.append(result)
            parents.append(name)
        frames = pool.ready()
        summary = dict(pool.finished[0], cohort_id=metadata['cohort_id'])
        geometries = [dict(**tasks.geometry(m), metadata=m.get_metadata()) for m in frames]
        support.event('cohort_selected', run, summary=summary, frames=geometries)
        selected_ids.extend(m.get_moleculeid() for m in frames)
        support.event('sampler_end', run, ids=[m.get_moleculeid() for m in results], selected_ids=[], width=4)
    support.event('builder_end', run, ids=parents)
    support.event('task_done', run, name='label_sequentially', width=4, success=True)
    for name in selected_ids:
        support.event('qm_end', run, id=name, converged=True)
    write_h5(run / 'h5store/data-0000.h5', selected_ids)
    support.atomic_json(run / 'training/model-0001.json', dict(model_id=1, input=dict(ids=selected_ids), members=members()))
    support.atomic_json(run / 'status.json', dict(current_model_id=1))
    return case, selected_ids


def test_cohort_lineage_uses_selected_child_ids_and_checks_replay(tmp_path):
    case, ids = cohort_case(tmp_path)
    assert support.completion(tmp_path)['qualifying_ids'] == sorted(ids)
    assert len(replay_cohorts(support.events(tmp_path))) == 2
    # A stored selected frame without its built parent cannot establish lineage.
    for path in (tmp_path / 'events').glob('*.json'):
        if support.read_json(path)['kind'] == 'builder_end':
            path.unlink()
    assert support.completion(tmp_path) is None


def test_replay_rejects_incorrect_top_k_and_completion_rejects_missing_candidates(tmp_path):
    cohort_case(tmp_path)
    records = support.events(tmp_path)
    selected = next(r for r in records if r['kind'] == 'cohort_selected')
    selected['summary']['selected'].reverse()
    with pytest.raises(ValueError, match='global top-K'):
        replay_cohorts(records)
    for path in (tmp_path / 'events').glob('*.json'):
        if support.read_json(path)['kind'] == 'candidate_created':
            path.unlink()
    with pytest.raises(ValueError, match='generated candidate'):
        support.completion(tmp_path)


def test_label_failure_cases_wait_for_the_lost_label_before_retraining(tmp_path):
    thresholds = {}
    for case in support.matrix('failures'):
        run = support.prepare_run(tmp_path / case.name, case, profile('failures'), support.darwin_site())
        thresholds[case.name] = support.read_json(run / 'master_config.json')['save_h5_threshold']
    needed = profile('failures')['new_labels_per_case']
    assert thresholds['label_exception'] == thresholds['label_unconverged'] == needed + 1
    assert thresholds['trajectory_failure'] == 7


def test_cleanup_retries_slow_slurm_commands_within_the_window(tmp_path, monkeypatch):
    support.event('job_submitted', tmp_path, job_id='123')
    monkeypatch.setattr(support.time, 'sleep', lambda seconds: None)
    calls = []

    def execute(command, **kwargs):
        calls.append(command[0])
        if calls.count('scancel') == 1:
            raise support.subprocess.TimeoutExpired(command, kwargs['timeout'])
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    assert support.cleanup_jobs(tmp_path, execute=execute) == ['123']
    assert calls.count('scancel') == 2 and calls[-1] == 'squeue'


def test_cleanup_gives_up_when_slurm_never_responds(tmp_path, monkeypatch):
    support.event('job_submitted', tmp_path, job_id='123')
    monkeypatch.setattr(support.time, 'sleep', lambda seconds: None)

    def execute(command, **kwargs):
        raise support.subprocess.TimeoutExpired(command, kwargs['timeout'])

    with pytest.raises(RuntimeError, match='unverified'):
        support.cleanup_jobs(tmp_path, execute=execute, timeout=0)


def test_controlled_label_failure_requires_surviving_sibling_and_exclusion():
    case = support.matrix('failures')[0]
    records = [dict(kind='fault_injected', fault=case.fault, id='bad'),
               dict(kind='task_submitted', name='label_sequentially', ids=['bad', 'good'])]
    assert campaign_accepted(case, records, {'good'}, {'good'}, {'good'})
    assert not campaign_accepted(case, records, {'other'}, {'other'}, {'other'})
    with pytest.raises(ValueError, match='storage/training'):
        campaign_accepted(case, records, {'good'}, {'bad', 'good'}, {'good'})
    assert not campaign_accepted(case, records + records[:1], {'good'}, {'good'}, {'good'})


def test_native_fallback_is_not_full_batch_evidence():
    case = support.matrix('pyseqm')[1]
    records = [dict(kind='native_solve', width=4, success=False), dict(kind='native_solve', width=1, success=True)]
    assert not campaign_accepted(case, records, set(), set(), set())
    assert campaign_accepted(case, records + [dict(kind='native_solve', width=4, success=True)], set(), set(), set())


def test_partition_names_cannot_substitute_for_actual_target_hardware():
    amd = dict(hardware='AMD Instinct MI250X', hip='6.4', cuda=None)
    nvidia = dict(hardware='NVIDIA A100', hip=None, cuda='12.8')
    support.check_campaign_gpu('mi250', amd)
    support.check_campaign_gpu('pyseqm', nvidia)
    with pytest.raises(ValueError, match='MI250'):
        support.check_campaign_gpu('mi250', nvidia)
    with pytest.raises(ValueError, match='MI250'):
        support.check_campaign_gpu('mi250', dict(amd, hardware='AMD MI300X'))
    with pytest.raises(ValueError, match='NVIDIA'):
        support.check_campaign_gpu('pyseqm', amd)


def test_fresh_worker_initializes_selected_gpu_before_resetting_allocator_stats(monkeypatch):
    import torch
    calls = []
    class Device:
        def __init__(self, device):
            assert device == 'cuda:1'
        def __enter__(self):
            calls.append('selected')
        def __exit__(self, *args):
            calls.append('restored')
    def reset(device):
        assert calls == ['selected', 'initialized']
        assert device == 'cuda:1'
        calls.append('reset')
    monkeypatch.setattr(torch.cuda, 'device', Device)
    monkeypatch.setattr(torch.cuda, 'init', lambda: calls.append('initialized'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    tasks.start_gpu_memory_measurement('cuda:1')
    assert calls == ['selected', 'initialized', 'reset', 'restored']


def test_fault_wrapper_retains_actual_candidate_before_failure(tmp_path, monkeypatch):
    from tests.test_cohorts import WATER, cohort_config
    from tests.helpers.fakes import spring_ensemble
    from alframework.samplers.batched_mlmd import sample_trajectories
    from alframework.tools.cohorts import CohortPool
    from alframework.tools.molecules_class import MoleculesObject
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(tmp_path))
    config = dict(cohort_config(4, 4, 'continue'), maxt=.02)
    pool = CohortPool(config['acquisition']['selection'])
    molecules = [MoleculesObject(WATER.copy(), f'mol-0000-{i:010d}') for i in range(4)]
    for molecule in molecules:
        pool.admit(molecule, 0)
        support.event('cohort_admitted', tmp_path, id=molecule.get_moleculeid(), metadata=molecule.get_metadata())
    with sampling_audit(molecules, 'trajectory_failure', config):
        result = sample_trajectories(molecules, spring_ensemble(), config)
    failed = [m for m in result if m.get_metadata()['trajectory_outcome'] == 'failed']
    assert len(failed) == 1
    metadata = failed[0].get_metadata()
    assert metadata['trajectory_outcome'] == 'failed'
    assert metadata['acquisition_candidates']
    records = support.events(tmp_path)
    assert sum(r['kind'] == 'fault_injected' for r in records) == 1
    assert any(r['kind'] == 'candidate_created' and r['generated_geometry_bytes'] > 0 for r in records)
    for molecule in result:
        pool.record(molecule)
    assert any(m.get_metadata()['trajectory_id'] == failed[0].get_moleculeid() for m in pool.ready())
    fault = next(r for r in records if r['kind'] == 'fault_injected')
    assert fault['placement'] == 'after_final_observation'
    assert fault['frame_id'] in {c['frame_id'] for c in metadata['acquisition_candidates']}
    assert metadata['step'] == 36


def test_failure_placement_waits_for_full_cohort_and_preserves_losing_candidates(tmp_path):
    from tests.test_cohorts import candidate, cohort_config, member
    from alframework.tools.cohorts import CohortPool
    config = dict(cohort_config(8, 4, 'continue'), maxt=.005)
    pool = CohortPool(config['acquisition']['selection'])
    molecules = [member(f't{i}') for i in range(8)]
    for molecule in molecules:
        pool.admit(molecule, 0)
        support.event('cohort_admitted', tmp_path, id=molecule.get_moleculeid(), metadata=molecule.get_metadata())
    for i in range(8):
        support.event('candidate_created', tmp_path, candidate=candidate(f't{i}', 1, 100. if i < 4 else 1.))
    current = molecules[4:]
    engine = SimpleNamespace(steps=[6]*4)
    assert failure_target(current, list(range(4)), engine, config, tmp_path) is None
    for molecule in molecules[:4]:
        support.event('cohort_terminal', tmp_path, id=molecule.get_moleculeid())
    # Naturally losing frames are retained in the audit, never boosted or forced in.
    assert failure_target(current, list(range(4)), engine, config, tmp_path) is None
    support.event('candidate_created', tmp_path, candidate=candidate('t4', 6, 200.))
    assert failure_target(current, list(range(4)), SimpleNamespace(steps=[1]*4), config, tmp_path) is None
    index, frame, rank = failure_target(current, list(range(4)), engine, config, tmp_path)
    assert (index, frame['frame_id'], rank) == (0, 't4:step-000000006', 0)


def test_real_sampler_coordinator_pins_admission_and_task_generation(tmp_path):
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from ase import Atoms
    from alframework.tools.molecules_class import MoleculesObject
    from alframework.tools.sampler_coordinator import SamplerCoordinator
    @python_app(executors=['pinning'])
    def sampler(molecule_objects, sampler_config, current_model_id):
        for m in molecule_objects:
            m.update_metadata(dict(executed_model_id=current_model_id, trajectory_outcome='completed', acquisition_candidates=[]))
        return molecule_objects
    config = dict(driver=dict(engine='ase', strategy='batched'),
                  batching=dict(batch_size=2, partial_policy='timeout', max_wait_seconds=0),
                  acquisition=dict(selection=dict(mode='cohort_top_k', cohort_size=3, k=1)), Escut=20, Fscut=.3)
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='pinning', max_threads=1)], run_dir=str(tmp_path / 'parsl')))
    try:
        coordinator = SamplerCoordinator(sampler, 'test.sampler', config)
        for name, model in [('a', 0), ('b', 0), ('c', 1), ('d', 1)]:
            coordinator.submit(MoleculesObject(Atoms('H'), name), [dict(sampler_config=config, current_model_id=model)])
        coordinator.submit_ready([dict(sampler_config=config, current_model_id=1)])
        # Wait through actual Parsl results, then use the normal harvest path.
        for task in list(dfk.tasks.values()):
            task['app_fu'].result()
        results, failed = coordinator.queue.get_task_results()
        assert failed == 0
        assert {m.get_moleculeid(): m.get_metadata()['executed_model_id'] for m in results} == dict(a=0, b=0, c=0, d=1)
        assert coordinator.harvest(results) == []
        assert coordinator.pool.finished[0]['model_id'] == 0
        assert coordinator.status()['open'][0]['model_id'] == 1
    finally:
        dfk.cleanup()
        parsl.clear()


@pytest.mark.parametrize('width', [1, 2])
def test_real_label_coordinator_submits_timeout_tails_without_flush(tmp_path, width):
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from ase import Atoms
    from alframework.tools.labeling import LabelingCoordinator
    from alframework.tools.molecules_class import MoleculesObject
    @python_app(executors=['tail'])
    def label(molecule_object):
        return molecule_object
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='tail', max_threads=1)], run_dir=str(tmp_path / 'parsl')))
    try:
        coordinator = LabelingCoordinator(label, 'test.label', dict(batching=dict(
            mode='sequential', batch_size=4, partial_policy='timeout', max_wait_seconds=0)))
        for i in range(width):
            coordinator.submit(MoleculesObject(Atoms('H'), str(i)), [])
        assert coordinator.submit_ready([]) == 1
        for task in list(dfk.tasks.values()):
            task['app_fu'].result()
        results, failures = coordinator.queue.get_task_results()
        assert failures == 0 and len(results) == width
    finally:
        dfk.cleanup()
        parsl.clear()


def test_seed_path_traversal_is_rejected(tmp_path):
    seed = tmp_path / 'seed'
    outside = tmp_path / 'outside'
    outside.write_text('not a seed')
    support.atomic_json(seed / 'seed.json', dict(ids=[], files_sha256={'../outside': support.fingerprint(outside)}))
    with pytest.raises(ValueError, match='Invalid seed path'):
        support.seed_case(seed, tmp_path / 'case')


@pytest.mark.parametrize('campaign', ['pyseqm', 'mi250'])
def test_resource_profiles_serialize_native_qm_and_keep_training_on_nvidia(tmp_path, monkeypatch, campaign):
    import importlib.util
    import parsl.addresses
    import parsl.config
    import parsl.executors
    import parsl.providers
    site = dict(support.darwin_site(), cxx='/qualified/bin/g++', cc='/qualified/bin/gcc',
                cuda_module='cuda/13.2.1')
    run = support.prepare_run(tmp_path, support.matrix(campaign)[0], profile(campaign), site)
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    class Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setattr(parsl.providers, 'SlurmProvider', Provider)
    monkeypatch.setattr(parsl.executors, 'HighThroughputExecutor', lambda **kwargs: kwargs)
    monkeypatch.setattr(parsl.config, 'Config', lambda **kwargs: kwargs)
    monkeypatch.setattr(parsl.addresses, 'address_by_hostname', lambda: '127.0.0.1')
    spec = importlib.util.spec_from_file_location('remaining_resource_test', support.HERE / 'campaign_resources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executors = {e['label']: e for e in module.config['executors']}
    sampler = executors['alf_sampler_executor']
    training = executors['alf_ML_executor']
    assert 'export CXX=/qualified/bin/g++' in training['provider'].kwargs['worker_init']
    assert 'module load cuda/13.2.1' in training['provider'].kwargs['worker_init']
    assert training['provider'].kwargs['partition'] == support.darwin_site()['gpu_partition']
    if campaign == 'pyseqm':
        assert set(executors) == {'alf_ML_executor', 'alf_sampler_executor'}
        assert sampler['max_workers_per_node'] == 1
        assert tasks.label_molecules.executors == ['alf_sampler_executor']
        assert support.darwin_site()['pyseqm_source'] in sampler['provider'].kwargs['worker_init']
    else:
        assert 'module load cuda/13.2.1' not in sampler['provider'].kwargs['worker_init']
        assert sampler['provider'].kwargs['partition'] == support.darwin_site()['rocm_partition']
        assert sampler['interchange_launch_cmd'][0] == support.darwin_site()['rocm_python']
        assert support.darwin_site()['rocm_module'] in sampler['provider'].kwargs['worker_init']


def test_verified_import_rejects_tampering_units_and_changed_science(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write
    from campaign_evidence import verify_seed
    import alframework.ml_interfaces.hippynn_interface as interface
    bootstrap = support.prepare_run(tmp_path / 'old/bootstrap', None,
        support.read_json(support.HERE / 'profile.json'), support.darwin_site())
    destination = support.prepare_run(tmp_path / 'new/bootstrap', None, profile('cohort'), support.darwin_site())
    seed = tmp_path / 'old/seed'
    ids = [f'mol-boot-{i:010d}' for i in range(32)]
    write_h5(seed / 'h5store/data-0000.h5', ids)
    reports = []
    for index in range(4):
        directory = seed / f'models/model-0000/model-{index:02d}'
        directory.mkdir(parents=True)
        for name in ('best_model.pt', 'best_checkpoint.pt', 'experiment_structure.pt'):
            (directory / name).write_bytes(b'fixture')
        reports.append(dict(member=directory.name, complete=True, epochs=1,
            files_sha256={p.name: support.fingerprint(p) for p in directory.iterdir()}))
    write(seed / 'preflight.extxyz', [Atoms('H', positions=[[0, 0, 0]])])
    hashes = {str(p.relative_to(seed)): support.fingerprint(p) for p in seed.rglob('*') if p.is_file()}
    support.atomic_json(seed / 'seed.json', dict(files_sha256=hashes, ids=ids))
    support.atomic_json(bootstrap / 'result.json', dict(status='passed', accepted=dict(training=dict(
        model_id=0, input=support.h5_inventory(seed / 'h5store'), members=reports))))
    class Model:
        def calculate(self, atoms, properties):
            self.results = dict(energy=0., forces=np.zeros((len(atoms), 3)))
    monkeypatch.setattr(interface, 'HIPNN_ASE_calculator', lambda *a, **k: Model())
    assert verify_seed(seed, destination)['verified_members'] == 4
    path = destination / 'mlmd_config.json'
    original = support.read_json(path)
    support.atomic_json(path, dict(original, Fscut=.01))
    with pytest.raises(ValueError, match='mlmd_config'):
        verify_seed(seed, destination)
    support.atomic_json(path, original)
    path = destination / 'master_config.json'
    original = support.read_json(path)
    changed = dict(original, properties_list={'energy': ['energy', 'system', 1.]})
    support.atomic_json(path, changed)
    with pytest.raises(ValueError, match='units'):
        verify_seed(seed, destination)
    support.atomic_json(path, original)
    (seed / 'models/model-0000/model-00/best_model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Invalid seed file'):
        verify_seed(seed, destination)


def test_combined_runner_stops_after_unverified_owned_job_cleanup(tmp_path, monkeypatch):
    import remaining
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            support.event('job_submitted', tmp_path / 'cohort/bootstrap', job_id='12345')
        def wait(self):
            return 1
    monkeypatch.setattr(remaining.subprocess, 'Popen', Process)
    monkeypatch.setattr(remaining, 'source_identity', lambda: {})
    with pytest.raises(RuntimeError, match='Unverified cleanup'):
        remaining.run_remaining(tmp_path, tmp_path / 'seed')
    assert len(calls) == 1
    assert support.read_json(tmp_path / 'result.json')['status'] == 'incomplete'


def test_combined_retry_forwards_original_cohort_budget(tmp_path, monkeypatch):
    import remaining
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            directory = Path(command[command.index('--run-root')+1])
            support.atomic_json(directory / 'result.json', dict(status='passed'))
        def wait(self):
            return 0
    monkeypatch.setattr(remaining.subprocess, 'Popen', Process)
    monkeypatch.setattr(remaining, 'source_identity', lambda: {})
    remaining.run_remaining(tmp_path, tmp_path / 'seed', budget_start=10.)
    assert commands[0][-2:] == ['--budget-start', '10.0']
    assert all('--budget-start' not in command for command in commands[1:])
    assert support.read_json(tmp_path / 'execution.json')['start'] == 10.
    assert support.read_json(tmp_path / 'result.json')['status'] == 'incomplete'  # Scientific gates remain open.


def test_early_cohort_retry_does_not_refund_case_or_group_deadline(tmp_path, monkeypatch):
    import campaign_evidence
    from tests.test_water_campaign import runner
    p = dict(profile('cohort'), stage_seconds=100, case_seconds=30, cleanup_seconds=5, overall_seconds=400)
    support.atomic_json(tmp_path / 'profile.json', p)
    support.atomic_json(tmp_path / 'site.json', {})
    monkeypatch.setattr(runner.time, 'time', lambda: 50.)
    monkeypatch.setattr(campaign_evidence, 'verify_seed', lambda *a: dict(initial_training={}))
    monkeypatch.setattr(runner, 'seed_case', lambda *a: None)
    def unexpected_launch(*args, **kwargs):
        raise AssertionError('An expired case must not allocate resources')
    monkeypatch.setattr(runner, 'launch', unexpected_launch)
    assert not runner.suite(tmp_path, selected='ase-uncertainty-stop', budget_start=0., import_seed=tmp_path / 'seed')
    deadlines = support.read_json(tmp_path / 'cases/ase-uncertainty-stop/deadlines.json')
    assert deadlines['case_deadline'] == 30. and deadlines['group_deadline'] == 100.
    assert deadlines['overall_deadline'] == 400.


def test_partial_preflight_creates_sampling_directory_before_submission(tmp_path, monkeypatch):
    from campaign_preflights import partial_preflight
    run = support.prepare_run(tmp_path, None, profile('failures'), support.darwin_site())
    class StopBeforeSubmission(Exception):
        pass
    class Builder:
        func = staticmethod(tasks.build_water.func)
        def __call__(self, **kwargs):
            assert (run/'sampling').is_dir()
            raise StopBeforeSubmission
    monkeypatch.setattr(tasks, 'build_water', Builder())
    with pytest.raises(StopBeforeSubmission):
        partial_preflight(run, tmp_path/'seed')


def test_nonperiodic_storage_validation_uses_real_alf_no_cell_convention(tmp_path):
    from ase import Atoms
    from alframework.tools.molecules_class import MoleculesObject
    from alframework.tools.tools import store_current_data
    run = support.prepare_run(tmp_path, support.matrix('pyseqm')[0], profile('pyseqm'), support.darwin_site())
    molecule = MoleculesObject(Atoms('HOH', positions=[[1,0,0],[0,0,0],[0,1,0]], cell=[20]*3), 'sample')
    molecule.store_results(dict(energy=-1., forces=np.arange(9.).reshape(3,3)))
    molecule.set_converged_flag(True)
    support.event('qm_end', run, id='sample', converged=True, structure=tasks.geometry(molecule), properties=molecule.get_results())
    (run/'h5store').mkdir()
    store_current_data(str(run/'h5store/data-0000.h5'), [molecule], support.read_json(run/'master_config.json')['properties_list'])
    assert support.validate_storage(run) == 1


def test_selected_retry_does_not_rerun_accepted_campaigns_and_defers_mi350(tmp_path, monkeypatch):
    import remaining
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            directory = Path(command[command.index('--run-root')+1])
            support.atomic_json(directory/'result.json', dict(status='passed'))
        def wait(self):
            return 0
    monkeypatch.setattr(remaining.subprocess, 'Popen', Process)
    monkeypatch.setattr(remaining, 'source_identity', lambda: {})
    remaining.run_remaining(tmp_path, tmp_path/'seed', campaigns=('failures','pyseqm'))
    assert [c[c.index('--campaign')+1] for c in commands] == ['failures','pyseqm']
    report = support.read_json(tmp_path/'result.json')
    assert report['selected_campaigns_passed'] and not report['runnable_campaigns_passed']
    assert report['dependencies']['mi350']['status'] == 'deferred'
