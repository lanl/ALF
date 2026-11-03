"""Acceptance regressions for the remaining real-backend campaign harness."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/simple_water_campaign'))
import campaign_support as support
import campaign_tasks as tasks
from campaign_evidence import campaign_accepted, replay_cohorts, sampling_audit
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


def test_fault_wrapper_retains_actual_candidate_before_failure(tmp_path, monkeypatch):
    from tests.test_cohorts import WATER, cohort_config
    from tests.helpers.fakes import spring_ensemble
    from alframework.samplers.batched_mlmd import sample_trajectories
    from alframework.tools.molecules_class import MoleculesObject
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(tmp_path))
    molecules = [MoleculesObject(WATER.copy(), 'mol-0000-0000000000')]
    with sampling_audit(molecules, 'trajectory_failure'):
        result = sample_trajectories(molecules, spring_ensemble(), cohort_config(8, 4, 'continue'))
    metadata = result[0].get_metadata()
    assert metadata['trajectory_outcome'] == 'failed'
    assert metadata['acquisition_candidates']
    records = support.events(tmp_path)
    assert sum(r['kind'] == 'fault_injected' for r in records) == 1
    assert any(r['kind'] == 'candidate_created' and r['generated_geometry_bytes'] > 0 for r in records)


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
    run = support.prepare_run(tmp_path, support.matrix(campaign)[0], profile(campaign), support.darwin_site())
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
    assert training['provider'].kwargs['partition'] == support.darwin_site()['gpu_partition']
    if campaign == 'pyseqm':
        assert set(executors) == {'alf_ML_executor', 'alf_sampler_executor'}
        assert sampler['max_workers_per_node'] == 1
        assert tasks.label_molecules.executors == ['alf_sampler_executor']
        assert support.darwin_site()['pyseqm_source'] in sampler['provider'].kwargs['worker_init']
    else:
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
