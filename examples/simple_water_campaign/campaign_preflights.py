"""Real-backend preflights; outputs are evidence, never bootstrap training data."""
from pathlib import Path
import copy
import time

from campaign_support import atomic_json, events, read_json


def native_preflight(run):
    import numpy as np
    from ase.io import write
    from campaign_tasks import build_molecules, label_molecules
    run = Path(run)
    builder = read_json(run / 'builder_config.json')
    qm = read_json(run / 'orca_config.json')
    molecules = []
    for start in (0, 4):
        ids = [f'preflight-{i}' for i in range(start, start+4)]
        molecules.extend(build_molecules(ids, builder).result())
    assert len({m.get_moleculeid() for m in molecules}) == 8
    assert all(not m.get_atoms().pbc.any() for m in molecules)
    write(run / 'preflight.extxyz', [m.get_atoms() for m in molecules[:4]])
    reference = [label_molecules([copy.deepcopy(m)], qm).result()[0] for m in molecules]
    assert all(m.check_convergence() for m in reference)
    comparisons = []
    for width in (1, 2, 4, 8):
        started = time.time()
        result = []
        for index in range(0, len(molecules), width):
            result.extend(label_molecules(copy.deepcopy(molecules[index:index+width]), qm).result())
        assert all(m.check_convergence() for m in result)
        de, df = 0., 0.
        for actual, expected in zip(result, reference):
            assert actual.get_moleculeid() == expected.get_moleculeid()
            a, b = actual.get_results(), expected.get_results()
            de = max(de, abs(a['energy']-b['energy']))
            df = max(df, float(np.max(np.abs(a['forces']-b['forces']))))
            np.testing.assert_array_equal(actual.get_atoms().numbers, expected.get_atoms().numbers)
            assert a['forces'].shape == (len(actual), 3)
        assert de <= 1e-6 and df <= 1e-5, (de, df)
        comparisons.append(dict(width=width, energy_error_ev=de, force_error_ev_angstrom=df,
                                elapsed_seconds=time.time()-started))
    # Check the sign against a central finite difference on the same GPU backend.
    energies = []
    for shift in (-1e-4, 1e-4):
        molecule = copy.deepcopy(molecules[0])
        molecule.get_atoms().positions[0, 0] += shift
        value = label_molecules([molecule], qm).result()[0]
        assert value.check_convergence()
        energies.append(value.get_results()['energy'])
    gradient_error = abs(reference[0].get_results()['forces'][0, 0] + (energies[1]-energies[0])/2e-4)
    assert gradient_error <= 1e-4, gradient_error
    solves = [r for r in events(run) if r['kind'] == 'native_solve']
    assert {1, 2, 4, 8} <= {r['width'] for r in solves if r['success']}
    assert all(r['success'] for r in solves), 'Fallback cannot establish native batch parity'
    atomic_json(run / 'preflight.json', dict(passed=True, comparisons=comparisons,
                finite_difference_error_ev_angstrom=float(gradient_error)))


def partial_preflight(run, seed):
    from alframework.tools.sampler_coordinator import SamplerCoordinator
    from alframework.tools.labeling import LabelingCoordinator
    from alframework.tools.tools import build_input_dict, load_config_file
    from campaign_tasks import build_water, sample_many, label_water
    run = Path(run)
    master = load_config_file(run / 'master_config.json')
    builder = load_config_file(master['builder_config_path'], master['master_directory'])
    sampler = load_config_file(master['sampler_config_path'], master['master_directory'])
    qm = load_config_file(master['QM_config_path'], master['master_directory'])
    if sampler.get('meta_dir') is not None:
        Path(sampler['meta_dir']).mkdir(parents=True, exist_ok=True)
    sampler.update(driver=dict(engine='ase', strategy='batched'),
                   batching=dict(batch_size=4, partial_policy='timeout', max_wait_seconds=60))
    qm['batching'] = dict(mode='sequential', batch_size=4, partial_policy='timeout', max_wait_seconds=60)
    args = build_input_dict(build_water.func, [dict(moleculeids=[f'tail-{i}' for i in range(3)],
                                                    builder_config=builder), builder], raise_on_fail=True)
    molecules = build_water(**args).result()
    sampling = SamplerCoordinator(sample_many, 'campaign_tasks.sample_many', sampler)
    labeling = LabelingCoordinator(label_water, 'campaign_tasks.label_water', qm, master)
    task_inputs = dict(sampler_config=sampler, model_path=str(Path(seed) / 'models/model-{:04d}'),
                       current_model_id=0, gpus_per_node=4)
    qm_inputs = dict(QM_config=qm, QM_scratch_dir=str(run / 'partial_orca'), properties_list=master['properties_list'])
    started = time.time()
    for molecule in molecules:
        sampling.submit(copy.deepcopy(molecule), [task_inputs])
        labeling.submit(copy.deepcopy(molecule), [qm_inputs])
    submitted = dict(sampling=0, labeling=0)
    while not all(submitted.values()):
        submitted['sampling'] += sampling.submit_ready([task_inputs])
        submitted['labeling'] += labeling.submit_ready([qm_inputs])
        time.sleep(.25)
    while sampling.queue.get_completed_number() < 3 or labeling.queue.get_completed_number() < 3:
        time.sleep(.25)
    samples, sample_failures = sampling.queue.get_task_results()
    labels, label_failures = labeling.queue.get_task_results()
    assert len(samples) == len(labels) == 3 and sample_failures == label_failures == 0
    assert all(m.check_convergence() for m in labels)
    records = events(run)
    for name in ('sample_many', 'label_sequentially'):
        assert any(r['kind'] == 'task_done' and r['name'] == name and r['width'] == 3 and r['success'] for r in records)
    atomic_json(run / 'partial.json', dict(passed=True, sampler_width=3, labeling_width=3,
                max_wait_seconds=60, elapsed_seconds=time.time()-started, forced_flush=False))
