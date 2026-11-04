"""Bounded original-example workflows, with observation-only task adapters."""
import argparse
from dataclasses import asdict
import importlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from parsl import python_app

from campaign_support import (Case, HERE, SOURCE, atomic_json, events, fingerprint,
    h5_inventory, member_reports, read_json, seed_case, source_identity)
from campaign_tasks import geometry, identify, sampler_result, unwrap

FAMILIES = ('simple_water', 'simple_water_multi_builder', 'seeded_active_learning',
            'fine_tuning', 'IL', 'reactive_sampling', 'molten_salt', 'UO2')
WATER = FAMILIES[:4]


def original(role, arguments):
    """Call the exact original example task with its ordinary configuration."""
    from alframework.tools.tools import build_input_dict
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN'])/'run.json')
    example = Path(spec['source'])/'examples'/spec['compatibility']['family']
    sys.path.insert(0, str(example))
    module, name = spec['compatibility']['tasks'][role].rsplit('.', 1)
    task = getattr(importlib.import_module(module), name)
    function = task.func
    dictionaries = [arguments, *[v for k, v in arguments.items() if k.endswith('_config')]]
    kwargs = build_input_dict(function, dictionaries, raise_on_fail=True)
    from campaign_support import event
    event('original_task_called', role=role, task=spec['compatibility']['tasks'][role])
    return unwrap(function(**kwargs))


def build(arguments):
    from campaign_support import event
    identify()
    result = original('builder_task', arguments)
    molecules = result if isinstance(result, list) else [result]
    event('builder_end', ids=[m.get_moleculeid() for m in molecules], width=len(molecules),
          structures=[geometry(m) for m in molecules])
    return result


@python_app(executors=['alf_sampler_executor'])
def build_single(moleculeid, builder_config):
    return build(dict(moleculeid=moleculeid, builder_config=builder_config))


@python_app(executors=['alf_sampler_executor'])
def build_many(moleculeids, builder_config):
    return build(dict(moleculeids=moleculeids, builder_config=builder_config))


@python_app(executors=['alf_sampler_executor'])
def sample(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
    from campaign_support import event
    identify()
    start = time.time()
    event('sampler_start', ids=[molecule_object.get_moleculeid()], width=1, model_id=current_model_id)
    result = original('sampler_task', dict(molecule_object=molecule_object,
        sampler_config=sampler_config, model_path=model_path, current_model_id=current_model_id,
        gpus_per_node=gpus_per_node))
    return sampler_result(result, 1, current_model_id, start)


@python_app(executors=['alf_QM_executor'])
def label(molecule_object, QM_config, QM_scratch_dir, properties_list):
    from campaign_support import event
    identify()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN'])/'run.json')
    if spec['site'].get('vasp_pp_path'):
        os.environ['VASP_PP_PATH'] = spec['site']['vasp_pp_path']
    event('qm_start', id=molecule_object.get_moleculeid(), structure=geometry(molecule_object))
    result = original('QM_task', dict(molecule_object=molecule_object, QM_config=QM_config,
        QM_scratch_dir=QM_scratch_dir, properties_list=properties_list))
    event('qm_end', id=result.get_moleculeid(), converged=bool(result.check_convergence()),
          structure=geometry(result), properties=result.get_results())
    return result


@python_app(executors=['alf_ML_executor'])
def train(ML_config, h5_dir, model_path, current_training_id, gpus_per_node,
          current_model_id=-1, remove_existing=False, h5_test_dir=None):
    from campaign_support import event
    identify()  # No Torch import in the parent of the production trainer pool.
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(); '
                    'assert torch.cuda.device_count() == 4'], check=True, timeout=60)
    inputs = h5_inventory(h5_dir)
    event('training_start', model_id=current_training_id, input=inputs)
    result = original('ML_task', dict(ML_config=ML_config, h5_dir=h5_dir, model_path=model_path,
        current_training_id=current_training_id, gpus_per_node=gpus_per_node,
        current_model_id=current_model_id, remove_existing=remove_existing, h5_test_dir=h5_test_dir))
    members = member_reports(model_path.format(current_training_id))
    atomic_json(Path(os.environ['ALF_CAMPAIGN_RUN'])/'training'/f'model-{current_training_id:04d}.json',
                dict(model_id=current_training_id, input=inputs, members=members, returned=result))
    event('training_end', model_id=current_training_id, members=members)
    return result


def prepare(run, family, site, source=SOURCE):
    """Copy example settings; change only bounded workload and site resources."""
    run, source = Path(run).resolve(), Path(source).resolve()
    example = source/'examples'/family
    master = read_json(example/'master_config.json')
    tasks = {role: master[role] for role in ('builder_task', 'sampler_task', 'QM_task', 'ML_task')}
    configs = {role: read_json(example/master[role+'_config_path']) for role in ('builder', 'sampler', 'QM', 'ML')}
    for key in ('molecule_library_path', 'molecule_library_dir'):
        if key in configs['builder']:
            configs['builder'][key] = str(example/configs['builder'][key])+'/'
    profile = read_json(HERE/'profile.json')
    # Compatibility smokes have a smaller bootstrap; matrix acceptance stays at 32.
    profile.update(bootstrap_structures=16, builder_width=master.get('maximum_builder_structures', 1))
    master.update(master_directory=str(run), status_path='status.json', h5_path='h5store/data-{:04d}.h5',
        model_path='models/model-{:04d}', parsl_configuration='campaign_resources.config',
        bootstrap_set=16, target_queued_QM=8, minimum_QM=0, save_h5_threshold=7,
        parallel_samplers=max(4, master.get('maximum_builder_structures', 1)),
        gpus_per_node=4, QM_scratch_dir='qm_scratch/',
        builder_task='compatibility.build_many' if master.get('maximum_builder_structures', 1)>1 else 'compatibility.build_single',
        sampler_task='compatibility.sample', QM_task='compatibility.label', ML_task='compatibility.train')
    for key in ('parsl_debug_configuration', 'plotting_utility', 'plot_dir', 'maximum_completed_QM'):
        master.pop(key, None)
    if 'maxt' in configs['sampler']:
        configs['sampler']['maxt'] = .1
    configs['ML']['n_models'] = 4
    qm = configs['QM']
    if family not in ('molten_salt', 'UO2'):
        qm.update(QM_run_command=site['orca'], ncpu=8)
        qm['orcablocks'] = re.sub(r'%maxcore\s+\d+', '%maxcore 2000', qm.get('orcablocks', ''))
    else:
        qm['QM_run_command'] = site.get('vasp_command', qm['QM_run_command'])
        inputs = qm.get('input_list', [qm.get('input', {})])
        for parameters in inputs:
            parameters.update(ncore=1, kpar=1)
    values = {}
    for role, config in configs.items():
        name = role.lower()+'_config.json'
        master[role+'_config_path'] = name
        values[name] = config
    values['master_config.json'] = master
    values['run.json'] = dict(case=asdict(Case('ase', 1, 1)), profile=profile, site=site,
        source=str(source), compatibility=dict(family=family, tasks=tasks))
    for name, value in values.items():
        path = run/name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f'Immutable compatibility configuration differs: {path}')
        else:
            atomic_json(path, value)
    return run


def availability(family, site):
    if family == 'UO2' and importlib.util.find_spec('anitraintools') is None:
        return 'NeuroChem training backend unavailable in the qualified environment'
    if family in ('molten_salt', 'UO2') and not all(site.get(k) for k in ('vasp_command', 'vasp_pp_path')):
        return 'VASP executable/runtime and pseudopotential location are not qualified in site settings'
    return None


def import_water_seed(seed, run, family):
    seed, run = Path(seed), Path(run)
    if family != 'seeded_active_learning':
        seed_case(seed, run)
        return
    # This example starts with labels only, so initial training must actually run.
    manifest = read_json(seed/'seed.json')
    files = {name: digest for name, digest in manifest['files_sha256'].items() if name.startswith('h5store/')}
    for name, digest in files.items():
        if not (seed/name).resolve().is_relative_to(seed.resolve()):
            raise ValueError('Invalid seed path')
        if fingerprint(seed/name) != digest:
            raise ValueError('Seed changed')
        target = run/name
        if target.exists():
            raise ValueError(f'Case output already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed/name, target)
    atomic_json(run/'seed.json', dict(ids=manifest['ids'], files_sha256=files,
        source_manifest_sha256=fingerprint(seed/'seed.json'), labels_only=True))


def verify_water_seed(seed, run):
    """Never mix the PBE campaign labels with the examples' wB97X labels."""
    seed, run = Path(seed), Path(run)
    bootstrap = seed.parent/'bootstrap'
    result = read_json(bootstrap/'result.json')
    members = result['accepted']['training']['members']
    if result['status'] != 'passed' or len(members) != 4 or not all(
            m['complete'] and m['epochs'] > 0 for m in members):
        raise ValueError('Compatibility seed needs four actually trained initial members')
    a, b = (read_json(p/'master_config.json') for p in (bootstrap, run))
    if a['properties_list'] != b['properties_list']:
        raise ValueError('Compatibility seed label properties or units differ')
    a, b = (read_json(p/master['QM_config_path']) for p, master in ((bootstrap, a), (run, b)))
    if a['orcasimpleinput'] != b['orcasimpleinput']:
        raise ValueError('Compatibility seed QM method differs; relabel and retrain a separate seed')
    a, b = (read_json(p/'ml_config.json') for p in (bootstrap, run))
    for key in ('network_params', 'energy_key', 'force_key', 'species_key', 'coordinates_key',
                'cell_key', 'hipnn_order', 'first_is_interacting'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Compatibility seed model architecture differs: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--families', choices=FAMILIES, nargs='+', default=list(FAMILIES))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.families)) != len(args.families):
        parser.error('Each example family must appear only once')
    from campaign_support import darwin_site
    site = dict(darwin_site(), **read_json(args.site))
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only and (root/'execution.json').exists():
        raise ValueError('Use a fresh run root; previous compatibility evidence is immutable')
    if not args.prepare_only:
        descriptor = os.open(root/'.execution.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    for family in args.families:
        prepare(root/family, family, site)
    if args.prepare_only:
        print(f'Prepared {len(args.families)} original-example families; no jobs submitted.')
        return
    atomic_json(root/'execution.json', dict(started=time.time(), source=source_identity()))
    from run import launch, case_metrics, freeze_seed
    seed = args.seed
    if seed is None and any(family in WATER for family in args.families):
        bootstrap = prepare(root/'bootstrap', 'simple_water', site)
        try:
            accepted = launch(bootstrap, 'coordinator', time.time()+3000, bootstrap=True)
        except Exception as exc:
            atomic_json(bootstrap/'result.json', dict(status='incomplete', error=repr(exc)))
            atomic_json(root/'result.json', dict(status='incomplete', bootstrap_error=repr(exc), cases={}))
            raise
        from ase import Atoms
        from ase.io import write
        frames = [frame for e in events(bootstrap) if e['kind']=='builder_end' for frame in e['structures']][:4]
        write(bootstrap/'preflight.extxyz', [Atoms(numbers=f['numbers'], positions=f['positions'],
              cell=f['cell'], pbc=f['pbc']) for f in frames])
        atomic_json(bootstrap/'result.json', dict(status='passed', accepted=accepted))
        freeze_seed(bootstrap, root/'seed', accepted)
        seed = root/'seed'
    outcomes = {}
    for family in args.families:
        run = root/family
        reason = availability(family, site)
        if family in WATER and seed is None:
            reason = 'A freshly trained compatible water seed is required'
        if reason:
            outcome = dict(status='unverified', reason=reason)
        else:
            started = time.time()
            try:
                if family in WATER:
                    verify_water_seed(seed, run)
                    import_water_seed(seed, run, family)
                accepted = launch(run, 'coordinator', started+3000)
                outcome = dict(status='passed', accepted=accepted)
            except Exception as exc:
                outcome = dict(status='incomplete', error=repr(exc))
                cleanup = run/'cleanup.json'
                if events(run) and (not cleanup.exists() or not read_json(cleanup).get('verified')):
                    raise RuntimeError(f'Cleanup unverified for {family}; refusing more jobs') from exc
            outcome.update(elapsed=time.time()-started, metrics=case_metrics(run))
        atomic_json(run/'result.json', outcome)
        outcomes[family] = outcome
        atomic_json(root/'result.json', dict(status='running', cases=outcomes))
    atomic_json(root/'result.json', dict(status='passed' if all(v['status']=='passed' for v in outcomes.values())
                                        else 'incomplete', cases=outcomes))
    if not all(v['status']=='passed' for v in outcomes.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
