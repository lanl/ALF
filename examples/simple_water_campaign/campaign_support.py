"""Configuration, provenance, and acceptance for the opt-in water campaign.

This module does not import Torch, HIPPYNN, or Parsl. In particular, importing
the task adapters must not initialize CUDA before the legacy training pool forks.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


@dataclass(frozen=True)
class Case:
    engine: str
    width: int
    qm_width: int

    @property
    def name(self):
        return f'{self.engine}-w{self.width}-q{self.qm_width}'


def matrix():
    return [Case(engine, width, qm) for engine in ('ase', 'torchsim', 'alchemi')
            for width, qm in ((1, 1), (4, 4), (4, 1), (1, 4))]


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + '\n')
    temporary.replace(path)


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def event(kind, run=None, **values):
    run = Path(run or os.environ['ALF_CAMPAIGN_RUN'])
    record = dict(kind=kind, time=time.time(), pid=os.getpid(), **values)
    atomic_json(run / 'events' / f'{time.time_ns()}-{uuid.uuid4().hex}.json', record)
    return record


def events(run):
    return [read_json(path) for path in sorted((Path(run) / 'events').glob('*.json'))]


def source_identity(source=SOURCE):
    source = Path(source).resolve()
    names = sorted((source / 'alframework').rglob('*.py'))
    names += sorted((source / 'examples/simple_water_campaign').glob('*.py'))
    names += [source / 'examples/simple_water_campaign/profile.json']
    hashes = {str(path.relative_to(source)): fingerprint(path) for path in names}
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain',
                                         '--untracked-files=no'], text=True).strip())
    return dict(root=str(source), revision=revision, dirty=dirty, files_sha256=hashes,
                tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def worker_identity():
    import importlib.util
    import socket
    import sys
    import alframework
    packages = {}
    for name in ('torch', 'hippynn', 'parsl', 'ase', 'numpy', 'torch-sim-atomistic',
                 'nvalchemi-toolkit', 'nvalchemi-toolkit-ops', 'warp-lang'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hippynn = importlib.util.find_spec('hippynn')
    dependency = None
    if hippynn and hippynn.origin:
        package = Path(hippynn.origin).resolve().parent
        hashes = {str(p.relative_to(package)): fingerprint(p) for p in sorted(package.rglob('*.py'))}
        dependency = dict(root=str(package), tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    return dict(host=socket.gethostname(), python=sys.executable, alf=str(Path(alframework.__file__).resolve()),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                hippynn_source=dependency,
                packages=packages, masks={key: os.environ.get(key) for key in
                ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')})


def darwin_site():
    home = Path.home()
    return dict(cpu_partition='general', gpu_partition='shared-redstone', account='y2020-bf',
                base_python=str(home / '.conda/envs/atomistic/bin/python'),
                torchsim_python=str(home / 'ALF_LANL/ALF_fork/ALF/debug/'
                                    'portable_gpu_20261027_8795a40/envs/torchsim-cuda/bin/python'),
                alchemi_python=str(home / '.conda/envs/alf_alchemi_02/bin/python'),
                orca='/usr/projects/ml4chem/Programs/orca_5.0.4/orca',
                mpi_module='openmpi/4.1.1')


def prepare_run(directory, case, profile, site, source=SOURCE):
    """Create configs exactly once; repeated preparation verifies immutability."""
    directory, source = Path(directory).resolve(), Path(source).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    example = source / 'examples/simple_water'
    master = read_json(example / 'master_config.json')
    master.update(master_directory=str(directory), builder_task='campaign_tasks.build_water',
                  maximum_builder_structures=4, sampler_task='campaign_tasks.sample_single',
                  QM_task='campaign_tasks.label_water', ML_task='campaign_tasks.train_water',
                  status_path='status.json', target_queued_QM=8, minimum_QM=0, save_h5_threshold=7,
                  parallel_samplers=4, bootstrap_set=profile['bootstrap_structures'], gpus_per_node=4,
                  parsl_configuration='campaign_resources.config', QM_scratch_dir='orca_scratch/')
    for key in ('parsl_debug_configuration', 'plotting_utility', 'plot_dir', 'maximum_completed_QM'):
        master.pop(key, None)
    builder = read_json(example / 'builder_config.json')
    builder['molecule_library_path'] = str(example / 'fragment_library') + '/'
    sampler = read_json(example / 'mlmd_config.json')
    sampler['maxt'] = profile['trajectory_ps']
    sampler['device'] = 'cuda'
    sampler['meta_dir'] = 'sampling/'
    qm = read_json(example / 'orca_config.json')
    qm.update(QM_run_command=site['orca'], ncpu=profile['orca_cores'], orcasimpleinput=profile['orca_input'])
    qm['orcablocks'] = qm['orcablocks'].replace('%maxcore 15000', f"%maxcore {profile['orca_maxcore_mb']}")
    ml = read_json(example / 'hippynn_config.json')
    ml['n_models'] = profile['ensemble_members']
    if case is not None:
        if case.engine != 'ase' or case.width == 4:
            master['sampler_task'] = 'campaign_tasks.sample_many'
            sampler['driver'] = dict(engine=case.engine, strategy='single' if case.width == 1 else 'batched')
            sampler['random_seed'] = 20261102
            if case.width == 4:
                sampler['batching'] = dict(batch_size=4, partial_policy='timeout',
                                           max_wait_seconds=profile['batch_wait_seconds'])
            if case.engine != 'ase':
                sampler.pop('ase_calculator')
                sampler['batched_calculator'] = ('alframework.ml_interfaces.hippynn_interface.'
                                                  'load_hippynn_batched_ensemble')
        if case.qm_width == 4:
            qm['batching'] = dict(mode='sequential', batch_size=4, partial_policy='timeout',
                                  max_wait_seconds=profile['batch_wait_seconds'])
    values = {'master_config.json': master, 'builder_config.json': builder, 'mlmd_config.json': sampler,
              'orca_config.json': qm, 'hippynn_config.json': ml,
              'run.json': dict(case=None if case is None else asdict(case), profile=profile,
                               site=site, source=str(source))}
    for name, value in values.items():
        path = directory / name
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f'Immutable configuration differs: {path}')
        else:
            atomic_json(path, value)
    return directory


def h5_inventory(directory):
    import h5py
    ids, files = [], {}
    for path in sorted(Path(directory).glob('data-*.h5')):
        with h5py.File(path, 'r') as handle:
            for group in handle.values():
                ids.extend(value.decode() if isinstance(value, bytes) else str(value) for value in group['_id'][()])
        files[str(path.resolve())] = fingerprint(path)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate structure IDs in the HDF5 store')
    return dict(ids=ids, files_sha256=files)


def member_reports(directory):
    reports = []
    members = sorted(Path(directory).glob('model-*'))
    for member in members:
        log = (member / 'training_log.txt').read_text()
        hashes = {name: fingerprint(member / name) for name in
                  ('experiment_structure.pt', 'best_checkpoint.pt', 'best_model.pt')}
        counts = {}
        for label, text in [('retained', log.split('Array Shapes After Cleaning')[-1]),
                            *[(name, log.split(f'Arrays for split: {name}')[-1]) for name in ('train', 'valid', 'test')]]:
            match = re.search(r'\|\s*coordinates\s*\|[^\n]*?\|\s*\((\d+),', text)
            counts[label] = int(match[1]) if match else None
        reports.append(dict(member=member.name, complete='Training complete' in log,
                            epochs=len(re.findall(r'Epoch\s+\d+\s*:', log)), counts=counts,
                            files_sha256=hashes))
    return reports


def validate_storage(run):
    """Verify observed labels against storage's units, atom sorting, and cells."""
    import h5py
    import numpy as np
    from ase.data import chemical_symbols
    run = Path(run)
    master = read_json(run / 'master_config.json')
    labels = {rec['id']: rec for rec in events(run) if rec['kind'] == 'qm_end' and rec['converged']}
    checked = 0
    for path in sorted((run / 'h5store').glob('data-*.h5')):
        with h5py.File(path, 'r') as handle:
            for group in handle.values():
                for index, raw_id in enumerate(group['_id'][()]):
                    name = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    if name not in labels:
                        if name.startswith('mol-boot-') and (run / 'seed.json').exists():
                            continue
                        raise ValueError(f'No converged QM evidence for stored frame {name}')
                    record = labels[name]
                    structure = record['structure']
                    order = np.argsort(structure['numbers'])
                    assert all(structure['pbc'])
                    expected_species = [chemical_symbols[n] for n in np.array(structure['numbers'])[order]]
                    assert [v.decode() if isinstance(v, bytes) else str(v) for v in group['species'][()]] == expected_species
                    np.testing.assert_allclose(group['coordinates'][index], np.array(structure['positions'])[order], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(group['cell'][index], structure['cell'], atol=1e-12, rtol=1e-12)
                    for prop, (key, kind, factor) in master['properties_list'].items():
                        expected = np.asarray(record['properties'][prop])
                        if kind == 'atomic':
                            expected = expected[order]
                        actual = group[key][index]
                        assert np.isfinite(actual).all()
                        np.testing.assert_allclose(actual, expected*factor, atol=1e-12, rtol=1e-12)
                    checked += 1
    return checked


def submitted_width(func, app_args, app_kwargs):
    """Read the actual submission, including the coordinator's packed QM app."""
    import inspect
    bound = inspect.signature(func).bind_partial(*app_args, **app_kwargs).arguments
    molecules = bound.get('molecule_objects')
    if molecules is not None:
        return len(molecules), [mol.get_moleculeid() for mol in molecules]
    if 'moleculeids' in bound:
        return len(bound['moleculeids']), list(bound['moleculeids'])
    molecule = bound.get('molecule_object')
    return (1, [molecule.get_moleculeid()]) if molecule is not None else (None, [])


def completion(run, bootstrap=False):
    """Require real sampled/QM/stored/training lineage, not a model counter."""
    run = Path(run)
    try:
        config = read_json(run / 'run.json')
        records = events(run)
        stored = set(h5_inventory(run / 'h5store')['ids'])
        candidates = {name for rec in records if rec['kind'] == 'sampler_end' for name in rec['selected_ids']}
        labels = {rec['id'] for rec in records if rec['kind'] == 'qm_end' and rec['converged']}
        built = {name for rec in records if rec['kind'] == 'builder_end' for name in rec['ids']}
        status = read_json(run / 'status.json') if not bootstrap else None
        for path in sorted((run / 'training').glob('model-*.json')):
            training = read_json(path)
            members = training['members']
            if len(members) != 4 or not all(m['complete'] and m['epochs'] > 0 and
                    all((m['counts'].get(k) or 0) > 0 for k in ('retained', 'train', 'valid', 'test')) for m in members):
                continue
            inputs = set(training['input']['ids'])
            if bootstrap:
                good = {name for name in inputs & stored & labels & built if name.startswith('mol-boot-')}
                if training['model_id'] != 0 or len(good) < config['profile']['bootstrap_structures']:
                    continue
            else:
                good = {name for name in inputs & stored & labels & candidates & built
                        if not name.startswith('mol-boot-')}
                if training['model_id'] < 1 or status['current_model_id'] < training['model_id']:
                    continue
                if len(good) < config['profile']['new_labels_per_case']:
                    continue
                case = Case(**config['case'])
                if case.width == 4 and not any(r['kind'] == 'sampler_end' and r['width'] == 4 for r in records):
                    continue
                if case.qm_width == 4 and not any(r['kind'] == 'task_done' and r['name'] == 'label_sequentially'
                                                 and r['width'] == 4 and r['success'] for r in records):
                    continue
            return dict(model_id=training['model_id'], qualifying_ids=sorted(good), training=training)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def seed_case(seed, run):
    """Copy, never hard-link, a verified immutable seed; no status is inherited."""
    seed, run = Path(seed), Path(run)
    manifest = read_json(seed / 'seed.json')
    if (run / 'seed.json').exists():
        raise ValueError(f'Case already seeded: {run}')
    for relative, expected in manifest['files_sha256'].items():
        original = seed / relative
        if fingerprint(original) != expected:
            raise ValueError(f'Seed changed: {original}')
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f'Case output already exists: {target}')
        shutil.copy2(original, target)
    atomic_json(run / 'seed.json', manifest)


def owned_jobs(run):
    result = set()
    for rec in events(run):
        if rec['kind'] == 'job_submitted':
            job = str(rec['job_id'])
            if not re.fullmatch(r'[0-9]+', job):
                raise ValueError('Invalid owned Slurm job ID')
            result.add(job)
    return sorted(result)


def cleanup_jobs(run, execute=subprocess.run, timeout=90):
    """Only cancel IDs recorded by this run's provider; verify disappearance."""
    jobs = owned_jobs(run)
    if not jobs:
        return []
    execute(['scancel', *jobs], capture_output=True, text=True, timeout=30, check=False)
    deadline = time.monotonic() + timeout
    while True:
        import getpass
        result = execute(['squeue', '--noheader', '--user=' + getpass.getuser(), '--format=%A'],
                         capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(f'Cannot verify job cleanup: {result.stderr}')
        remaining = sorted(set(result.stdout.split()) & set(jobs))
        if not remaining:
            return jobs
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Campaign jobs still active: {remaining}')
        time.sleep(2)
