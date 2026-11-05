#!/usr/bin/env python3
"""Opt-in Increment 8 labeling/storage qualification; no training or dynamics.

Preparation writes a frozen source snapshot and explicit site/fixture manifest.
Run --execute on a persistent Darwin front-end session; only its recorded Slurm
jobs are queried or cancelled. --worker is internal to each owned allocation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

BACKENDS = ('gpu4pyscf', 'pyseqm')
SOURCE = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def properties(nac=False, optical=False):
    result = {**{f'sE{i}': [f'state_{i}_energy', 'system', 1.] for i in range(4)},
              **{f'F{i}': [f'state_{i}_forces', 'atomic', 1.] for i in range(4)},
              'dE01': ['gap_01', 'system', 1.]}
    if nac:
        result['nacr'] = ['nacr', 'pair_atomic', 1.]
    if optical:
        for key in ('transition_dipoles', 'oscillator_strengths'):
            result[key] = [key, 'system', 1.]
    return result


def fixture_specs():
    import numpy as np
    from ase import Atoms
    rng = np.random.default_rng(20261105)
    base = Atoms('OCHH', positions=[[0, 0, 0], [1.21, 0, 0],
                                   [1.80, .93, 0], [1.80, -.93, 0]])
    specs = []
    for i in range(4):
        a = base.copy()
        a.positions += rng.normal(0, .005, a.positions.shape)
        specs.append(dict(id=f'formaldehyde-{i}', symbols=a.get_chemical_symbols(),
                          positions=a.positions.tolist()))
    specs.append(dict(id='ethene', symbols=list('CCHHHH'),
        positions=[[-.67, 0, 0], [.67, 0, 0], [-1.23, .93, 0], [-1.23, -.93, 0],
                   [1.23, .93, 0], [1.23, -.93, 0]]))
    return specs


def molecules(specs):
    import numpy as np
    from ase import Atoms
    from alframework.tools.molecules_class import MoleculesObject
    out = []
    for index, spec in enumerate(specs):
        atoms = Atoms(spec['symbols'], positions=spec['positions'])
        atoms.set_array('topology_atom_ids', np.arange(len(atoms)))
        out.append(MoleculesObject(atoms, spec['id']+f'-{index}'))
    return out


def assert_labels(items):
    if not all(m.check_convergence() for m in items):
        raise AssertionError([m.get_metadata() for m in items if not m.check_convergence()])


def compare(a, b, energy_tol, force_tol):
    import numpy as np
    errors = {'energy': 0., 'force': 0.}
    for x, y in zip(a, b, strict=True):
        for state in range(4):
            errors['energy'] = max(errors['energy'], abs(x.get_results()[f'sE{state}']-y.get_results()[f'sE{state}']))
            errors['force'] = max(errors['force'], float(np.max(np.abs(x.get_results()[f'F{state}']-y.get_results()[f'F{state}']))))
    if errors['energy'] > energy_tol or errors['force'] > force_tol:
        raise AssertionError(errors)
    return errors


def parsl_store(root, task, config, props, specs, mode):
    import parsl
    import h5py
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from alframework.tools.labeling import LabelingCoordinator
    from alframework.tools.tools import store_current_data
    config = dict(config)
    if mode:
        config['batching'] = dict(mode=mode, batch_size=4, partial_policy='timeout', max_wait_seconds=1.)
    run = root/('parsl-'+(mode or 'single'))
    run.mkdir()
    # This executor runs inside the owned Slurm allocation; there is no nested
    # scheduler allocation. It invokes the production QM_task and coordinator.
    dfk = parsl.load(Config(executors=[ThreadPoolExecutor(label='alf_QM_executor', max_threads=1)],
                            strategy='none', run_dir=str(run/'runinfo')))
    try:
        coordinator = LabelingCoordinator(task, task.func.__module__+'.'+task.func.__name__, config)
        inputs = [{'QM_config': config, 'properties_list': props, 'sampler_config': {}, 'gpus_per_node': None}]
        submitted = molecules(specs[:4])
        for m in submitted:
            coordinator.submit(m, inputs)
        coordinator.submit_ready(inputs, flush=True)
        tasks = list(coordinator.queue.task_list)
        widths = [coordinator.queue.widths.width(t) if mode else 1 for t in tasks]
        returned = []
        for task_future in tasks:
            value = task_future.result(timeout=1200)
            returned.extend(value if isinstance(value, list) else [value])
        assert_labels(returned)
        assert [m.get_moleculeid() for m in returned] == [m.get_moleculeid() for m in submitted]
        assert widths == ([4] if mode else [1]*4), widths
        store_current_data(str(run/'labels.h5'), returned, props)
        with h5py.File(run/'labels.h5') as h5:
            assert sum(len(g['_id']) for g in h5.values()) == 4
            for g in h5.values():
                if 'nacr' in props:
                    assert g['nac_pairs'][:].tolist() == [[1,2], [1,3], [2,3]]
                    assert g['nacr'].shape[1:] == (3, 4, 3)
        return dict(submitted_widths=widths, ids=[m.get_moleculeid() for m in returned],
                    metadata=[m.get_metadata() for m in returned], h5_sha256=digest(run/'labels.h5'))
    finally:
        dfk.cleanup()
        parsl.clear()


def finite_difference(label, specs, base, tolerance=1e-3):
    import copy
    step = 1e-4
    shifted = []
    for sign in (1, -1):
        spec = copy.deepcopy(specs[0])
        spec['positions'][0][0] += sign*step
        shifted.append(label([spec])[0])
    assert_labels(shifted)
    errors = []
    for state in range(4):
        fd = -(shifted[0].get_results()[f'sE{state}']-shifted[1].get_results()[f'sE{state}'])/(2*step)
        errors.append(abs(fd-base.get_results()[f'F{state}'][0, 0]))
    if max(errors) > tolerance:
        raise AssertionError({'finite_difference_errors': errors})
    return errors


def qualify_pyseqm(root, specs):
    from alframework.qm_interfaces.pyseqm_interface import (label_pyseqm_excited_batch,
        pyseqm_excited_state_task, pyseqm_excited_state_batch_task)
    props = properties()
    config = dict(method='AM1', device='cuda', scf_eps=1e-10, cis_tol=1e-8,
                  timeout_seconds=180, num_threads=4)
    write_json(root/'config.json', {'QM_config': config, 'properties_list': props})
    def label(items):
        result = label_pyseqm_excited_batch(molecules(items), config, props)
        assert_labels(result)
        return result
    singles = [label([s])[0] for s in specs]
    comparisons = {}
    for width in (1, 2, 4, 8):
        chosen = [specs[i % 4] for i in range(width)]
        batch = label(chosen)
        assert all(m.get_metadata()['qm_backend_width'] == width for m in batch)
        refs = [singles[specs.index(s)] for s in chosen]
        comparisons[str(width)] = compare(batch, refs, 1e-6, 1e-5)
    chosen = [specs[0], specs[-1], specs[1], specs[-1], specs[2], specs[-1], specs[3], specs[-1]]
    mixed = label(chosen)
    assert all(m.get_metadata()['qm_backend_width'] == 4 for m in mixed)
    assert mixed[0].get_metadata()['qm_solve_widths'] == [4, 4]
    mixed_errors = compare(mixed, [singles[specs.index(s)] for s in chosen], 1e-6, 1e-5)
    report = {'mixed_composition': {'errors': mixed_errors, 'solve_widths': [4,4]},
              'native_comparisons': comparisons,
              'finite_difference': finite_difference(label, specs, singles[0])}
    for mode in (None, 'sequential', 'native'):
        task = pyseqm_excited_state_batch_task if mode == 'native' else pyseqm_excited_state_task
        report['parsl_'+str(mode)] = parsl_store(root, task, config, props, specs, mode)
    # Updating PySEQM also gets a small ground-state compatibility check.
    from alframework.qm_interfaces.pyseqm_interface import label_pyseqm_batch
    ground = label_pyseqm_batch(molecules([specs[0], specs[-1]]), {'device': 'cuda', 'method': 'AM1'})
    assert_labels(ground)
    report['ground_state_check'] = True
    return report


def qualify_gpu4pyscf(root, specs):
    import numpy as np
    from alframework.qm_interfaces.gpu4pyscf_interface import (
        label_gpu4pyscf_excited_state_molecule, gpu4pyscf_excited_state_task)
    props = properties(nac=True, optical=True)
    config = dict(xc='cam-b3lyp', basis='6-31g', nroots=3, density_fit=True,
                  grids_level=3, scf_conv_tol=1e-10, tda_conv_tol=1e-8,
                  cphf_conv_tol=1e-10, num_threads=4, compute_nacr=True)
    write_json(root/'config.json', {'QM_config': config, 'properties_list': props})
    def label(items):
        result = []
        for m in molecules(items):
            labeled = label_gpu4pyscf_excited_state_molecule(m, QM_config=config, properties_list=props)
            assert_labels([labeled])
            result.append(labeled)
        return result
    gpu = label(specs[:4])
    from pyscf import gto, dft, lib
    from pyscf.data import nist
    lib.num_threads(4)
    mol = gto.M(atom=list(zip(specs[0]['symbols'], specs[0]['positions'])), basis=config['basis'], verbose=0)
    mf = dft.RKS(mol, xc=config['xc']).density_fit()
    mf.grids.level = config['grids_level']; mf.conv_tol = config['scf_conv_tol']
    mf.kernel(); assert mf.converged
    td = mf.TDA(); td.nstates=3; td.conv_tol=config['tda_conv_tol']; td.kernel()
    assert np.all(td.converged)
    energies = (mf.e_tot + np.r_[0., td.e])*nist.HARTREE2EV
    gradients = [mf.nuc_grad_method().kernel()]
    grad = td.nuc_grad_method(); grad.cphf_conv_tol=config['cphf_conv_tol']
    gradients.extend(grad.kernel(state=i) for i in range(1,4))
    force = -np.asarray(gradients)*nist.HARTREE2EV/nist.BOHR
    errors = {'energy': max(abs(gpu[0].get_results()[f'sE{i}']-energies[i]) for i in range(4)),
              'force': max(float(np.max(np.abs(gpu[0].get_results()[f'F{i}']-force[i]))) for i in range(4))}
    assert errors['energy'] <= 1e-5 and errors['force'] <= 1e-4, errors
    for m in gpu:
        assert m.get_metadata()['nac_pairs'] == [[1,2], [1,3], [2,3]]
        assert np.isfinite(m.get_results()['nacr']).all()
        assert np.isfinite(m.get_results()['transition_dipoles']).all()
        assert np.all(m.get_results()['oscillator_strengths'] >= 0)
    report = {'cpu_gpu_errors': errors, 'finite_difference': finite_difference(label, specs, gpu[0])}
    for mode in (None, 'sequential'):
        report['parsl_'+str(mode)] = parsl_store(root, gpu4pyscf_excited_state_task, config, props, specs, mode)
    return report


def worker(root, backend):
    from importlib.metadata import distributions
    manifest = json.loads((root/'manifest.json').read_text())
    for rel, expected in manifest['source_sha256'].items():
        if digest(root/'source'/rel) != expected:
            raise RuntimeError(f'Frozen source changed: {rel}')
    out = root/backend
    out.mkdir()
    actual_packages = {d.metadata['Name']: d.version for d in distributions()}
    expected = manifest.get('environments', {}).get(backend)
    if expected and actual_packages != expected['packages']:
        raise RuntimeError('Frozen dependency versions changed')
    import alframework
    if not Path(alframework.__file__).resolve().is_relative_to(root/'source'):
        raise RuntimeError('Worker imported ALF outside frozen source')
    write_json(out/'environment.json', {'python': sys.executable,
        'packages': {d.metadata['Name']: d.version for d in distributions()},
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': os.uname().nodename})
    result = {'status': 'incomplete'}
    try:
        result.update((qualify_gpu4pyscf if backend=='gpu4pyscf' else qualify_pyseqm)(out, manifest['fixtures']))
        result['status'] = 'passed'
    except Exception as exc:
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        write_json(out/'result.json', result)


def prepare(root, site, backends):
    import tarfile
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'; source.mkdir()
    # Export exactly the committed tracked source; exclude user untracked files.
    archive = subprocess.check_output(['git', '-C', str(SOURCE), 'archive', 'HEAD'])
    import io
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter='data')
    manifest = {'revision': subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip(),
                'site': site, 'backends': backends, 'fixtures': fixture_specs(),
                'environments': {b: json.loads(Path(site[b+'_environment']).read_text())
                                 for b in backends if b+'_environment' in site},
                'source_sha256': {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}}
    write_json(root/'manifest.json', manifest)
    for backend in backends:
        python = site[backend+'_python']
        command = [python, str(source/'examples/excited_state_labeling/qualify.py'),
                   '--root', str(root), '--worker', backend]
        script = f'''#!/bin/bash -l
#SBATCH --job-name=alf-inc8-{backend}
#SBATCH --partition={site.get('partition','shared-redstone')}
#SBATCH --account={site.get('account','y2020-bf')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output={root}/{backend}-slurm.log
{('#SBATCH --dependency=afterany:'+str(site['after_job'])) if site.get('after_job') else ''}
set -euo pipefail
module load {shlex.quote(site.get('cuda_module','cuda/13.2.1'))}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CXX=/projects/opt/centos8/x86_64/gcc/11.2.0/bin/g++
export CC=/projects/opt/centos8/x86_64/gcc/11.2.0/bin/gcc
export PYTHONPATH={shlex.quote(str(source))}
exec {shlex.join(command)}
'''
        (root/(backend+'.sbatch')).write_text(script)
    return manifest


def cleanup_job(job, run=subprocess.run, deadline_seconds=30):
    """The caller supplies one numeric job ID from its own submission journal."""
    if not str(job).isdigit():
        raise ValueError('Invalid owned Slurm job ID')
    run(['scancel', str(job)], check=False, capture_output=True, text=True)
    end = time.monotonic()+deadline_seconds
    while True:
        state = run(['squeue','-h','-j',str(job),'-o','%i'], check=True, capture_output=True, text=True).stdout.strip()
        if not state:
            return {'job_id': str(job), 'verified_terminated': True}
        if time.monotonic() >= end:
            raise RuntimeError(f'Owned job {job} did not terminate')
        time.sleep(1)


def execute(root, budget_seconds=7200):
    manifest = json.loads((root/'manifest.json').read_text())
    deadline = time.monotonic()+budget_seconds
    result = {'status': 'running', 'stages': {}, 'owned_jobs': []}
    write_json(root/'result.json', result)
    try:
        for backend in manifest['backends']:
            if time.monotonic() >= deadline:
                result['stages'][backend] = {'status':'incomplete','reason':'overall deadline'}
                continue
            job = subprocess.check_output(['sbatch','--parsable',str(root/(backend+'.sbatch'))], text=True).strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError(f'Invalid submitted Slurm ID: {job}')
            result['owned_jobs'].append(job)
            write_json(root/'result.json', result)
            try:
                while time.monotonic() < deadline:
                    state = subprocess.check_output(['squeue','-h','-j',job,'-o','%i'], text=True).strip()
                    if not state:
                        break
                    time.sleep(5)
            finally:
                cleanup = cleanup_job(job)
                write_json(root/(backend+'-cleanup.json'), cleanup)
            path = root/backend/'result.json'
            result['stages'][backend] = json.loads(path.read_text()) if path.exists() else {'status':'incomplete','reason':'no worker result before job end/deadline'}
            write_json(root/'result.json', result)
        result['status'] = 'passed' if all(x['status']=='passed' for x in result['stages'].values()) else 'incomplete'
    except BaseException as exc:
        result.update(status='incomplete', error=repr(exc))
        raise
    finally:
        write_json(root/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--site', type=Path)
    p.add_argument('--deadline-epoch', type=float, help='Reuse an earlier overall deadline for retries')
    p.add_argument('--backend', nargs='+', choices=BACKENDS, default=list(BACKENDS))
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--worker', choices=BACKENDS)
    args = p.parse_args()
    root = args.root.resolve()
    if args.prepare_only:
        if not args.site: p.error('--site is required for preparation')
        prepare(root, json.loads(args.site.read_text()), args.backend)
    elif args.execute:
        result = execute(root, budget_seconds=7200 if args.deadline_epoch is None
                         else max(0., args.deadline_epoch-time.time()))
        raise SystemExit(0 if result['status'] == 'passed' else 1)
    else:
        worker(root, args.worker)


if __name__ == '__main__':
    main()
