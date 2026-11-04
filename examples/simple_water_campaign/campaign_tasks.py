"""Observation-only adapters around the existing production task functions."""
from pathlib import Path
import os
import time

from parsl import python_app

from campaign_support import atomic_json, check_campaign_gpu, event, h5_inventory, member_reports, read_json, worker_identity

_identified = False


def alchemi_backend(function, args):
    """Transfer data-only protocol-4 pickles across the qualified Python versions."""
    if os.environ.get('ALF_CAMPAIGN_BACKEND') == '1':
        return False, None
    run = Path(os.environ['ALF_CAMPAIGN_RUN'])
    spec = read_json(run / 'run.json')
    if (spec['case'] or {}).get('engine') != 'alchemi':
        return False, None
    import pickle
    import subprocess
    import tempfile
    directory = run / 'backend'
    directory.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=function + '-', dir=directory))
    with (directory / 'input.pkl').open('wb') as handle:
        pickle.dump(dict(function=function, args=args), handle, protocol=4)
    python = spec['site']['alchemi_python']
    environment = dict(os.environ, ALF_CAMPAIGN_BACKEND='1',
                       PATH=str(Path(python).parent) + ':' + os.environ.get('PATH', ''))
    with (directory / 'backend.log').open('w') as log:
        result = subprocess.run([python, '-B', str(Path(__file__).with_name('backend.py')), str(directory)],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'ALCHEMI backend failed; see {directory / "backend.log"}')
    with (directory / 'output.pkl').open('rb') as handle:
        return True, pickle.load(handle)


def identify():
    global _identified
    if not _identified:
        identity = worker_identity()
        expected = Path(read_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'run.json')['source']).resolve()
        assert Path(identity['alf']).parent.parent == expected, 'Worker imported a different ALF checkout'
        event('worker', **identity)
        _identified = True


def unwrap(value):
    from parsl.app.errors import RemoteExceptionWrapper
    if isinstance(value, RemoteExceptionWrapper):
        value.reraise()
    return value


def geometry(molecule):
    atoms = molecule.get_atoms()
    return dict(id=molecule.get_moleculeid(), numbers=atoms.numbers, positions=atoms.positions,
                cell=atoms.cell.array, pbc=atoms.pbc)


def gpu_identity(device):
    import torch
    return dict(device=str(device), hardware=torch.cuda.get_device_name(device),
                torch=str(torch.__version__), cuda=torch.version.cuda, hip=torch.version.hip,
                visible_devices=torch.cuda.device_count())


def start_gpu_memory_measurement(device):
    """Allocator-stat APIs do not initialize CUDA in a fresh sampling worker."""
    import torch
    with torch.cuda.device(device):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


@python_app(executors=['alf_sampler_executor'])
def build_water(moleculeids, builder_config, molecule_library_dir, cell_range, solute_molecule_options, Rrange):
    from alframework.builders.builders import simple_multi_condensed_phase_builder_task
    identify()
    event('builder_start', ids=moleculeids, width=len(moleculeids))
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'run.json')
    fixture = None
    if (spec.get('case') or {}).get('fault') == 'trajectory_failure':
        from campaign_evidence import load_failure_fixture
        fixture = load_failure_fixture(os.environ['ALF_CAMPAIGN_RUN'])
    if fixture and any(name in fixture['structures'] for name in moleculeids):
        from ase import Atoms
        from alframework.tools.molecules_class import MoleculesObject
        if not all(name in fixture['structures'] for name in moleculeids):
            raise ValueError('Recorded builder batch must preserve all requested IDs')
        result = []
        for name in moleculeids:
            frame = fixture['structures'][name]
            atoms = Atoms(numbers=frame['numbers'], positions=frame['positions'],
                          cell=frame['cell'], pbc=frame['pbc'])
            result.append(MoleculesObject(atoms, name))
        event('builder_fixture_replayed', ids=moleculeids, source_run=fixture['source_run'],
              fixture_sha256=spec['profile']['trajectory_failure_fixture_sha256'])
    else:
        result = unwrap(simple_multi_condensed_phase_builder_task.func(
            moleculeids, builder_config, molecule_library_dir, cell_range, solute_molecule_options, Rrange))
    assert [m.get_moleculeid() for m in result] == list(moleculeids)
    assert all(m.get_atoms().pbc.all() and m.get_atoms().cell.volume > 0 for m in result)
    event('builder_end', ids=moleculeids, width=len(result), structures=[geometry(m) for m in result])
    return result


@python_app(executors=['alf_sampler_executor'])
def build_molecules(moleculeids, builder_config):
    import hashlib
    import numpy as np
    from ase import Atoms
    from alframework.samplers.geometry import minimum_distance
    from alframework.tools.molecules_class import MoleculesObject
    identify()
    result = []
    for name in moleculeids:
        number = int(name.rsplit('-', 1)[-1])
        atoms = (Atoms('HOH', positions=[[.96, 0, 0], [0, 0, 0], [-.24, .93, 0]]) if number % 2 == 0 else
                 Atoms('HCOHHH', positions=[[-.36, 1.03, 0], [0, 0, 0], [1.43, 0, 0],
                                           [-.36, -.51, .89], [-.36, -.51, -.89], [1.75, .9, 0]]))
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
            f"{builder_config['random_seed']}:{name}".encode()).digest()[:8], 'big'))
        base = atoms.positions.copy()
        for _ in range(100):
            atoms.positions = base + rng.normal(0, builder_config['perturbation_angstrom'], base.shape)
            if minimum_distance(atoms) >= builder_config['distcut']:
                break
        else:
            raise ValueError(f'Could not build acceptable molecular geometry: {name}')
        # A nonperiodic bookkeeping cell avoids singular-cell storage normalization.
        atoms.set_cell([20, 20, 20])
        result.append(MoleculesObject(atoms, name))
    event('builder_end', ids=moleculeids, width=len(result), structures=[geometry(m) for m in result])
    return result


def sampler_result(result, width, model_id, start):
    molecules = result if isinstance(result, list) else [result]
    selected = [m for m in molecules if m.get_atoms() is not None]
    event('sampler_end', width=width, model_id=model_id, elapsed=time.time()-start,
          ids=[m.get_moleculeid() for m in molecules], selected_ids=[m.get_moleculeid() for m in selected],
          selected=[geometry(m) for m in selected],
          metadata=[dict(id=m.get_moleculeid(), metadata=m.get_metadata()) for m in molecules])
    return result


@python_app(executors=['alf_sampler_executor'])
def sample_single(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
    from alframework.samplers.mlmd_sampling import simple_mlmd_sampling_task
    identify()
    start = time.time()
    event('sampler_start', ids=[molecule_object.get_moleculeid()], width=1, model_id=current_model_id,
          structures=[geometry(molecule_object)])
    result = unwrap(simple_mlmd_sampling_task.func(molecule_object, sampler_config, model_path,
                                                  current_model_id, gpus_per_node))
    return sampler_result(result, 1, current_model_id, start)


@python_app(executors=['alf_sampler_executor'])
def sample_many(molecule_objects, sampler_config, model_path, current_model_id, gpus_per_node):
    delegated, result = alchemi_backend('sample_many',
        (molecule_objects, sampler_config, model_path, current_model_id, gpus_per_node))
    if delegated:
        return result
    from alframework.samplers.batched_mlmd import mlmd_trajectories_task
    identify()
    start = time.time()
    event('sampler_start', ids=[m.get_moleculeid() for m in molecule_objects], width=len(molecule_objects),
          model_id=current_model_id, structures=[geometry(m) for m in molecule_objects])
    from campaign_evidence import sampling_audit
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'run.json')
    measured = spec['profile'].get('campaign', 'core') != 'core'
    if measured:
        import torch
        from alframework.tools.gpu import sampling_device
        device = sampling_device(sampler_config, gpus_per_node)
        start_gpu_memory_measurement(device)
    with sampling_audit(molecule_objects, (spec.get('case') or {}).get('fault', ''), sampler_config):
        result = unwrap(mlmd_trajectories_task.func(molecule_objects, sampler_config, model_path,
                                                   current_model_id, gpus_per_node))
    if measured:
        from campaign_evidence import memory
        event('sampler_memory', peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),
              gpu=gpu_identity(device), **memory())
    return sampler_result(result, len(molecule_objects), current_model_id, start)


@python_app(executors=['alf_QM_executor'])
def label_water(molecule_object, QM_config, QM_scratch_dir, properties_list):
    import numpy as np
    from alframework.qm_interfaces.orca5_interface import orca_calculator_task
    # HTEX affinity sets OMP threads to its block size. Each ORCA MPI rank
    # uses one thread, with eight ranks inside this worker's inherited block.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    identify()
    event('qm_start', id=molecule_object.get_moleculeid(), structure=geometry(molecule_object))
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'run.json')
    fault = (spec.get('case') or {}).get('fault', '')
    inject = molecule_object.get_moleculeid() == 'mol-0000-0000000000'
    if fault == 'label_exception' and inject:
        event('fault_injected', id=molecule_object.get_moleculeid(), fault=fault)
        event('qm_end', id=molecule_object.get_moleculeid(), converged=False,
              structure=geometry(molecule_object), properties={})
        raise RuntimeError('controlled campaign labeling exception')
    result = unwrap(orca_calculator_task.func(molecule_object, QM_config, QM_scratch_dir, properties_list))
    converged = bool(result.check_convergence())
    if fault == 'label_unconverged' and inject:
        event('fault_injected', id=result.get_moleculeid(), fault=fault, backend_converged=converged)
        result.set_converged_flag(False)
        result.update_metadata(dict(labeling_error='controlled campaign unconverged flag'))
        converged = False
    if converged:
        assert all(np.isfinite(value).all() for value in result.get_results().values())
    event('qm_end', id=result.get_moleculeid(), converged=converged, structure=geometry(result),
          properties=result.get_results())
    return result


@python_app(executors=['alf_sampler_executor'])
def label_molecules(molecule_objects, QM_config):
    """Real native GPU solves on the serialized sampling/labeling executor."""
    from alframework.qm_interfaces.pyseqm_interface import label_pyseqm_batch, run_pyseqm
    import torch
    import seqm
    identify()
    assert torch.cuda.is_available(), 'Native GPU acceptance requires CUDA'
    check_campaign_gpu('pyseqm', gpu_identity(QM_config['device']))
    torch.cuda.reset_peak_memory_stats()
    for molecule in molecule_objects:
        event('qm_start', id=molecule.get_moleculeid(), structure=geometry(molecule))

    def runner(*args, **kwargs):
        started = time.time()
        try:
            values = run_pyseqm(*args, **kwargs)
        except Exception:
            event('native_solve', width=len(args[0]), success=False, elapsed=time.time()-started)
            raise
        event('native_solve', width=len(args[0]), success=True, elapsed=time.time()-started,
              device=kwargs['device'], pyseqm_source=str(Path(seqm.__file__).resolve()),
              gpu=gpu_identity(kwargs['device']),
              peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
        return values

    result = label_pyseqm_batch(molecule_objects, QM_config, runner=runner)
    for molecule in result:
        event('qm_end', id=molecule.get_moleculeid(), converged=bool(molecule.check_convergence()),
              structure=geometry(molecule), properties=molecule.get_results())
    return result


@python_app(executors=['alf_ML_executor'])
def train_water(ML_config, h5_dir, model_path, current_training_id, gpus_per_node,
                remove_existing=False, h5_test_dir=None):
    import subprocess
    import sys
    from alframework.ml_interfaces.hippynn_interface import train_HIPPYNN_ensemble_task
    # Identify without importing Torch: the original trainer forks its GPU pool.
    identify()
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(); '
                    'assert torch.cuda.device_count() == 4'], check=True, timeout=60)
    inputs = h5_inventory(h5_dir)
    event('training_start', model_id=current_training_id, input=inputs)
    result = unwrap(train_HIPPYNN_ensemble_task.func(ML_config, h5_dir, model_path, current_training_id,
                    gpus_per_node, remove_existing=remove_existing, h5_test_dir=h5_test_dir))
    members = member_reports(model_path.format(current_training_id))
    report = dict(model_id=current_training_id, input=inputs, members=members, returned=result)
    atomic_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'training' / f'model-{current_training_id:04d}.json', report)
    event('training_end', model_id=current_training_id, members=members)
    return result


@python_app(executors=['alf_sampler_executor'])
def prediction_check(model_directory, structures_file):
    delegated, result = alchemi_backend('prediction_check', (model_directory, structures_file))
    if delegated:
        return result
    import numpy as np
    import torch
    from ase.io import read
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_calculator, load_hippynn_batched_ensemble
    from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
    from alframework.tools.gpu import sampling_device
    identify()
    device = sampling_device({'device': 'cuda'}, 4)
    spec = read_json(Path(os.environ['ALF_CAMPAIGN_RUN']) / 'run.json')
    gpu = gpu_identity(device)
    check_campaign_gpu(spec['profile'].get('campaign', 'core'), gpu)
    atoms = read(structures_file, index=':')
    reference = ASEEnsembleEvaluator([HIPNN_ASE_calculator(str(p), device=device)
        for p in sorted(Path(model_directory).glob('model-*'))]).predict(atoms)
    packed = load_hippynn_batched_ensemble(model_directory, device=device)
    assert packed.n_models == 4

    def evaluate(group):
        energies, forces = packed.evaluate(
            torch.as_tensor(np.concatenate([a.numbers for a in group]), device=device),
            torch.as_tensor(np.concatenate([a.positions for a in group]), device=device),
            torch.as_tensor(np.stack([a.cell.array for a in group]), device=device),
            group[0].pbc, [len(a) for a in group])
        energies, forces = energies.detach().cpu().numpy(), forces.detach().cpu().numpy()
        cuts = np.cumsum([0] + [len(a) for a in group])
        return dict(energies=energies, forces=[forces[:, lo:hi] for lo, hi in zip(cuts[:-1], cuts[1:])])

    rows = []
    for width in (1, 4):
        predictions = [evaluate(atoms[lo:lo+width]) for lo in range(0, len(atoms), width)]
        actual = dict(energies=np.concatenate([p['energies'] for p in predictions], axis=1),
                      forces=[f for p in predictions for f in p['forces']])
        np.testing.assert_allclose(actual['energies'], reference['energies'], rtol=1e-5, atol=1e-5)
        signals, expected = ensemble_statistics(actual), ensemble_statistics(reference)
        errors = []
        for index, forces in enumerate(actual['forces']):
            np.testing.assert_allclose(forces, reference['forces'][index], rtol=1e-5, atol=1e-5)
            delta_e = actual['energies'][:, index].astype(float) - reference['energies'][:, index]
            delta_f = forces.astype(float) - reference['forces'][index]
            component_bound = np.std(delta_f, axis=0)
            bounds = dict(Es=float(np.std(delta_e)), Fs=float(component_bound.mean()),
                          Fsmax=float(component_bound.max()), Fsrms=float(np.sqrt((component_bound**2).mean())))
            assert all(abs(signals[index][key]-expected[index][key]) <= bound + 1e-12
                       for key, bound in bounds.items())
            errors.append(dict(bounds=bounds, signals=signals[index],
                               max_force_error=float(np.abs(delta_f).max())))
        rows.append(dict(width=width, max_energy_error=float(np.abs(actual['energies']-reference['energies']).max()),
                         structures=errors))
    return dict(device=device, n_models=4, comparisons=rows, worker=worker_identity(), gpu=gpu)
