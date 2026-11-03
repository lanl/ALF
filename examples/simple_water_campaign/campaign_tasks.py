"""Observation-only adapters around the existing production task functions."""
from pathlib import Path
import os
import time

from parsl import python_app

from campaign_support import atomic_json, event, h5_inventory, member_reports, read_json, worker_identity

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


@python_app(executors=['alf_sampler_executor'])
def build_water(moleculeids, builder_config, molecule_library_dir, cell_range, solute_molecule_options, Rrange):
    from alframework.builders.builders import simple_multi_condensed_phase_builder_task
    identify()
    event('builder_start', ids=moleculeids, width=len(moleculeids))
    result = unwrap(simple_multi_condensed_phase_builder_task.func(
        moleculeids, builder_config, molecule_library_dir, cell_range, solute_molecule_options, Rrange))
    assert [m.get_moleculeid() for m in result] == list(moleculeids)
    assert all(m.get_atoms().pbc.all() and m.get_atoms().cell.volume > 0 for m in result)
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
    result = unwrap(mlmd_trajectories_task.func(molecule_objects, sampler_config, model_path,
                                               current_model_id, gpus_per_node))
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
    result = unwrap(orca_calculator_task.func(molecule_object, QM_config, QM_scratch_dir, properties_list))
    converged = bool(result.check_convergence())
    if converged:
        assert all(np.isfinite(value).all() for value in result.get_results().values())
    event('qm_end', id=result.get_moleculeid(), converged=converged, structure=geometry(result),
          properties=result.get_results())
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
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble, load_hippynn_batched_ensemble
    from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator, ensemble_statistics
    from alframework.tools.gpu import sampling_device
    identify()
    device = sampling_device({'device': 'cuda'}, 4)
    atoms = read(structures_file, index=':')
    reference = ASEEnsembleEvaluator(HIPNN_ASE_load_ensemble(model_directory, device=device)).predict(atoms)
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
    return dict(device=device, n_models=4, comparisons=rows, worker=worker_identity())
