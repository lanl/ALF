"""Fine-tune copies of an existing model in ALF's loop, so ALF selects the samples that adapt it.

Each round trains ``n_models`` copies of the original checkpoint on every stored label, each with
its own seed and validation split. Their disagreement drives sampling and selection exactly as a
trained ensemble's does, and ``__main__.py`` doesn't change: the bootstrap labels the first
round, and each later store starts another round from the original checkpoint.

A model family plugs in through an adapter class, named by ``fine_tune.adapter``. It is
constructed with the validated settings and provides:

- ``validate(master_config)``: check the run's configuration at startup;
- ``prepare_data(shards, member_dir)``: turn ALF's HDF5 shards into the family's training input;
- ``train_member(data, member_dir, member, seed, device, environment)``: fine-tune copy ``member``;
- ``member_complete(member_dir)``: whether that copy finished;
- ``identity()``: package name and version, recorded with each round.

An adapter whose source is several files, such as a model directory, also provides
``fingerprint()``, returning each file's SHA-256 for the lineage.

The adapters are ``fine_tuning.hippynn.HippynnFineTuner``, ``fine_tuning.mace.MaceFineTuner``, and
``fine_tuning.uma.UmaFineTuner``. ``shard_atoms`` reads ALF's shards for the families that train on
``ase.Atoms``.
"""
import concurrent.futures
import json
import os
import queue
import time
from importlib import import_module
from pathlib import Path

from parsl import join_app, python_app
from alframework.tools.task_timeline import timed
from alframework.tools.capabilities import declares

from alframework.tools import ensemble_training
from alframework.tools.gpu import mask_entries
from alframework.tools.files import cached_sha256, write_json_atomic

DEVICES = ('cpu', 'cuda')
# Where shard_atoms puts each frame's labels, as MACE's extended XYZ names them.
ENERGY_KEY, FORCES_KEY = 'REF_energy', 'REF_forces'
LINEAGE = 'fine_tune_lineage.json'


def _resolve(path, master_directory=None):
    path = Path(str(path)).expanduser()
    if not path.is_absolute() and master_directory:
        path = Path(master_directory) / path
    return path


def fine_tune_settings(ML_config, master_directory=None):
    """Return the validated ``fine_tune`` block, with the foundation model as an absolute path."""
    block = (ML_config or {}).get('fine_tune')
    if not isinstance(block, dict):
        raise ValueError("Fine-tuning needs a 'fine_tune' block in the ML configuration.")
    settings = dict(block)
    for key in ('adapter', 'foundation_model'):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f"fine_tune.{key} must name {'an adapter class' if key == 'adapter' else 'a local model'}.")
    settings['foundation_model'] = str(_resolve(settings['foundation_model'], master_directory))
    # Other local files an adapter reads, such as a replay set, are resolved the same way.
    for key, value in list(settings.items()):
        if key.endswith('_file') and isinstance(value, str) and value:
            settings[key] = str(_resolve(value, master_directory))
    n_models = settings.setdefault('n_models', 4)
    # One copy suits a run that chooses frames by the model's own uncertainty; the coordinator refuses one for a run
    # whose acquisition reads the copies' disagreement (alframework.samplers.acquisition.uses_spread).
    if isinstance(n_models, bool) or not isinstance(n_models, int) or n_models < 1:
        raise ValueError('fine_tune.n_models must be a positive integer.')
    seed = settings.setdefault('seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('fine_tune.seed must be a nonnegative integer.')
    if settings.setdefault('device', 'cuda') not in DEVICES:
        raise ValueError(f"fine_tune.device must be one of {DEVICES}, not {settings['device']!r}.")
    # How many processes read each shard; ML_config's own setting, shared with HIPPYNN training.
    from alframework.tools.h5_parallel import read_budget_setting, read_workers_setting
    settings['h5_read_workers'] = read_workers_setting((ML_config or {}).get('h5_read_workers'))
    settings['h5_read_max_inflight_bytes'] = read_budget_setting((ML_config or {}).get('h5_read_max_inflight_bytes'))
    return settings


def load_adapter(settings):
    module, name = settings['adapter'].rsplit('.', 1)
    return getattr(import_module(module), name)(settings)


def validate_fine_tune_config(ML_config, master_config=None):
    """Check a fine-tuning configuration when ALF starts, rather than after the bootstrap is labeled."""
    master_config = master_config or {}
    settings = fine_tune_settings(ML_config, master_config.get('master_directory'))
    if not Path(settings['foundation_model']).exists():
        raise ValueError(f"Foundation model not found: {settings['foundation_model']}")
    load_adapter(settings).validate(master_config)


def _label_targets(settings):
    # A multi-state adapter names its state datasets; shards with the first state's labels train it.
    energies, forces = settings.get('energy_datasets'), settings.get('forces_datasets')
    if energies and forces:
        return energies[0], forces[0], None
    if settings.get('n_states'):
        return 'E0', 'F0', None
    return (settings.get('energy_dataset', settings.get('energy_key', 'energy')),
            settings.get('forces_dataset', settings.get('force_key', 'forces')),
            settings.get('quadrupole_key'))


def shard_atoms(shards, energy_dataset='energy', forces_dataset='forces', read_workers=None, *,
                max_inflight_bytes=None, include_frame_ids=False):
    """Read ALF shards as ``ase.Atoms`` carrying ``REF_energy`` and ``REF_forces``.

    A shard's ``total_charge`` and ``spin_multiplicity``, if it has them, go to ``atoms.info`` as
    ``charge`` and ``spin``. A shard published over label parts is read from its parts, and a missing part
    is an error (see :func:`alframework.tools.h5_parallel.read_shard`). With ``include_frame_ids``,
    ``atoms.info['alf_frame']`` records the shard, group, row, and optional ALF structure ID.
    """
    import numpy as np
    from ase import Atoms

    from alframework.tools.data_shards import STRUCTURE_STATE_DATASETS
    from alframework.tools.h5_parallel import read_shard

    keys = {'species', 'coordinates', 'cell', energy_dataset, forces_dataset, *STRUCTURE_STATE_DATASETS}
    if include_frame_ids:
        keys.add('_id')
    frames = []
    for shard in shards:
        source = str(Path(shard).resolve()) if include_frame_ids else None
        for group in read_shard(shard, keys, read_workers, max_inflight_bytes=max_inflight_bytes):
            if energy_dataset not in group:
                continue
            symbols = [str(value) for value in group['species']]
            coordinates = group['coordinates']
            energies = np.asarray(group[energy_dataset], dtype=float).reshape(-1)
            forces = group.get(forces_dataset)
            if forces is None:
                raise ValueError(f'{shard}:{group["path"].lstrip("/")} has no {forces_dataset!r} dataset to '
                                 'fine-tune forces on.')
            cells = group.get('cell')
            ids = group.get('_id')
            for index in range(len(energies)):
                atoms = Atoms(symbols, positions=coordinates[index],
                              cell=None if cells is None else cells[index], pbc=cells is not None)
                atoms.info[ENERGY_KEY] = float(energies[index])
                atoms.arrays[FORCES_KEY] = np.asarray(forces[index], dtype=float)
                if include_frame_ids:
                    atoms.info['alf_frame'] = dict(shard=source, group=group['path'], row=index,
                                                   id=None if ids is None else str(ids[index]))
                for key, dataset in zip(('charge', 'spin'), STRUCTURE_STATE_DATASETS):
                    if dataset in group:
                        atoms.info[key] = int(group[dataset][index])
                frames.append(atoms)
    return frames


SMOOTH_NACS_KEY = 'REF_smooth_nacs'


def shard_state_atoms(shards, energy_datasets, forces_datasets, nacs_dataset=None, read_workers=None, *,
                      max_inflight_bytes=None, signed_gap=False, include_frame_ids=False):
    """Read multi-state ALF shards as ``ase.Atoms`` for excited-state training.

    Each frame carries ``REF_energy`` ``[S]`` in ``atoms.info``, and ``REF_forces`` ``[N, S, 3]`` in
    ``atoms.arrays``, in eV and eV/Å. With ``nacs_dataset`` (ALF's ``pair_atomic`` couplings, ``[P, N, 3]``
    in 1/Å, with their state pairs in the shard's ``nac_pairs``), it also carries ``REF_smooth_nacs``
    ``[N, P, 3]``. That's each coupling times the size of its pair's gap, X-MACE's smooth form, which stays
    finite at crossings. ``signed_gap`` instead scales each coupling by E_j − E_i.
    ``atoms.info["nac_pairs"]`` keeps the pairs. A shard's ``total_charge`` and
    ``spin_multiplicity`` go to ``atoms.info`` as ``charge`` and ``spin``. With ``include_frame_ids``,
    ``atoms.info['alf_frame']`` records the shard, group, row, and optional ALF structure ID.
    """
    import numpy as np
    from ase import Atoms

    from alframework.tools.data_shards import PAIR_INDEX_DATASET, STRUCTURE_STATE_DATASETS
    from alframework.tools.h5_parallel import read_shard

    energy_datasets, forces_datasets = list(energy_datasets), list(forces_datasets)
    if not energy_datasets or len(energy_datasets) != len(forces_datasets):
        raise ValueError('Give one forces dataset for each state energy dataset.')
    keys = {'species', 'coordinates', 'cell', *energy_datasets, *forces_datasets, *STRUCTURE_STATE_DATASETS}
    if include_frame_ids:
        keys.add('_id')
    if nacs_dataset:
        keys |= {nacs_dataset, PAIR_INDEX_DATASET}
    frames = []
    for shard in shards:
        source = str(Path(shard).resolve()) if include_frame_ids else None
        for group in read_shard(shard, keys, read_workers, max_inflight_bytes=max_inflight_bytes):
            if any(name not in group for name in energy_datasets):
                continue
            missing = [name for name in forces_datasets if name not in group]
            if missing:
                raise ValueError(f'{shard}:{group["path"].lstrip("/")} has no {missing} datasets to train forces on.')
            symbols = [str(value) for value in group['species']]
            energies = np.stack([np.asarray(group[name], dtype=float).reshape(-1) for name in energy_datasets], 1)
            forces = np.stack([np.asarray(group[name], dtype=float) for name in forces_datasets], 2)  # [K, N, S, 3]
            nacs = pairs = None
            if nacs_dataset and nacs_dataset in group:
                if PAIR_INDEX_DATASET not in group:
                    raise ValueError(f'{shard}:{group["path"].lstrip("/")} has {nacs_dataset!r} but no '
                                     f'{PAIR_INDEX_DATASET!r} saying which states each coupling joins.')
                nacs = np.asarray(group[nacs_dataset], dtype=float)  # [K, P, N, 3]
                pairs = np.asarray(group[PAIR_INDEX_DATASET], dtype=int).reshape(-1, 2)
                if nacs.shape[1] != len(pairs) or pairs.max() >= len(energy_datasets):
                    raise ValueError(f'{shard}: {nacs_dataset!r} pairs do not match the {len(energy_datasets)} states.')
            cells = group.get('cell')
            ids = group.get('_id')
            for index in range(len(energies)):
                atoms = Atoms(symbols, positions=group['coordinates'][index],
                              cell=None if cells is None else cells[index], pbc=cells is not None)
                atoms.info[ENERGY_KEY] = energies[index].copy()
                atoms.arrays[FORCES_KEY] = forces[index].copy()
                if include_frame_ids:
                    atoms.info['alf_frame'] = dict(shard=source, group=group['path'], row=index,
                                                   id=None if ids is None else str(ids[index]))
                if nacs is not None:
                    gaps = energies[index][pairs[:, 1]] - energies[index][pairs[:, 0]]
                    gaps = gaps if signed_gap else np.abs(gaps)
                    atoms.arrays[SMOOTH_NACS_KEY] = np.transpose(nacs[index] * gaps[:, None, None], (1, 0, 2))
                    atoms.info['nac_pairs'] = pairs.tolist()
                for key, dataset in zip(('charge', 'spin'), STRUCTURE_STATE_DATASETS):
                    if dataset in group:
                        atoms.info[key] = int(group[dataset][index])
                frames.append(atoms)
    return frames


def child_environment(device):
    """Return the device name and environment for a child process that should use ``device``.

    ``device`` is ``'cpu'`` or a local ordinal such as ``'cuda:2'``, counted among the GPUs this worker
    sees. The child sees only that GPU. Its mask narrows the worker's innermost one: ``HIP_VISIBLE_DEVICES``
    when set, else ``CUDA_VISIBLE_DEVICES``. A ROCr mask (``ROCR_VISIBLE_DEVICES``, which Slurm sets on AMD
    nodes) is inherited unchanged, since both of those apply to the GPUs it leaves, numbered from 0.
    """
    environment = dict(os.environ)
    if device == 'cpu':
        return 'cpu', environment
    if environment.get('HIP_VISIBLE_DEVICES') is not None and environment.get('CUDA_VISIBLE_DEVICES') is not None:
        raise NotImplementedError('Fine-tuning cannot pick a GPU when both HIP_VISIBLE_DEVICES and '
                                  'CUDA_VISIBLE_DEVICES are set; keep one of them.')
    index = int(str(device).partition(':')[2] or 0)
    name = 'HIP_VISIBLE_DEVICES' if environment.get('HIP_VISIBLE_DEVICES') is not None else 'CUDA_VISIBLE_DEVICES'
    mask = environment.get(name)
    if mask is not None:
        visible = mask_entries(mask)
        if index >= len(visible):
            raise ValueError(f'{device} is outside this worker\'s {name}={mask!r}.')
        environment[name] = visible[index]
    else:
        environment['CUDA_VISIBLE_DEVICES'] = str(index)
    return 'cuda', environment


def member_seed(settings, member):
    return settings['seed'] + member


def fine_tune_member(settings, shards, member_dir, member, device):
    """Fine-tune one copy of the foundation model in ``member_dir``. Returns whether it finished."""
    member_dir = Path(member_dir)
    member_dir.mkdir(parents=True, exist_ok=True)
    if any(member_dir.iterdir()):
        raise FileExistsError(f'Refusing to fine-tune into nonempty {member_dir}')
    adapter = load_adapter(settings)
    name, environment = child_environment(device)
    data = adapter.prepare_data(shards, member_dir)
    adapter.train_member(data, member_dir, member, member_seed(settings, member), name, environment)
    return bool(adapter.member_complete(member_dir))


def write_lineage(ensemble_dir, settings, shards, completed=None):
    """Record what a round started from and learned from, beside its members."""
    adapter = load_adapter(settings)
    fingerprint = getattr(adapter, 'fingerprint', None)
    source = (dict(foundation_model_files=fingerprint()) if fingerprint is not None
              else dict(foundation_model_sha256=cached_sha256(settings['foundation_model'])))
    record = dict(
        foundation_model=settings['foundation_model'], **source,
        adapter=settings['adapter'], **adapter.identity(),
        # The adapter's settings include its defaults, so the record says exactly what ran.
        settings={key: value for key, value in getattr(adapter, 'settings', settings).items()
                  if key not in ('adapter', 'foundation_model')},
        shards=[os.path.basename(shard) for shard in shards],
        seeds=[member_seed(settings, member) for member in range(settings['n_models'])],
        completed=completed, written=time.time())
    write_json_atomic(Path(ensemble_dir) / LINEAGE, record)


def _member_dir(ensemble_dir, member):
    return ensemble_training.member_directory(ensemble_dir, member)


def _single_node_devices(settings, gpus_per_node):
    if settings['device'] == 'cpu':
        return ['cpu']  # Each copy uses every core, so they run one after another.
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    visible = len([entry for entry in mask.split(',') if entry.strip()]) if mask is not None else None
    count = min(filter(None, (gpus_per_node, visible)), default=1)
    return [f'cuda:{index}' for index in range(count)]


def fine_tune_ensemble(ML_config, h5_dir, model_path, current_training_id, gpus_per_node=None, h5_path=None,
                       current_h5_id=None, master_directory=None):
    """Fine-tune every copy on this worker, one per GPU at a time. Returns ``(completed, training_id)``."""
    settings = fine_tune_settings(ML_config, master_directory)
    shards = ensemble_training.training_shards(h5_dir, h5_path, current_h5_id, targets=_label_targets(settings))
    ensemble_dir = model_path.format(int(current_training_id))
    root = Path(ensemble_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty ensemble directory {root}')
    free = queue.Queue()
    for device in _single_node_devices(settings, gpus_per_node):
        free.put(device)

    def run(member):
        device = free.get()
        try:
            return fine_tune_member(settings, shards, _member_dir(ensemble_dir, member), member, device)
        except Exception as exc:
            print(f'Fine-tuning {int(current_training_id):04d}: member {member} failed: '
                  f'{type(exc).__name__}: {exc}', flush=True)
            return False
        finally:
            free.put(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=free.qsize()) as pool:
        completed = list(pool.map(run, range(settings['n_models'])))
    write_lineage(ensemble_dir, settings, shards, completed)
    from alframework.tools.label_sources import write_label_sources
    write_label_sources(ensemble_dir, shards=shards)
    return completed, int(current_training_id)


@python_app(executors=['alf_ML_executor'])
@declares('h5_read_workers', 'h5_read_max_inflight_bytes')
@timed('ML')
def fine_tune_ensemble_task(ML_config, h5_dir, model_path, current_training_id, gpus_per_node=None,
                            h5_path=None, current_h5_id=None, master_directory=None):
    """Fine-tune ``fine_tune.n_models`` copies of the foundation model on one node.

    ``h5_path`` and ``current_h5_id`` come from the master configuration and status; with both
    given, training waits until every stored shard is visible.
    """
    from alframework.ml_interfaces.fine_tuning import fine_tune_ensemble
    return fine_tune_ensemble(ML_config, h5_dir, model_path, current_training_id, gpus_per_node, h5_path,
                              current_h5_id, master_directory)


def _fine_tune_member_in_child(settings, shards, member_dir, member, gpus_per_node, attempt):
    device = ensemble_training.member_device(settings['device'], gpus_per_node)
    with ensemble_training.member_placement(member_dir, attempt, device) as record:
        record['completed'] = fine_tune_member(settings, shards, member_dir, member, device)
    return record['completed']


@python_app(executors=['alf_ML_executor'])
@timed('ML')
def _fine_tune_member_task(settings, shards, member_dir, member, gpus_per_node, attempt):
    from alframework.ml_interfaces import fine_tuning
    from alframework.tools import ensemble_training
    ensemble_training.wait_for_shards(shards)
    return ensemble_training.run_member_in_child(fine_tuning._fine_tune_member_in_child, settings, shards,
                                                 member_dir, member, gpus_per_node, attempt)


@join_app
@declares('h5_read_workers', 'h5_read_max_inflight_bytes')
def fine_tune_ensemble_multinode_task(ML_config, h5_dir, model_path, current_training_id, gpus_per_node=None,
                                      master_directory=None, member_retries=1, h5_path=None, current_h5_id=None):
    """Fine-tune each copy as its own app, so a round can span any number of nodes.

    This body runs on the coordinator and doesn't train. Failed members are retried as for the
    HIPPYNN multi-node tasks; see the ML interfaces guide for the Parsl setup.
    """
    settings = fine_tune_settings(ML_config, master_directory)
    ensemble_dir = model_path.format(int(current_training_id))
    root = Path(ensemble_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty ensemble directory {root}')
    shards = ensemble_training.training_shards(h5_dir, h5_path, current_h5_id, targets=_label_targets(settings))
    executor = None
    try:
        import parsl
        executor = parsl.dfk().executors.get('alf_ML_executor')
    except Exception:
        pass

    def finish():
        from alframework.tools.label_sources import write_label_sources
        # Placement is written by the actual member child before its app
        # returns. Use the last attempt, including failed attempts, rather
        # than treating the mere existence of a checkpoint as success.
        completed = []
        for member in range(settings['n_models']):
            attempts = [json.loads(path.read_text()) for path in
                        root.glob(f'placement-model-{member:02d}-attempt-*.json')]
            latest = max(attempts, key=lambda record: record['attempt'], default={})
            completed.append(bool(latest.get('completed', False)))
        write_lineage(ensemble_dir, settings, shards, completed)
        write_label_sources(ensemble_dir, shards=shards)

    return ensemble_training.train_ensemble_across_nodes(
        lambda member, attempt: _fine_tune_member_task(settings, shards, _member_dir(ensemble_dir, member), member,
                                                       gpus_per_node, attempt),
        settings['n_models'], ensemble_dir, int(current_training_id), gpus_per_node, settings['device'],
        member_retries, executor, finish=finish)


# __main__ calls this at startup. The docs build replaces the apps with plain functions.
for _task in (fine_tune_ensemble_task, fine_tune_ensemble_multinode_task):
    getattr(_task, 'func', _task).validate_config = validate_fine_tune_config

getattr(fine_tune_ensemble_multinode_task, 'func', fine_tune_ensemble_multinode_task).member_executors = (
    'alf_ML_executor',)
# Each copy trains on one GPU of the configured device, as fine_tune_ensemble_multinode_task submits it.
getattr(fine_tune_ensemble_multinode_task, 'func', fine_tune_ensemble_multinode_task).member_layout = (
    lambda ML_config: ((ML_config.get('fine_tune') or {}).get('device', 'cuda'), 1, 1))
