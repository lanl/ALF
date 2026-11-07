"""MACE in ALF: fine-tune a MACE foundation model on ALF's labels, and sample with the copies.

``MaceFineTuner`` is the adapter for ``fine_tuning.fine_tune_ensemble_task``. It writes ALF's shards
as extended XYZ and runs MACE's own training (``python -m mace.cli.run_train``) once per copy, in
that copy's directory, so each copy keeps MACE's logs and checkpoints. ``load_mace_batched_ensemble``
loads the copies for the TorchSim and ALCHEMI sampling engines.

Nothing here imports MACE or Torch until it's used, so ALF runs without them installed.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = 'mace'
ENERGY_KEY, FORCES_KEY = 'REF_energy', 'REF_forces'
DTYPES = ('float32', 'float64')
E0_CHOICES = ('average', 'foundation', 'estimated')
# Arguments ALF sets itself, from fine_tune settings or per copy.
MANAGED = {'name', 'seed', 'train_file', 'valid_file', 'valid_fraction', 'foundation_model', 'E0s',
           'energy_key', 'forces_key', 'lr', 'max_num_epochs', 'batch_size', 'valid_batch_size', 'device',
           'default_dtype', 'multiheads_finetuning', 'pt_train_file', 'freeze', 'model_dir', 'work_dir',
           'log_dir', 'checkpoints_dir', 'results_dir'}


def _positive(settings, key, kind):
    value = settings[key]
    kinds = (int, float) if kind is float else (kind,)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ValueError(f'fine_tune.{key} must be a positive {kind.__name__}, not {value!r}.')


class MaceFineTuner:
    """Fine-tune a MACE foundation model with MACE's own training. Settings come from ``fine_tune``.

    ``E0s`` has no default, because it depends on the labels' level of theory: ``"foundation"``
    keeps the foundation model's, ``"average"`` and ``"estimated"`` fit new ones, and a dictionary
    from atomic number to energy, or a ``.json`` file, sets them. Without ``replay_file``, copies are
    fine-tuned on ALF's labels alone; with it, MACE's multihead fine-tuning also trains on that local
    sample of the foundation model's data, to limit forgetting.
    """

    def __init__(self, settings):
        settings = dict(settings)
        settings.setdefault('learning_rate', 1e-3)
        settings.setdefault('max_num_epochs', 50)
        settings.setdefault('batch_size', 4)
        settings.setdefault('valid_fraction', 0.1)
        settings.setdefault('dtype', 'float64')
        settings.setdefault('freeze', None)
        settings.setdefault('replay_file', None)
        settings.setdefault('energy_dataset', 'energy')
        settings.setdefault('forces_dataset', 'forces')
        settings.setdefault('mace_options', {})
        if 'E0s' not in settings:
            raise ValueError("fine_tune.E0s is required, because it depends on the labels' level of theory: "
                             f"one of {E0_CHOICES}, a dictionary from atomic number to energy, or a .json file.")
        e0s = settings['E0s']
        if isinstance(e0s, dict):
            settings['E0s'] = str({int(z): float(energy) for z, energy in e0s.items()})
        elif not (isinstance(e0s, str) and (e0s.lower() in E0_CHOICES or e0s.endswith('.json'))):
            raise ValueError(f'fine_tune.E0s must be one of {E0_CHOICES}, a dictionary, or a .json file, not {e0s!r}.')
        _positive(settings, 'learning_rate', float)
        _positive(settings, 'max_num_epochs', int)
        _positive(settings, 'batch_size', int)
        if not 0 < settings['valid_fraction'] < 1:
            raise ValueError('fine_tune.valid_fraction must be between 0 and 1.')
        if settings['dtype'] not in DTYPES:
            raise ValueError(f"fine_tune.dtype must be one of {DTYPES}, not {settings['dtype']!r}.")
        freeze = settings['freeze']
        if freeze is not None and (isinstance(freeze, bool) or not isinstance(freeze, int)):
            raise ValueError('fine_tune.freeze must be an integer, as for MACE\'s --freeze, or null.')
        options = settings['mace_options']
        if not isinstance(options, dict):
            raise ValueError('fine_tune.mace_options must be a dictionary of extra mace_run_train arguments.')
        clash = sorted(MANAGED & set(options))
        if clash:
            raise ValueError(f'fine_tune.mace_options may not set {clash}; ALF sets them from fine_tune settings.')
        self.settings = settings

    def validate(self, master_config):
        properties = master_config.get('properties_list') or {}
        names = {key: spec[0] for key, spec in properties.items()}
        for key, dataset in (('energy', self.settings['energy_dataset']), ('forces', self.settings['forces_dataset'])):
            if names.get(key) != dataset:
                raise ValueError(f"MACE fine-tuning reads '{dataset}' from each shard, so properties_list must store "
                                 f"'{key}' there. Set fine_tune.{key}_dataset to match.")
        replay = self.settings['replay_file']
        if replay is not None and not Path(replay).is_file():
            raise ValueError(f'fine_tune.replay_file not found: {replay}')

    def identity(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            package_version = version('mace-torch')
        except PackageNotFoundError:
            package_version = None
        return dict(package='mace-torch', package_version=package_version)

    def prepare_data(self, shards, member_dir):
        """Write every labeled frame in ``shards`` to ``member_dir/train.xyz``, in eV and eV/Å."""
        from ase.io import write
        frames = shard_frames(shards, self.settings['energy_dataset'], self.settings['forces_dataset'])
        if not frames:
            raise ValueError('There are no labeled frames to fine-tune on.')
        path = Path(member_dir) / 'train.xyz'
        temporary = path.with_name(path.name + '.tmp')
        write(str(temporary), frames, format='extxyz')
        os.replace(temporary, path)
        return str(path)

    def command(self, data, member_dir, seed, device):
        s = self.settings
        arguments = dict(name=MODEL_NAME, work_dir=member_dir, model_dir=member_dir, seed=seed,
                         foundation_model=s['foundation_model'], train_file=data, valid_fraction=s['valid_fraction'],
                         E0s=s['E0s'], energy_key=ENERGY_KEY, forces_key=FORCES_KEY, lr=s['learning_rate'],
                         max_num_epochs=s['max_num_epochs'], batch_size=s['batch_size'],
                         valid_batch_size=s['batch_size'], device=device, default_dtype=s['dtype'],
                         multiheads_finetuning=s['replay_file'] is not None)
        if s['replay_file'] is not None:
            arguments['pt_train_file'] = s['replay_file']
        if s['freeze'] is not None:
            arguments['freeze'] = s['freeze']
        arguments.update(s['mace_options'])
        command = [sys.executable, '-m', 'mace.cli.run_train']
        for key, value in arguments.items():
            if isinstance(value, bool) and key != 'multiheads_finetuning':
                if value:  # A switch, such as --swa.
                    command.append(f'--{key}')
            else:
                command.append(f'--{key}={value}')
        return command

    def train_member(self, data, member_dir, member, seed, device, environment):
        log = Path(member_dir) / 'train.log'
        with open(log, 'w') as handle:
            finished = subprocess.run(self.command(data, str(member_dir), seed, device), cwd=member_dir,
                                      env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if finished.returncode != 0:
            tail = log.read_text(errors='replace').strip().splitlines()[-5:]
            raise RuntimeError(f'MACE training in {member_dir} exited with {finished.returncode}: ' + ' | '.join(tail))

    def member_complete(self, member_dir):
        return (Path(member_dir) / f'{MODEL_NAME}.model').is_file()


def shard_frames(shards, energy_dataset='energy', forces_dataset='forces'):
    """Read ALF shards as ``ase.Atoms`` carrying ``REF_energy`` and ``REF_forces``."""
    import h5py
    from ase import Atoms

    frames = []
    for shard in shards:
        with h5py.File(shard, 'r') as handle:
            for name in sorted(handle):
                group = handle[name]
                if not isinstance(group, h5py.Group) or energy_dataset not in group:
                    continue
                symbols = [value.decode() if isinstance(value, bytes) else str(value) for value in group['species'][()]]
                coordinates = group['coordinates'][()]
                energies = np.asarray(group[energy_dataset][()], dtype=float).reshape(-1)
                forces = group[forces_dataset][()] if forces_dataset in group else None
                if forces is None:
                    raise ValueError(f'{shard}:{name} has no {forces_dataset!r} dataset to fine-tune forces on.')
                cells = group['cell'][()] if 'cell' in group else None
                for index in range(len(energies)):
                    atoms = Atoms(symbols, positions=coordinates[index],
                                  cell=None if cells is None else cells[index], pbc=cells is not None)
                    atoms.info[ENERGY_KEY] = float(energies[index])
                    atoms.arrays[FORCES_KEY] = np.asarray(forces[index], dtype=float)
                    frames.append(atoms)
    return frames


class MaceBatchedEnsemble:
    """Evaluate every fine-tuned copy on a whole batch, for the batched sampling engines.

    Implements ``evaluate(numbers, positions, cells, pbc, counts)``: ``cells`` are row vectors, as
    in ASE, and the result is ``(energies [models, systems], forces [models, atoms, 3])`` in eV and
    eV/Å, through MACE's own TorchSim interface.
    """

    def __init__(self, model_directory, device='cuda:0', dtype='float32', models=None):
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        if models is None:
            from mace.calculators.mace_torchsim import MaceTorchSimModel
            paths = sorted(Path(model_directory).glob(f'model-*/{MODEL_NAME}.model'))
            models = [MaceTorchSimModel(str(path), device=self.device, dtype=self.dtype, compute_forces=True,
                                        compute_stress=False) for path in paths]
        if len(models) < 2:
            raise ValueError(f'An uncertainty ensemble requires at least two fine-tuned copies in {model_directory}.')
        self._members = list(models)
        self.n_models = len(self._members)

    def evaluate(self, numbers, positions, cells, pbc, counts):
        torch = self._torch
        counts = torch.as_tensor(counts, device=self.device)
        system_idx = torch.repeat_interleave(torch.arange(len(counts), device=self.device), counts)
        positions = positions.detach().to(self.device, self.dtype)
        state = dict(positions=positions, masses=torch.ones(len(positions), device=self.device, dtype=self.dtype),
                     cell=cells.detach().to(self.device, self.dtype).transpose(-1, -2),
                     pbc=torch.as_tensor([bool(value) for value in pbc], device=self.device),
                     atomic_numbers=numbers.to(self.device, torch.long), system_idx=system_idx)
        energies, forces = [], []
        for model in self._members:
            output = model(dict(state))
            energies.append(output['energy'].detach().reshape(len(counts)))
            forces.append(output['forces'].detach().reshape(len(positions), 3))
        return torch.stack(energies), torch.stack(forces)


def load_mace_batched_ensemble(model_directory, device='cuda:0', dtype='float32'):
    """Load fine-tuned MACE copies for batched sampling (``batched_calculator`` loader)."""
    return MaceBatchedEnsemble(model_directory, device=device, dtype=dtype)
