"""MACE in ALF: fine-tune a MACE foundation model on ALF's labels, and sample with the copies.

``MaceFineTuner`` is the MACE adapter for ``fine_tuning.fine_tune_ensemble_task``. It writes ALF's shards
as extended XYZ and runs MACE's own training (``python -m mace.cli.run_train``) once per copy, in
that copy's directory, so each copy keeps MACE's logs and checkpoints. ``load_mace_batched_ensemble``
loads the copies for the TorchSim and ALCHEMI sampling engines.

Nothing here imports MACE or Torch until it's used, so ALF runs without them installed.
"""
import os
import subprocess
import sys
from pathlib import Path

from alframework.ml_interfaces.fine_tuning import ENERGY_KEY, FORCES_KEY, shard_atoms
from alframework.ml_interfaces.torchsim_ensemble import TorchSimBatchedEnsemble

MODEL_NAME = 'mace'
DTYPES = ('float32', 'float64')
E0_CHOICES = ('average', 'foundation', 'estimated')
# Arguments ALF sets itself, from fine_tune settings or per copy.
MANAGED = {'name', 'seed', 'train_file', 'valid_file', 'valid_fraction', 'foundation_model', 'E0s',
           'energy_key', 'forces_key', 'lr', 'max_num_epochs', 'batch_size', 'valid_batch_size', 'device',
           'default_dtype', 'multiheads_finetuning', 'pt_train_file', 'freeze', 'model_dir', 'work_dir',
           'log_dir', 'checkpoints_dir', 'results_dir'}
# Defaults from MACE's own fine-tuning guidance (mace-docs, "Fine-tuning foundation models"), which
# mace_options can override: plain (naive) fine-tuning, and multihead fine-tuning with replay.
GUIDE = 'https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html'
NAIVE = dict(learning_rate=1e-3, max_num_epochs=6, options=dict(
    ema=True, ema_decay=0.999, amsgrad=True, clip_grad=1.0, weight_decay=0.0, energy_weight=10.0, forces_weight=10.0,
    scaling='rms_forces_scaling'))
REPLAY = dict(learning_rate=1e-4, max_num_epochs=20, options=dict(NAIVE['options'], ema_decay=0.9999))


def _positive(settings, key, kind):
    value = settings[key]
    kinds = (int, float) if kind is float else (kind,)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ValueError(f'fine_tune.{key} must be a positive {kind.__name__}, not {value!r}.')


class MaceFineTuner:
    """Fine-tune a MACE foundation model with MACE's own training. Settings come from ``fine_tune``.

    ``E0s`` has no default, because it depends on the labels' level of theory. MACE recommends
    isolated-atom energies (a dictionary from atomic number to energy, or a ``.json`` file), or else
    ``"estimated"``; ``"foundation"`` only for labels at the foundation model's level of theory; and
    not ``"average"``. Without ``replay_file``, copies are fine-tuned on ALF's labels alone; with it,
    MACE's multihead fine-tuning also trains on that local sample of the foundation model's data, to
    limit forgetting. The other defaults follow MACE's guidance for each of the two.
    """

    def __init__(self, settings):
        settings = dict(settings)
        recipe = REPLAY if settings.get('replay_file') else NAIVE
        settings.setdefault('learning_rate', recipe['learning_rate'])
        settings.setdefault('max_num_epochs', recipe['max_num_epochs'])
        settings.setdefault('batch_size', 2)
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
        # MACE's recommended options, under any the user sets; recorded in full in the lineage.
        settings['mace_options'] = {**recipe['options'], **options}
        self.settings = settings

    def validate(self, master_config):
        properties = master_config.get('properties_list') or {}
        names = {spec[0] for spec in properties.values()}
        for key, dataset in (('energy', self.settings['energy_dataset']), ('forces', self.settings['forces_dataset'])):
            if dataset not in names:
                raise ValueError(f"MACE fine-tuning reads '{dataset}' from each shard, so properties_list must store "
                                 f"'{key}' there. Set fine_tune.{key}_dataset to match.")
        replay = self.settings['replay_file']
        if replay is not None and not Path(replay).is_file():
            raise ValueError(f'fine_tune.replay_file not found: {replay}')
        if str(self.settings['E0s']).lower() == 'average':
            print("Warning: fine_tune.E0s is 'average', which MACE's fine-tuning guidance advises against. It "
                  f"recommends isolated-atom energies, or 'estimated'; see {GUIDE}.", flush=True)

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
        frames = shard_atoms(shards, self.settings['energy_dataset'], self.settings['forces_dataset'])
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


class MaceBatchedEnsemble(TorchSimBatchedEnsemble):
    """Every fine-tuned MACE copy, evaluated on a whole batch through MACE's own TorchSim interface."""

    def __init__(self, model_directory, device='cuda:0', dtype='float32', models=None):
        if models is None:
            import torch
            from mace.calculators.mace_torchsim import MaceTorchSimModel
            paths = sorted(Path(model_directory).glob(f'model-*/{MODEL_NAME}.model'))
            models = [MaceTorchSimModel(str(path), device=torch.device(device),
                                        dtype=torch.float64 if dtype == 'float64' else torch.float32,
                                        compute_forces=True, compute_stress=False) for path in paths]
        super().__init__(models, device=device, dtype=dtype, source=model_directory)


def load_mace_batched_ensemble(model_directory, device='cuda:0', dtype='float32'):
    """Load fine-tuned MACE copies for batched sampling (``batched_calculator`` loader)."""
    return MaceBatchedEnsemble(model_directory, device=device, dtype=dtype)
