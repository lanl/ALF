"""X-MACE in ALF: fine-tune an excited-state model from a ground-state MACE foundation model.

`X-MACE <https://github.com/rhyan10/X-MACE>`_ puts excited states on a ground-state MACE backbone:
E_s = E_backbone + scale · readout_s + n_atoms · shift_s. Its training script fine-tunes such a model from a
foundation model (MACE-OFF, MACE-OMOL, the MACE-MP family, ...) or from an earlier X-MACE model.
``XMaceFineTuner`` is the adapter for ``fine_tuning.fine_tune_ensemble_task``:

- it writes ALF's multi-state shards (``E0…``, ``F0…``, and optionally GPU4PySCF's ``nacr`` couplings) as
  extended XYZ;
- it runs ``python -m mace.cli.run_train --model=ExcitedMACE …`` once per copy;
- by default, it then fits each copy's NTK posterior on that copy's training frames
  (:mod:`alframework.ml_interfaces.ntk.xmace`), so one copy can choose frames by its own uncertainty.

That matters because copies fine-tuned from one foundation model disagree very little.

``load_xmace_ntk`` loads a copy with its NTK basis for packed MD. ``load_xmace_excited_state_ensemble``
loads the copies for ALF's state-aware sampling.

X-MACE installs as the package ``mace`` in place of mace-torch, so the ML and sampler workers need an
environment of their own. Nothing here imports it, or Torch, until it's used. The default recipe is X-MACE's
fine-tuning tutorial (``tutorials/xmace/04_commands.sh``): learning rate 5e-3, 15 epochs, batch 10, and
loss weights 10 (energies), 100 (forces), and 10 (couplings).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from alframework.ml_interfaces.fine_tuning import ENERGY_KEY, FORCES_KEY, SMOOTH_NACS_KEY, shard_state_atoms

MODEL_NAME = 'xmace'
MODELS = ('ExcitedMACE', 'AutoencoderExcitedMACE', 'LEHExcitedMACE')
DTYPES = ('float32', 'float64')
E0_CHOICES = ('average', 'foundation', 'estimated')
NTK_BASIS_FILE = 'ntk_basis.npz'
NTK_SETTINGS_FILE = 'ntk_settings.json'
NTK_PARAMETERS = ('trainable', 'excited', 'all')
# Arguments ALF sets itself, from fine_tune settings or per copy.
MANAGED = {'name', 'seed', 'train_file', 'valid_file', 'valid_fraction', 'foundation_model', 'E0s', 'energy_key',
           'forces_key', 'nacs_key', 'lr', 'max_num_epochs', 'batch_size', 'valid_batch_size', 'device',
           'default_dtype', 'multiheads_finetuning', 'pt_train_file', 'freeze', 'model', 'n_energies', 'n_states',
           'compute_nacs', 'nac_num', 'nacs_format', 'lora', 'lora_rank', 'lora_alpha', 'model_dir', 'work_dir',
           'log_dir', 'checkpoints_dir', 'results_dir'}
TUTORIAL = dict(learning_rate=5e-3, max_num_epochs=15, batch_size=10,
                options=dict(energy_weight=10.0, forces_weight=100.0, nacs_weight=10.0))
NTK_DEFAULTS = dict(energy_scale=0.05, force_scale=0.1, sketch_dim=64, seed=0, ridge_factor=1e-3, parameters=None)


def _positive(settings, key, kind):
    value = settings[key]
    kinds = (int, float) if kind is float else (kind,)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ValueError(f'fine_tune.{key} must be a positive {kind.__name__}, not {value!r}.')


class XMaceFineTuner:
    """Fine-tune an X-MACE excited-state model with X-MACE's own training. Settings come from ``fine_tune``.

    - ``foundation_model``: a ground-state MACE checkpoint, which becomes the backbone, or an X-MACE model.
    - ``n_states``: S, read from the datasets ``E0…`` and ``F0…``. Or give ``energy_datasets`` and
      ``forces_datasets`` as lists.
    - ``nacs_dataset``: optional, ALF's ``pair_atomic`` couplings (GPU4PySCF's ``nacr``). They're trained in
      X-MACE's smooth form (each coupling times the size of its pair's gap), for the pairs the labels have.
    - ``E0s``: required, as for MACE. It's the isolated-atom energies at the labels' level of theory, a
      ``.json`` file, ``estimated``, or ``average``. Use ``foundation`` only when the labels share the
      foundation model's level of theory.
    - ``model``: one of ``ExcitedMACE`` (default), ``AutoencoderExcitedMACE``, ``LEHExcitedMACE``.
    - ``lora`` (with ``lora_rank`` and ``lora_alpha``): freeze the backbone and train low-rank adapters in it.
      ``freeze`` freezes the backbone's first N layers.
    - ``replay_file``: multi-head fine-tuning with replay of the foundation model's data.
    - ``xmace_options``: extra ``mace_run_train`` arguments, such as loss weights.
    - ``ntk``: the NTK fit after training, or ``null`` for none. ``parameters`` is ``trainable``,
      ``excited``, or ``all``. ``excited`` linearizes in the readouts and adapter only, which is the
      default with LoRA: X-MACE merges LoRA's low-rank weights into the backbone when training ends.
    """

    def __init__(self, settings):
        settings = dict(settings)
        for key in ('learning_rate', 'max_num_epochs', 'batch_size'):
            settings.setdefault(key, TUTORIAL[key])
        settings.setdefault('valid_fraction', 0.1)
        settings.setdefault('dtype', 'float64')
        settings.setdefault('model', 'ExcitedMACE')
        settings.setdefault('freeze', None)
        settings.setdefault('lora', False)
        settings.setdefault('lora_rank', 4)
        settings.setdefault('lora_alpha', 1.0)
        settings.setdefault('replay_file', None)
        settings.setdefault('nacs_dataset', None)
        settings.setdefault('xmace_options', {})
        energies, forces = settings.get('energy_datasets'), settings.get('forces_datasets')
        if energies is None and forces is None:
            n_states = settings.get('n_states')
            if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
                raise ValueError('fine_tune.n_states must be a positive integer, or give energy_datasets and '
                                 'forces_datasets.')
            energies, forces = [f'E{i}' for i in range(n_states)], [f'F{i}' for i in range(n_states)]
        if not energies or not forces or len(energies) != len(forces):
            raise ValueError('fine_tune.energy_datasets and forces_datasets need one entry per state.')
        settings['energy_datasets'], settings['forces_datasets'] = list(energies), list(forces)
        settings['n_states'] = len(energies)
        if 'E0s' not in settings:
            raise ValueError("fine_tune.E0s is required, because it depends on the labels' level of theory: "
                             f"one of {E0_CHOICES}, a dictionary from atomic number to energy, or a .json file.")
        e0s = settings['E0s']
        if isinstance(e0s, dict):
            settings['E0s'] = str({int(z): float(energy) for z, energy in e0s.items()})
        elif not (isinstance(e0s, str) and (e0s.lower() in E0_CHOICES or e0s.endswith('.json'))):
            raise ValueError(f'fine_tune.E0s must be one of {E0_CHOICES}, a dictionary, or a .json file, not {e0s!r}.')
        for key, kind in (('learning_rate', float), ('max_num_epochs', int), ('batch_size', int),
                          ('lora_rank', int), ('lora_alpha', float)):
            _positive(settings, key, kind)
        if not 0 < settings['valid_fraction'] < 1:
            raise ValueError('fine_tune.valid_fraction must be between 0 and 1.')
        if settings['dtype'] not in DTYPES:
            raise ValueError(f"fine_tune.dtype must be one of {DTYPES}, not {settings['dtype']!r}.")
        if settings['model'] not in MODELS:
            raise ValueError(f"fine_tune.model must be one of {MODELS}, not {settings['model']!r}.")
        freeze = settings['freeze']
        if freeze is not None and (isinstance(freeze, bool) or not isinstance(freeze, int)):
            raise ValueError("fine_tune.freeze must be an integer, as for X-MACE's --freeze, or null.")
        options = settings['xmace_options']
        if not isinstance(options, dict):
            raise ValueError('fine_tune.xmace_options must be a dictionary of extra mace_run_train arguments.')
        clash = sorted(MANAGED & set(options))
        if clash:
            raise ValueError(f'fine_tune.xmace_options may not set {clash}; ALF sets them from fine_tune settings.')
        settings['xmace_options'] = {**TUTORIAL['options'], **options}
        ntk = settings.get('ntk', {})
        if ntk is not None:
            if not isinstance(ntk, dict) or set(ntk) - set(NTK_DEFAULTS):
                raise ValueError(f'fine_tune.ntk must be null or a dictionary with keys from {sorted(NTK_DEFAULTS)}.')
            ntk = {**NTK_DEFAULTS, **ntk}
            if ntk['parameters'] is None:
                ntk['parameters'] = 'excited' if settings['lora'] else 'trainable'
            if ntk['parameters'] not in NTK_PARAMETERS:
                raise ValueError(f'fine_tune.ntk.parameters must be one of {NTK_PARAMETERS}.')
        settings['ntk'] = ntk
        self.settings = settings

    def validate(self, master_config):
        properties = master_config.get('properties_list') or {}
        names = {spec[0] for spec in properties.values()}
        wanted = [*self.settings['energy_datasets'], *self.settings['forces_datasets']]
        if self.settings['nacs_dataset']:
            wanted.append(self.settings['nacs_dataset'])
        missing = [name for name in wanted if name not in names]
        if missing:
            raise ValueError(f'X-MACE fine-tuning reads {missing} from each shard, so properties_list must store '
                             'them. Set fine_tune.energy_datasets, forces_datasets, or nacs_dataset to match.')
        replay = self.settings['replay_file']
        if replay is not None and not Path(replay).is_file():
            raise ValueError(f'fine_tune.replay_file not found: {replay}')

    def identity(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            package_version = version('x-mace')
        except PackageNotFoundError:
            package_version = None
        return dict(package='x-mace', package_version=package_version)

    def frames(self, shards):
        s = self.settings
        return shard_state_atoms(shards, s['energy_datasets'], s['forces_datasets'], s['nacs_dataset'],
                                 s.get('h5_read_workers'), max_inflight_bytes=s.get('h5_read_max_inflight_bytes'))

    def prepare_data(self, shards, member_dir):
        """Write every labeled frame to ``member_dir/train.xyz``, in eV, eV/Å, and eV/Å for smooth couplings."""
        from ase.io import write
        frames = self.frames(shards)
        if not frames:
            raise ValueError('There are no frames with every state labeled to fine-tune on.')
        pairs = {json.dumps(atoms.info.get('nac_pairs')) for atoms in frames}
        if self.settings['nacs_dataset'] and len(pairs) != 1:
            raise ValueError('Every shard must couple the same state pairs to train one coupling layout.')
        for atoms in frames:
            # Extended XYZ stores per-atom arrays as [N, k]; X-MACE reads [N, 3S] forces and [N, 3P] couplings.
            atoms.arrays[FORCES_KEY] = atoms.arrays[FORCES_KEY].reshape(len(atoms), -1)
            if SMOOTH_NACS_KEY in atoms.arrays:
                atoms.arrays[SMOOTH_NACS_KEY] = atoms.arrays[SMOOTH_NACS_KEY].reshape(len(atoms), -1)
            atoms.info.pop('nac_pairs', None)
        path = Path(member_dir) / 'train.xyz'
        temporary = path.with_name(path.name + '.tmp')
        write(str(temporary), frames, format='extxyz')
        os.replace(temporary, path)
        if self.settings['nacs_dataset']:
            (Path(member_dir) / 'nac_pairs.json').write_text(pairs.pop())
        return str(path)

    def command(self, data, member_dir, seed, device):
        s = self.settings
        arguments = dict(name=MODEL_NAME, work_dir=member_dir, model_dir=member_dir, seed=seed, model=s['model'],
                         n_energies=s['n_states'], foundation_model=s['foundation_model'], train_file=data,
                         valid_fraction=s['valid_fraction'], E0s=s['E0s'], energy_key=ENERGY_KEY,
                         forces_key=FORCES_KEY, lr=s['learning_rate'], max_num_epochs=s['max_num_epochs'],
                         batch_size=s['batch_size'], valid_batch_size=s['batch_size'], device=device,
                         default_dtype=s['dtype'], multiheads_finetuning=s['replay_file'] is not None)
        if s['nacs_dataset']:
            pairs = json.loads((Path(member_dir) / 'nac_pairs.json').read_text())
            arguments.update(compute_nacs=True, nac_num=len(pairs), nacs_key=SMOOTH_NACS_KEY, nacs_format='smooth')
        if s['replay_file'] is not None:
            arguments['pt_train_file'] = s['replay_file']
        if s['freeze'] is not None:
            arguments['freeze'] = s['freeze']
        if s['lora']:
            arguments.update(lora='True', lora_rank=s['lora_rank'], lora_alpha=s['lora_alpha'])
        arguments.update(s['xmace_options'])
        command = [sys.executable, '-m', 'mace.cli.run_train']
        for key, value in arguments.items():
            if isinstance(value, bool) and key != 'multiheads_finetuning':
                if value:  # A switch, such as --compute_nacs.
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
            raise RuntimeError(f'X-MACE training in {member_dir} exited with {finished.returncode}: ' + ' | '.join(tail))
        if self.settings['ntk'] is not None:
            fit_member_ntk(member_dir, data, self.settings['ntk'], device='cuda' if device.startswith('cuda') else 'cpu')

    def member_complete(self, member_dir):
        member_dir = Path(member_dir)
        done = (member_dir / f'{MODEL_NAME}.model').is_file()
        return done and (self.settings['ntk'] is None or (member_dir / NTK_BASIS_FILE).is_file())


def _adapter(member_dir, parameters, device, dtype=None):
    from alframework.ml_interfaces.ntk.xmace import XMACENTKAdapter
    return XMACENTKAdapter.from_checkpoint(Path(member_dir) / f'{MODEL_NAME}.model', device=device, dtype=dtype,
                                           parameters=parameters)


def fit_member_ntk(member_dir, train_file, ntk, device='cpu'):
    """Fit and save the NTK posterior of a fine-tuned copy on the frames it trained on."""
    from ase.io import read
    from alframework.ml_interfaces.ntk.estimator import NTKEstimator

    adapter = _adapter(member_dir, ntk['parameters'], device)
    frames = read(str(train_file), index=':')
    estimator = NTKEstimator(adapter, frames, energy_scale=ntk['energy_scale'], force_scale=ntk['force_scale'],
                             sketch_dim=ntk['sketch_dim'], seed=ntk['seed'], ridge_factor=ntk['ridge_factor'])
    estimator.save_basis(Path(member_dir) / NTK_BASIS_FILE)
    (Path(member_dir) / NTK_SETTINGS_FILE).write_text(json.dumps(ntk, sort_keys=True))
    return estimator


def load_xmace_ntk_estimator(member_dir, device='cpu'):
    """A fine-tuned copy with the NTK basis its fine-tuning saved, on ``device``."""
    import torch
    from alframework.ml_interfaces.ntk.estimator import NTKEstimator, read_basis_metadata

    member_dir = Path(member_dir)
    ntk = json.loads((member_dir / NTK_SETTINGS_FILE).read_text())
    dtype = getattr(torch, read_basis_metadata(member_dir / NTK_BASIS_FILE)['dtype'])
    return NTKEstimator.from_basis(_adapter(member_dir, ntk['parameters'], device, dtype), member_dir / NTK_BASIS_FILE)


def load_xmace_ntk(model_directory, device='cuda:0', *, member=0, selected_state=0, scores=None,
                   crossing_gap_eV=0.1):
    """``batched_calculator`` loader: one fine-tuned copy, MD on ``selected_state``, with NTK scores.

    ``scores`` are the options of ``NTKEstimator.score``, for example ``{"hessian_probes": 4,
    "outputs": ["state_offsets"]}``.
    """
    from alframework.ml_interfaces.ntk.estimator import NTKPackedEvaluator

    members = sorted(Path(model_directory).glob('model-*/'))
    if not 0 <= member < len(members):
        raise ValueError(f'{model_directory} has {len(members)} fine-tuned copies, not a copy {member}.')
    return NTKPackedEvaluator(load_xmace_ntk_estimator(members[member], device), selected_state=selected_state,
                              crossing_gap_eV=crossing_gap_eV, **(scores or {}))


class XMaceExcitedStateEnsemble:
    """The fine-tuned copies for ALF's state-aware sampling: ``evaluate_states`` as for HIPPYNN.

    It gives energies ``[members, states, systems]`` and forces ``[members, states, atoms, 3]`` in eV and
    eV/Å.
    """

    def __init__(self, model_directory, device='cuda:0', dtype='float32'):
        import torch
        from alframework.ml_interfaces.ntk.xmace import XMACENTKAdapter

        paths = sorted(Path(model_directory).glob(f'model-*/{MODEL_NAME}.model'))
        if not paths:
            raise ValueError(f'No fine-tuned X-MACE copies in {model_directory}')
        self._adapters = [XMACENTKAdapter.from_checkpoint(path, device=device, dtype=getattr(torch, dtype),
                                                          parameters='all') for path in paths]
        self.n_models, self.n_states = len(paths), len(self._adapters[0].state_names)
        self.device = torch.device(device)

    def evaluate_states(self, numbers, positions, cells, pbc, counts):
        import torch
        energies, forces = zip(*(adapter.predict_batch(numbers, positions, cells, pbc, counts)
                                 for adapter in self._adapters))
        return torch.stack(energies), torch.stack(forces)


def load_xmace_excited_state_ensemble(model_directory, device='cuda:0', dtype='float32'):
    """Load the fine-tuned copies for state-aware batched sampling."""
    return XMaceExcitedStateEnsemble(model_directory, device=device, dtype=dtype)
