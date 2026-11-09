"""Fine-tune a UMA checkpoint on ALF's labels, through ``fine_tuning.fine_tune_ensemble_task``.

Each copy runs fairchem's own training in a child process
(``python -m alframework.ml_interfaces.fine_tuning.uma``), from the local checkpoint, so compute
nodes need no network. ALF writes fairchem's training configuration itself. Its structure follows
fairchem's UMA fine-tuning templates (``configs/uma/finetune`` in fairchem 2.25, Copyright (c) Meta
Platforms, Inc. and affiliates, MIT license), which ship only in fairchem's git repository.

``load_uma_batched_ensemble`` loads the copies for the TorchSim and ALCHEMI sampling engines.
Nothing here imports fairchem or Torch until it's used.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

TASKS = ('omol', 'omat', 'odac', 'oc20', 'oc25', 'omc')
E0_CHOICES = ('average', 'foundation')
MODEL_NAME = 'uma.pt'
ARGUMENTS = 'fine_tune_args.json'
CONFIG = 'fine_tune.yaml'
LOG = 'train.log'
# As fairchem's template sets them; raise epochs for a small label set.
DEFAULTS = dict(epochs=1, steps=None, batch_size=2, lr=4e-4, weight_decay=1e-3, max_neighbors=300,
                valid_fraction=0.1, energy_dataset='energy', forces_dataset='forces',
                keep_training_checkpoints=False)
# fairchem's job config takes the enum's names.
DEVICE_TYPES = {'cpu': 'CPU', 'cuda': 'CUDA'}


class UmaFineTuner:
    """Fine-tune one UMA task of a local checkpoint. Settings come from ``fine_tune``.

    ``task`` picks the UMA head, and so the level of theory the labels must match. ``E0s`` is
    ``"average"``, which refits per-element reference energies to the labels as fairchem's recipe
    does, or ``"foundation"``, which keeps the checkpoint's own references for that task.
    """

    def __init__(self, settings):
        settings = {**DEFAULTS, **dict(settings)}
        if settings.get('task') not in TASKS:
            raise ValueError(f'fine_tune.task is required for UMA: one of {TASKS}. Each is a different level of '
                             'theory, and the labels must match it.')
        if settings.get('E0s') not in E0_CHOICES:
            raise ValueError(f"fine_tune.E0s is required for UMA: one of {E0_CHOICES}. 'average' refits the "
                             "per-element reference energies to the labels; 'foundation' keeps the checkpoint's.")
        if (settings['epochs'] is None) == (settings['steps'] is None):
            raise ValueError('Set exactly one of fine_tune.epochs and fine_tune.steps for UMA.')
        for key in ('epochs', 'steps', 'batch_size', 'max_neighbors'):
            value = settings[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f'fine_tune.{key} must be a positive integer, not {value!r}.')
        for key in ('lr', 'weight_decay'):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (key == 'lr' and value == 0):
                raise ValueError(f'fine_tune.{key} must be a {"positive" if key == "lr" else "nonnegative"} number.')
        if not 0 < settings['valid_fraction'] < 1:
            raise ValueError('fine_tune.valid_fraction must be between 0 and 1.')
        self.settings = settings

    def validate(self, master_config):
        if not Path(self.settings['foundation_model']).is_file():
            raise ValueError(f"A UMA foundation_model is a checkpoint file: {self.settings['foundation_model']}")
        names = {spec[0] for spec in (master_config.get('properties_list') or {}).values()}
        for key, dataset in (('energy', self.settings['energy_dataset']), ('forces', self.settings['forces_dataset'])):
            if dataset not in names:
                raise ValueError(f"UMA fine-tuning reads '{dataset}' from each shard, so properties_list must store "
                                 f"'{key}' there. Set fine_tune.{key}_dataset to match.")

    def identity(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            package_version = version('fairchem-core')
        except PackageNotFoundError:
            package_version = None
        return dict(package='fairchem-core', package_version=package_version)

    def prepare_data(self, shards, member_dir):
        """The copy converts the shards itself, with its own seed for its split, so this only checks them."""
        if not shards:
            raise ValueError('There are no labeled shards to fine-tune on.')
        return [str(shard) for shard in shards]

    def train_member(self, data, member_dir, member, seed, device, environment):
        path = Path(member_dir) / ARGUMENTS
        path.write_text(json.dumps(dict(shards=data, member_dir=str(member_dir), seed=seed, device=device,
                                        settings=self.settings), indent=2, default=str) + '\n')
        log = Path(member_dir) / LOG
        with open(log, 'w') as handle:
            finished = subprocess.run([sys.executable, '-m', 'alframework.ml_interfaces.fine_tuning.uma', str(path)],
                                      cwd=member_dir, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if finished.returncode != 0:
            tail = log.read_text(errors='replace').strip().splitlines()[-3:]
            raise RuntimeError(f'UMA fine-tuning in {member_dir} exited with {finished.returncode}: ' + ' | '.join(tail))

    def member_complete(self, member_dir):
        return (Path(member_dir) / MODEL_NAME).is_file()


def split(frames, fraction, seed):
    """Training and validation frames, split with this copy's seed. Each gets at least one."""
    import numpy as np
    if len(frames) < 2:
        raise ValueError(f'UMA fine-tuning needs at least 2 labeled structures, not {len(frames)}.')
    order = np.random.default_rng(seed).permutation(len(frames))
    n_valid = min(len(frames) - 1, max(1, int(fraction * len(frames))))
    return [frames[i] for i in order[n_valid:]], [frames[i] for i in order[:n_valid]]


def fitted_references(frames):
    """Per-element reference energies by least squares, and the force RMS, as fairchem's recipe fits them."""
    import numpy as np
    counts = np.vstack([np.bincount(atoms.numbers, minlength=100) for atoms in frames])
    energies = np.array([atoms.get_potential_energy() for atoms in frames])
    references = np.linalg.lstsq(counts, energies, rcond=None)[0]
    force_rms = float(np.sqrt(np.mean(np.concatenate([atoms.get_forces().reshape(-1) for atoms in frames]) ** 2)))
    if not (np.isfinite(references).all() and np.isfinite(force_rms)):
        raise ValueError('Fitting reference energies gave non-finite values; check the labels.')
    return references.tolist(), force_rms


def checkpoint_references(checkpoint, task):
    """The checkpoint's own reference energies and energy normalizer for ``task``."""
    import torch
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for entry in getattr(loaded, 'tasks_config', None) or []:
        if entry.get('name') == f'{task}_energy':
            references = entry['element_references']['element_references']['_args_'][0]
            return [float(value) for value in references], float(entry['normalizer']['rmsd'])
    raise ValueError(f"{checkpoint} records no reference energies for task {task!r}, so E0s 'foundation' "
                     "can't keep them. Use 'average'.")


def training_config(settings, train_dir, valid_dir, references, normalizer_rmsd, run_dir, seed, device):
    """fairchem's training configuration for one copy, as its UMA templates lay it out, in one file."""
    task = settings['task']
    dataset = dict(format='ase_db', transforms=dict(common_transform=dict(dataset_name='${data.dataset_name}'),
                                                    stress_reshape_transform=dict(dataset_name='${data.dataset_name}')))
    # Every task reads charge and spin; omol's model uses them, and they default to 0 and 1.
    dataset['a2g_args'] = {'r_data_keys': ['spin', 'charge']}
    normalizer = {'_target_': 'fairchem.core.modules.normalization.normalizer.Normalizer', 'mean': 0.0,
                  'rmsd': '${data.normalizer_rmsd}'}
    task_base = {'_target_': 'fairchem.core.units.mlip_unit.mlip_unit.Task', 'normalizer': normalizer,
                 'datasets': ['${data.dataset_name}']}
    tasks = [
        dict(task_base, name='energy', level='system', property='energy',
             loss_fn={'_target_': 'fairchem.core.modules.loss.DDPMTLoss',
                      'loss_fn': {'_target_': 'fairchem.core.modules.loss.PerAtomMAELoss'}, 'coefficient': 20},
             out_spec={'dim': [1], 'dtype': 'float32'},
             element_references={'_target_': 'fairchem.core.modules.normalization.element_references.ElementReferences',
                                 'element_references': {'_target_': 'torch.DoubleTensor', '_args_': ['${data.elem_refs}']}},
             metrics=['mae', 'per_atom_mae']),
        dict(task_base, name='forces', level='atom', property='forces', train_on_free_atoms=True, eval_on_free_atoms=True,
             loss_fn={'_target_': 'fairchem.core.modules.loss.DDPMTLoss',
                      'loss_fn': {'_target_': 'fairchem.core.modules.loss.L2NormLoss'}, 'reduction': 'mean', 'coefficient': 2},
             out_spec={'dim': [3], 'dtype': 'float32'}, metrics=['mae', 'cosine_similarity', 'magnitude_error'])]
    data = dict(dataset_name=task, elem_refs=list(references), normalizer_rmsd=float(normalizer_rmsd),
                train_dataset=dict(splits=dict(train=dict(src=str(train_dir))), **dataset),
                val_dataset=dict(splits=dict(val=dict(src=str(valid_dir))), **dataset),
                # ALF stores energies and forces, not stress, so the model is fine-tuned on those alone.
                regress_stress=False, pass_through_head_outputs=True,
                heads=dict(efs=dict(module='fairchem.core.models.uma.escn_md.MLP_EFS_Head')), tasks_list=tasks)

    def loader(source, shuffle):
        return {'_target_': 'fairchem.core.components.common.dataloader_builder.get_dataloader', 'dataset': source,
                'batch_sampler_fn': {'_target_': 'fairchem.core.common.data_parallel.BalancedBatchSampler',
                                     '_partial_': True, 'batch_size': '${batch_size}', 'shuffle': shuffle, 'seed': seed},
                'num_workers': 0,
                'collate_fn': {'_target_': 'fairchem.core.units.mlip_unit.mlip_unit.mt_collater_adapter',
                               'tasks': '${data.tasks_list}'}}

    def concat(key):
        return {'_target_': 'fairchem.core.datasets.mt_concat_dataset.create_concat_dataset',
                'dataset_configs': {task: '${data.%s}' % key},
                'combined_dataset_config': {'sampling': {'type': 'temperature', 'temperature': 1.0}}}

    unit = {'_target_': 'fairchem.core.units.mlip_unit.mlip_unit.MLIPTrainEvalUnit', 'job_config': '${job}',
            'tasks': '${data.tasks_list}',
            'model': {'_target_': 'fairchem.core.units.mlip_unit.mlip_unit.initialize_finetuning_model',
                      # A local file, where fairchem's template downloads by name.
                      'checkpoint_location': str(settings['foundation_model']),
                      'overrides': {'backbone': {'otf_graph': True, 'max_neighbors': '${max_neighbors}',
                                                 'regress_stress': '${data.regress_stress}', 'always_use_pbc': False},
                                    'pass_through_head_outputs': '${data.pass_through_head_outputs}'},
                      'heads': '${data.heads}'},
            'optimizer_fn': {'_target_': 'torch.optim.AdamW', '_partial_': True, 'lr': '${lr}',
                             'weight_decay': '${weight_decay}'},
            'cosine_lr_scheduler_fn': {'_target_': 'fairchem.core.units.mlip_unit.mlip_unit._get_consine_lr_scheduler',
                                       '_partial_': True, 'warmup_factor': 0.2, 'warmup_epochs': 0.01,
                                       'lr_min_factor': 0.01, 'epochs': '${epochs}', 'steps': '${steps}'},
            'print_every': 10, 'clip_grad_norm': 100}
    return {
        'defaults': ['_self_'],
        'job': {'device_type': DEVICE_TYPES[device], 'scheduler': {'mode': 'LOCAL', 'ranks_per_node': 1, 'num_nodes': 1},
                'debug': True, 'logger': None, 'run_dir': str(run_dir), 'run_name': 'copy', 'timestamp_id': 'copy',
                'seed': seed},
        'data': data, 'max_neighbors': settings['max_neighbors'], 'epochs': settings['epochs'], 'steps': settings['steps'],
        'batch_size': settings['batch_size'], 'lr': settings['lr'], 'weight_decay': settings['weight_decay'],
        'evaluate_every_n_steps': 100, 'checkpoint_every_n_steps': 1000,
        'train_dataset': concat('train_dataset'), 'train_dataloader': loader('${train_dataset}', True),
        'val_dataset': concat('val_dataset'), 'eval_dataloader': loader('${val_dataset}', False),
        'runner': {'_target_': 'fairchem.core.components.train.train_runner.TrainEvalRunner',
                   'train_dataloader': '${train_dataloader}', 'eval_dataloader': '${eval_dataloader}',
                   'train_eval_unit': unit, 'max_epochs': '${epochs}', 'max_steps': '${steps}',
                   'evaluate_every_n_steps': '${evaluate_every_n_steps}',
                   'callbacks': [{'_target_': 'fairchem.core.components.callbacks.TrainCheckpointCallback',
                                  'checkpoint_every_n_steps': '${checkpoint_every_n_steps}', 'max_saved_checkpoints': 5}]}}


def write_database(directory, frames):
    """Write frames as fairchem's ASE-LMDB, with the atom counts its samplers read."""
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.db import connect

    from alframework.ml_interfaces.fine_tuning import ENERGY_KEY, FORCES_KEY
    directory.mkdir(parents=True, exist_ok=True)
    with connect(str(directory / 'data.0000.aselmdb')) as database:
        for atoms in frames:
            atoms = atoms.copy()
            energy, forces = atoms.info.pop(ENERGY_KEY), atoms.arrays.pop(FORCES_KEY)
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.info.setdefault('charge', 0)
            atoms.info.setdefault('spin', 1)
            database.write(atoms, data=atoms.info)
    np.savez_compressed(directory / 'metadata.npz', natoms=[len(atoms) for atoms in frames])


def fine_tune_uma_member(arguments):
    """Fine-tune one copy. ``arguments`` is what ``UmaFineTuner.train_member`` writes."""
    import argparse

    from ase.calculators.singlepoint import SinglePointCalculator
    from fairchem.core._cli import main

    from alframework.ml_interfaces.fine_tuning import ENERGY_KEY, FORCES_KEY, shard_atoms

    settings, seed, member_dir = arguments['settings'], int(arguments['seed']), Path(arguments['member_dir'])
    frames = shard_atoms(arguments['shards'], settings['energy_dataset'], settings['forces_dataset'])
    train, valid = split(frames, settings['valid_fraction'], seed)
    write_database(member_dir / 'data' / 'train', train)
    write_database(member_dir / 'data' / 'valid', valid)
    if settings['E0s'] == 'average':
        labeled = []
        for atoms in train:
            copy = atoms.copy()
            copy.calc = SinglePointCalculator(copy, energy=atoms.info[ENERGY_KEY], forces=atoms.arrays[FORCES_KEY])
            labeled.append(copy)
        references, rmsd = fitted_references(labeled)
    else:
        references, rmsd = checkpoint_references(settings['foundation_model'], settings['task'])
    config = training_config(settings, member_dir / 'data' / 'train', member_dir / 'data' / 'valid', references, rmsd,
                             member_dir / 'runs', seed, arguments['device'])
    # JSON is valid YAML, so Hydra reads it.
    (member_dir / CONFIG).write_text(json.dumps(config, indent=1) + '\n')
    print(f"Fine-tuning UMA task {settings['task']} from {settings['foundation_model']}: {len(train)} training and "
          f"{len(valid)} validation structures, seed {seed}, E0s {settings['E0s']}.", flush=True)
    main(argparse.Namespace(config=str(member_dir / CONFIG)), [])
    final = member_dir / 'runs' / 'copy' / 'checkpoints' / 'final' / 'inference_ckpt.pt'
    if not final.is_file():
        raise RuntimeError(f'fairchem finished without writing {final}.')
    shutil.copyfile(final, member_dir / MODEL_NAME)
    if not settings['keep_training_checkpoints']:
        # Resume checkpoints hold optimizer state and several copies of the weights; the copy is in uma.pt.
        shutil.rmtree(member_dir / 'runs' / 'copy' / 'checkpoints')


class UmaBatchedEnsemble:
    """Factory for every fine-tuned UMA copy, evaluated on a whole batch through fairchem's TorchSim model."""

    def __new__(cls, model_directory, task, device='cuda:0', dtype='float32', charge=None, spin=None):
        if task not in TASKS:
            raise ValueError(f'load_uma_batched_ensemble needs the task the copies were fine-tuned on: one of {TASKS}.')
        import torch
        from fairchem.core.calculate.torchsim_interface import FairChemModel

        from alframework.ml_interfaces.torchsim_ensemble import TorchSimBatchedEnsemble
        if task == 'omol':
            # ALF's engines pass no charge or spin per structure; without them, fairchem would use spin 0.
            charge = 0 if charge is None else charge
            spin = 1 if spin is None else spin
        precision = torch.float64 if dtype == 'float64' else torch.float32
        models = [FairChemModel(str(path), task_name=task, device=torch.device(device), dtype=precision)
                  for path in sorted(Path(model_directory).glob(f'model-*/{MODEL_NAME}'))]
        return TorchSimBatchedEnsemble(models, device=device, dtype=dtype, charge=charge, spin=spin,
                                       source=model_directory)


def load_uma_batched_ensemble(model_directory, task, device='cuda:0', dtype='float32', charge=None, spin=None):
    """Load fine-tuned UMA copies for batched sampling (``batched_calculator`` loader).

    ``task`` must be the one they were fine-tuned on. For ``omol``, ``charge`` and ``spin`` apply to
    every structure sampled, and default to a neutral singlet.
    """
    return UmaBatchedEnsemble(model_directory, task, device=device, dtype=dtype, charge=charge, spin=spin)


if __name__ == '__main__':
    fine_tune_uma_member(json.loads(Path(sys.argv[1]).read_text()))
