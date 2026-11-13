"""Fine-tune a pretrained HIPPYNN model on ALF's labels, through ``fine_tuning.fine_tune_ensemble_task``.

Each copy loads the pretrained graph and weights (``restart_db=False``), trains them with a fresh
optimizer on every stored shard, and is written in HIPPYNN's usual layout, so every ALF loader for
HIPPYNN ensembles reads it unchanged. The source is one model directory, which every copy starts
from, or an ensemble directory of ``model-*`` members, where copy ``i`` starts from member ``i``.

Training runs in a child process (``python -m alframework.ml_interfaces.fine_tuning.hippynn``), so
each copy sees only its own GPU, as for MACE. Nothing here imports Torch until training starts.
"""
import json
import subprocess
import sys
from pathlib import Path

E0_CHOICES = ('foundation', 'average')
ARGUMENTS = 'fine_tune_args.json'
LOG = 'fine_tune.log'
DEFAULTS = dict(learning_rate=1e-4, scheduler_options={}, controller_options={}, stopping_key=None,
                valid_size=0.1, test_size=0.1, h5_test_dir=None, freeze=[], energy_key='energy',
                force_key='forces', species_key='species', coordinates_key='coordinates', quadrupole_key=None,
                remove_high_energy_cut=None, remove_high_energy_std=None, remove_high_forces_cut=None,
                remove_high_forces_std=None, structure_file='experiment_structure.pt',
                checkpoint_file='best_checkpoint.pt')
CONTROLLER_DEFAULTS = dict(batch_size=32, eval_batch_size=64, max_epochs=200, termination_patience=51)
SCHEDULER_DEFAULTS = dict(max_batch_size=64, patience=20, factor=0.5)


class HippynnFineTuner:
    """Fine-tune a pretrained HIPPYNN model or ensemble. Settings come from ``fine_tune``.

    ``E0s`` has no default: ``"foundation"`` keeps the model's atomic energies, for labels at its
    level of theory, and ``"average"`` refits them to the labels, as ALF's trainer fits them.
    ``freeze`` lists graph node names, such as ``"hipnn"``, or parameter-name prefixes, whose
    weights stay fixed.
    """

    def __init__(self, settings):
        settings = {**DEFAULTS, **dict(settings)}
        settings['controller_options'] = {**CONTROLLER_DEFAULTS, **dict(settings['controller_options'] or {})}
        settings['scheduler_options'] = {**SCHEDULER_DEFAULTS, **dict(settings['scheduler_options'] or {})}
        if settings.get('E0s') not in E0_CHOICES:
            raise ValueError(f"fine_tune.E0s is required for HIPPYNN: one of {E0_CHOICES}. 'foundation' keeps the "
                             "model's atomic energies, and 'average' refits them to the labels.")
        rate = settings['learning_rate']
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            raise ValueError(f'fine_tune.learning_rate must be a positive number, not {rate!r}.')
        for key in ('valid_size', 'test_size'):
            if not 0 < settings[key] < 1:
                raise ValueError(f'fine_tune.{key} must be between 0 and 1.')
        freeze = settings['freeze']
        if not isinstance(freeze, list) or not all(isinstance(entry, str) and entry for entry in freeze):
            raise ValueError('fine_tune.freeze must be a list of node names or parameter-name prefixes.')
        self.settings = settings

    def sources(self):
        """The model directories copies start from: one model, or each member of an ensemble."""
        root = Path(self.settings['foundation_model'])
        if (root / self.settings['structure_file']).is_file():
            members = [root]
        else:
            members = sorted(path for path in root.glob('model-*') if (path / self.settings['structure_file']).is_file())
        if not members:
            raise ValueError(f"{root} holds neither a HIPPYNN model ({self.settings['structure_file']}) nor model-* "
                             'members with one.')
        for member in members:
            checkpoint = member / self.settings['checkpoint_file']
            if not checkpoint.is_file():
                raise ValueError(f'{checkpoint} not found; fine-tuning starts from the checkpoint, not best_model.pt.')
        return members

    def source(self, member):
        members = self.sources()
        return members[member % len(members)]

    def validate(self, master_config):
        self.sources()
        names = {spec[0] for spec in (master_config.get('properties_list') or {}).values()}
        for key in ('energy_key', 'force_key'):
            dataset = self.settings[key]
            if names and dataset is not None and dataset not in names:
                raise ValueError(f"fine_tune.{key} is {dataset!r}, which properties_list doesn't store.")
        test_dir = self.settings['h5_test_dir']
        if test_dir is not None and not Path(test_dir).is_dir():
            raise ValueError(f'fine_tune.h5_test_dir not found: {test_dir}')

    def fingerprint(self):
        from alframework.tools.files import cached_sha256
        root = Path(self.settings['foundation_model'])
        return {str(path.relative_to(root)) if path != root else path.name: cached_sha256(path)
                for member in self.sources()
                for path in (member / self.settings['structure_file'], member / self.settings['checkpoint_file'])}

    def identity(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            package_version = version('hippynn')
        except PackageNotFoundError:
            package_version = None
        return dict(package='hippynn', package_version=package_version)

    def prepare_data(self, shards, member_dir):
        """HIPPYNN reads the shards where they are, so nothing is copied."""
        if not shards:
            raise ValueError('There are no labeled shards to fine-tune on.')
        directories = {str(Path(shard).parent) for shard in shards}
        if len(directories) != 1:
            raise ValueError(f'Shards must share one directory, not {sorted(directories)}.')
        return dict(directory=directories.pop(), files=[Path(shard).name for shard in shards])

    def train_member(self, data, member_dir, member, seed, device, environment):
        arguments = dict(source=str(self.source(member)), data=data, member_dir=str(member_dir), seed=seed,
                         device=device, settings=self.settings)
        path = Path(member_dir) / ARGUMENTS
        path.write_text(json.dumps(arguments, indent=2, default=str) + '\n')
        log = Path(member_dir) / LOG
        with open(log, 'w') as handle:
            finished = subprocess.run([sys.executable, '-m', 'alframework.ml_interfaces.fine_tuning.hippynn', str(path)],
                                      cwd=member_dir, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if finished.returncode != 0:
            # HIPPYNN writes its own output, including the traceback, to training_log.txt once training starts.
            training_log = Path(member_dir) / 'training_log.txt'
            source = training_log if training_log.is_file() else log
            tail = source.read_text(errors='replace').strip().splitlines()[-3:]
            raise RuntimeError(f'HIPPYNN fine-tuning in {member_dir} exited with {finished.returncode}: ' + ' | '.join(tail))

    def member_complete(self, member_dir):
        member_dir = Path(member_dir)
        log = member_dir / 'training_log.txt'
        return ((member_dir / 'best_model.pt').is_file() and log.is_file()
                and log.read_text(errors='replace').count('Training complete') == 1)


def trainable_names(model):
    """Parameters training updates. Integer ones, such as HIPPYNN's species map, never are."""
    return [name for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.requires_grad]


def frozen_parameters(model, entries):
    """Names of the trainable parameters ``entries`` select: graph node names, or parameter-name prefixes."""
    names = trainable_names(model)
    by_node = {node.name: f'moddict.{key}.' for node, key in model.names_dict.items()}
    selected = set()
    for entry in entries:
        prefix = by_node.get(entry, entry)
        matches = {name for name in names if name.startswith(prefix)}
        if not matches:
            raise ValueError(f'fine_tune.freeze entry {entry!r} matches no parameters. Nodes with parameters: '
                             f'{sorted(n for n, p in by_node.items() if any(x.startswith(p) for x in names))}.')
        selected |= matches
    return sorted(selected)


def clean_arrays(database, db_info, settings):
    """Give the shards the form the model trains on, as ALF's HIPPYNN trainer does."""
    import torch
    arrays = database.arr_dict
    species, energy = settings['species_key'], settings['energy_key']
    arrays[species] = torch.as_tensor(arrays[species]).to(torch.int64)
    if energy + 'peratom' not in arrays and energy in arrays:
        arrays[energy + 'peratom'] = arrays[energy] / arrays[species].to(torch.bool).to(torch.int32).sum(dim=1)
    quadrupole = settings['quadrupole_key']
    if quadrupole is not None and quadrupole in arrays:
        arrays[quadrupole] = arrays[quadrupole].reshape(-1, 9)
    for key, value in list(arrays.items()):
        if value.dtype == torch.float64:
            arrays[key] = value.to(torch.float32)
    database.inputs, database.targets = db_info['inputs'], db_info['targets']
    missing = [key for key in set(db_info['inputs']) | set(db_info['targets']) if key not in arrays]
    if missing:
        raise ValueError(f'The model needs {sorted(missing)}, which the shards lack. It reads {db_info}.')
    for key in list(arrays):
        if key not in db_info['inputs'] and key not in db_info['targets'] and key != 'indices':
            del arrays[key]
    force = settings['force_key']
    if force is not None and force in arrays:
        database.remove_high_property(force, True, species_key=species, cut=settings['remove_high_forces_cut'],
                                      std_factor=settings['remove_high_forces_std'])
    if energy in arrays:
        database.remove_high_property(energy, False, species_key=species, cut=settings['remove_high_energy_cut'],
                                      std_factor=settings['remove_high_energy_std'])


def split_size(database, fraction):
    """At least one structure, so a small early round still has validation and test splits."""
    return max(1, int(fraction * len(database)))


def fine_tune_hippynn_member(arguments):
    """Fine-tune one copy. ``arguments`` is what ``HippynnFineTuner.train_member`` writes."""
    import numpy as np
    import torch
    from hippynn.databases.database import prettyprint_arrays
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
    from alframework.ml_interfaces._hippynn_compat import checkpoints
    load_checkpoint = checkpoints().load_checkpoint
    from hippynn.graphs import find_unique_relative
    from hippynn.graphs.nodes.targets import HEnergyNode
    from hippynn.tools import active_directory, log_terminal

    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB, use_separate_test_set

    settings, seed = arguments['settings'], int(arguments['seed'])
    source, member_dir = Path(arguments['source']), str(Path(arguments['member_dir']).resolve())
    device = 'cuda:0' if arguments['device'] == 'cuda' else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device != 'cpu':
        torch.cuda.set_device(device)
    Path(member_dir).mkdir(parents=True, exist_ok=True)
    with active_directory(member_dir, create=False):
        with log_terminal('training_log.txt', 'wt'):
            print(f"Fine-tuning from {source} on {arguments['device']}, seed {seed}.")
            bundle = load_checkpoint(str(source / settings['structure_file']), str(source / settings['checkpoint_file']),
                                     restart_db=False, map_location='cpu')
            training_modules = bundle['training_modules']
            db_info = training_modules.evaluator.db_info
            stopping_key = settings['stopping_key'] or bundle['controller'].stopping_key
            losses = list(training_modules.evaluator.loss_names)
            if stopping_key not in losses:
                raise ValueError(f'stopping_key {stopping_key!r} is not one of the model\'s validation losses, {losses}.')

            schema = dict(species_key=settings['species_key'], coordinates_key=settings['coordinates_key'],
                          atom_keys=(settings['coordinates_key'], settings['force_key']))
            data = arguments['data']
            database = ALFPyAniDirectoryDB(directory=data['directory'], files=data['files'], allow_unfound=True,
                                           quiet=False, seed=seed, inputs=None, targets=None, **schema)
            clean_arrays(database, db_info, settings)
            print('Array Shapes After Cleaning')
            prettyprint_arrays(database.arr_dict)
            needed = 2 if settings['h5_test_dir'] is not None else 3
            if len(database) < needed:
                raise ValueError(f'Fine-tuning needs at least {needed} labeled structures for its training, validation, '
                                 f'and test splits, but the shards hold {len(database)} after filtering.')
            database.make_random_split('valid', split_size(database, settings['valid_size']))
            if settings['h5_test_dir'] is not None:
                use_separate_test_set(database, settings['h5_test_dir'], db_info, schema, settings['species_key'],
                                      settings['energy_key'], settings['quadrupole_key'])
            else:
                database.make_random_split('test', split_size(database, settings['test_size']))
                database.split_the_rest('train')

            if settings['E0s'] == 'average':
                from hippynn.pretraining import set_e0_values
                henergy = find_unique_relative(training_modules.model.nodes_to_compute, HEnergyNode,
                                               why_desc='Refitting E0s')
                set_e0_values(henergy, database, peratom=False, energy_name=settings['energy_key'], decay_factor=1e-2)
                print('Refit atomic energies (E0s) to the labels.')

            frozen = set(frozen_parameters(training_modules.model, settings['freeze']))
            names = set(trainable_names(training_modules.model))
            trainable = []
            for name, parameter in training_modules.model.named_parameters():
                if name in frozen:
                    parameter.requires_grad_(False)
                elif name in names:
                    trainable.append(parameter)
            print(f'Frozen parameters: {len(frozen)}; trainable: {len(trainable)}.')
            optimizer = torch.optim.Adam(trainable, lr=settings['learning_rate'])
            scheduler = RaiseBatchSizeOnPlateau(optimizer=optimizer, **settings['scheduler_options'])
            controller_options = dict(settings['controller_options'])
            n_train = len(database.splits['train']['indices'])
            if 'fraction_train_eval' not in controller_options and 0.1 * n_train < 1:
                # HIPPYNN scores 10% of the training split each epoch, which is nothing in a small early round.
                controller_options['fraction_train_eval'] = 1.0
            controller = PatienceController(optimizer=optimizer, scheduler=scheduler, stopping_key=stopping_key,
                                            **controller_options)
            setup_and_train(training_modules=training_modules, database=database,
                            setup_params=SetupParams(controller=controller, device=device))


if __name__ == '__main__':
    fine_tune_hippynn_member(json.loads(Path(sys.argv[1]).read_text()))
