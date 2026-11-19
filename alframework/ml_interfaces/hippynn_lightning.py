"""Train one HIPPYNN model on several devices, on one node or several, with PyTorch Lightning data parallelism.

``train_data_parallel`` stands in for HIPPYNN's ``setup_and_train``, or ``setup_training`` followed by
``train_model``, when an ensemble member uses more than one GPU. It writes the files those write, so
ALF's loaders read a member trained this way unchanged: ``experiment_structure.pt``, ``best_model.pt``,
``best_checkpoint.pt``, ``training_metrics.pt``, and ``Training complete.`` in ``training_log.txt``, which
also gets the per-epoch output.
It also writes ``lightning_training.json``, which records the devices and batch sizes used.

The configured batch sizes are totals across devices: batch 64 on 4 GPUs gives each GPU 16. Each
device trains on its own slice of the training split every epoch, and validation losses are averaged
over the devices, so every rank makes the same early-stopping and batch-size decisions. Afterwards,
the final test evaluates the best model on every split, in the same batches as single-device training;
each device predicts every n-th batch, and the parent computes the metrics over all of them, so they
are the ones a single device would report.

ALF starts one process per device itself, with ``spawn``, and sets the rank variables so Lightning
treats them as launched externally. A member on several nodes (``nodes`` above 1) runs a launcher
command, such as ``srun``, that starts one process on each node, ``python -m
alframework.ml_interfaces.hippynn_lightning``; each starts its node's device processes, and they meet
through a small file in the member's directory, which must be on storage every node shares. Lightning's own launchers either rerun ``sys.argv``, which inside a
Parsl worker would start another worker, or pass the model and data to each process through shared
memory, which fails on Linux for empty tensors. Each process reads the setup from a file instead. Import this
module only after the member has chosen its devices, since it imports Torch.
"""
import contextlib
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    import pytorch_lightning as pl
    from pytorch_lightning.strategies import DDPStrategy
    from hippynn.experiment.lightning_trainer import HippynnLightningModule
except ImportError as _missing:
    LIGHTNING_MISSING = _missing
    HippynnLightningModule = object
else:
    LIGHTNING_MISSING = None

RECORD = 'lightning_training.json'
# Rank 0's output; the parent copies it into its own log after training, then removes it.
RANK_LOG = 'lightning_rank0_log.txt'


def require_lightning():
    if LIGHTNING_MISSING is not None:
        raise ImportError("gpus_per_member above 1 trains with PyTorch Lightning: install ALF's 'lightning' extra "
                          "(pytorch-lightning) in the ML worker environment.") from LIGHTNING_MISSING


def _batch_schedulers(controller):
    return [s for s in controller.scheduler_list if hasattr(s, 'max_batch_size')]


def _scale_batches(controller, scale):
    """Apply ``scale`` to the controller's batch sizes and any batch-raising scheduler's limit."""
    controller.batch_size = scale(controller.batch_size)
    controller.eval_batch_size = scale(controller.eval_batch_size)
    for scheduler in _batch_schedulers(controller):
        scheduler.max_batch_size = scale(scheduler.max_batch_size)


def _split_batches(controller, n_devices):
    names = [('batch_size', controller.batch_size), ('eval_batch_size', controller.eval_batch_size)]
    names += [('scheduler max_batch_size', s.max_batch_size) for s in _batch_schedulers(controller)]
    for name, value in names:
        if value % n_devices:
            raise ValueError(f'{name} {value} is a total over {n_devices} devices, so it must divide evenly by '
                             f'gpus_per_member.')
    _scale_batches(controller, lambda size: size // n_devices)


class AlfLightningModule(HippynnLightningModule):
    """HIPPYNN's Lightning module, also writing the files HIPPYNN's own training loop writes."""

    def _eval_end(self, prefix, when=None):
        # The parent class registers metrics without keeping the best model, so catch its result.
        tracker, found = self.metric_tracker, {}
        register = tracker.register_metrics

        def recording(metric_info, when):
            result = register(metric_info, when)
            found['better'] = result[1]
            return result

        tracker.register_metrics = recording
        try:
            super()._eval_end(prefix, when)
        finally:
            del tracker.register_metrics
        if found.get('better') and self.global_rank == 0:
            self._save_best()

    def _save_best(self):
        import copy
        import torch
        from alframework.ml_interfaces._hippynn_compat import checkpoints
        serialization = checkpoints()

        best = copy.deepcopy(self.model.state_dict())
        self.metric_tracker.best_model = best
        torch.save(best, 'best_model.pt')
        state = serialization.create_state(self.model, self.controller, self.metric_tracker)
        # Lightning holds the optimizer and the per-device batch sizes; store the totals, as configured.
        state['controller']['optimizer'] = self.optimizer_list[0].state_dict()
        for key in ('batch_size', 'eval_batch_size'):
            state['controller'][key] *= self.n_devices
        torch.save(state, 'best_checkpoint.pt')
        self.print('**** NEW BEST MODEL - Saving! ****')

    def setup(self, stage):
        # Each process starts fresh, so repeat the parent's settings before anything runs.
        import hippynn
        import torch
        hippynn.custom_kernels.set_custom_kernels(self.parent_settings['custom_kernels'])
        hippynn.settings.WARN_LOW_DISTANCES = self.parent_settings['warn_low_distances']
        torch.backends.cuda.matmul.allow_tf32 = self.parent_settings['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = self.parent_settings['cudnn_tf32']
        os.chdir(self.parent_settings['directory'])
        self._log = contextlib.ExitStack()
        if self.global_rank == 0:
            self._log.enter_context(hippynn.tools.log_terminal(RANK_LOG, 'wt'))

    def teardown(self, stage):
        self._log.close()

    def on_train_epoch_start(self):
        self._epoch_started = time.perf_counter()
        super().on_train_epoch_start()

    def on_train_epoch_end(self):
        self.metric_tracker.epoch_times.append(time.perf_counter() - self._epoch_started)
        self.print('Total epoch time:', round(self.metric_tracker.epoch_times[-1], 2), 's')
        stats = getattr(self, 'rank_stats', None)
        if stats is not None and stats.enabled:
            stats.record.setdefault('epoch_seconds', []).append(self.metric_tracker.epoch_times[-1])

    # With ALF_LIGHTNING_STATS=1, time each optimizer step on every rank, synchronizing the device so the
    # time includes the gradient all-reduce; ranks that wait on a slower one show it here.
    def on_train_batch_start(self, batch, batch_idx):
        stats = getattr(self, 'rank_stats', None)
        if stats is not None:
            stats.step_start()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        stats = getattr(self, 'rank_stats', None)
        if stats is not None and stats.enabled:
            import torch
            stats.step_end(torch.cuda.synchronize if self.device.type == 'cuda' else lambda: None)

    def on_train_end(self):
        if self.global_rank != 0:
            return
        import torch
        torch.save(self.metric_tracker, 'training_metrics.pt')
        record = dict(devices=self.n_devices, nodes=self.parent_settings.get('nodes', 1),
                      accelerator=self.parent_settings['accelerator'],
                      epochs=int(self.controller.current_epoch),
                      batch_size_per_device=int(self.controller.batch_size),
                      batch_size=int(self.controller.batch_size) * self.n_devices,
                      eval_batch_size_per_device=int(self.controller.eval_batch_size),
                      eval_batch_size=int(self.controller.eval_batch_size) * self.n_devices)
        Path(RECORD).write_text(json.dumps(record, indent=2) + '\n')


def train_data_parallel(training_modules, database, controller, *, n_devices, accelerator, seed, nodes=1,
                        launcher=None):
    """Train ``training_modules`` on ``n_devices`` devices, in the current directory; return the metric tracker.

    ``accelerator`` is ``'gpu'`` or ``'cpu'``; CPU runs use the gloo backend and exist for testing.
    ``n_devices`` is the total over ``nodes`` nodes, which share it evenly. With ``nodes`` above 1,
    ``launcher`` is the command, as a list of words, that starts one process on each of those nodes,
    each seeing its node's share of the devices (see ``node_rank``).
    """
    require_lightning()
    import hippynn
    import torch
    from hippynn.experiment import SetupParams, setup_training
    from alframework.ml_interfaces._hippynn_compat import checkpoints
    serialization = checkpoints()

    local_devices = _local_devices(n_devices, nodes, launcher)
    if accelerator == 'gpu' and torch.cuda.device_count() < local_devices:
        raise RuntimeError(f'gpus_per_member is {n_devices} on {nodes} node(s), {local_devices} per node, but this '
                           f'worker sees {torch.cuda.device_count()} GPUs. Give each ML worker that many GPUs; '
                           'see the Parsl guide.')
    serialization.create_structure_file(training_modules, database, controller)
    _split_batches(controller, n_devices)
    training_modules, controller, metric_tracker = setup_training(
        training_modules, SetupParams(controller=controller, device='cpu'))
    settings = dict(
        custom_kernels=hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE or False,
        warn_low_distances=hippynn.settings.WARN_LOW_DISTANCES,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        directory=os.getcwd(), accelerator=accelerator, n_devices=n_devices, nodes=nodes, seed=seed)
    where = f' on {nodes} nodes, {local_devices} each' if nodes > 1 else ''
    print(f'Training with PyTorch Lightning on {n_devices} {accelerator} devices{where}: batch size '
          f'{controller.batch_size} per device, {controller.batch_size * n_devices} in total.')
    sys.stdout.flush()
    # The ranks read the setup from a file. Passing it as process arguments would share its tensors
    # through shared memory, which fails on Linux for empty tensors and can exhaust /dev/shm.
    splits = _share_splits(database) if os.environ.get('ALF_LIGHTNING_SHARED_DATA', '') not in ('', '0') else None
    try:
        torch.save(dict(training_modules=training_modules, database=database, controller=controller,
                        metric_tracker=metric_tracker, settings=settings), PAYLOAD)
    finally:
        if splits is not None:
            database.splits = splits
    Path(RECORD).unlink(missing_ok=True)  # the ranks' final test waits for this run's record
    try:
        if nodes == 1:
            _run_ranks(n_devices)
        else:
            _launch_nodes(launcher, nodes, local_devices)
    finally:
        Path(PAYLOAD).unlink(missing_ok=True)
        shutil.rmtree(SHARED, ignore_errors=True)
    if Path(RANK_LOG).exists():
        print(Path(RANK_LOG).read_text(), end='')
        Path(RANK_LOG).unlink()

    # Finish as HIPPYNN's train_model does, from the files rank 0 wrote.
    if not Path('best_model.pt').exists() or not Path(RECORD).exists():
        raise RuntimeError('Lightning training ended without storing a best model')
    print('Training phase ended.')
    record = json.loads(Path(RECORD).read_text())
    metric_tracker = torch.load('training_metrics.pt', weights_only=False)
    model = training_modules.model
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu', weights_only=True))
    print('Reverting to best model found.')
    _scale_batches(controller, lambda size: size * n_devices)
    controller.current_epoch = record['epochs']
    print('Making plots over training time...')
    metric_tracker.plot_over_time()
    # The ranks evaluated the best model in batches of one device's share of the evaluation batch, the size
    # each evaluated during training; the configured total can exceed one device's memory.
    print(f"Testing model on {n_devices} devices, eval batch size {record['eval_batch_size_per_device']} each...")
    _final_test(training_modules.evaluator, database, n_devices, metric_tracker)
    print('Training complete.')
    return metric_tracker


PAYLOAD = 'lightning_setup.pt'
# Where the nodes of a multi-node member find the first node's address and port.
RENDEZVOUS = 'lightning_rendezvous.json'
# Variables that can give a launched node process its node number, first found wins: ALF's own, then those
# set by srun, MPI launchers and torchrun. The launcher must start exactly one process per node.
NODE_RANK_VARIABLES = ('ALF_NODE_RANK', 'SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'GROUP_RANK')
# With ALF_LIGHTNING_SHARED_DATA=1 the parent writes the database's split arrays here, one .npy file each,
# instead of into the setup file, and every rank maps them read-only: one copy in each node's memory, not
# one per rank, and the same arrays and sampling as before. With ALF_NODE_LOCAL_DIR set, each node first
# copies them there.
SHARED = 'lightning_data'
# Opt-in per-rank timing and memory records, written to this directory: set ALF_LIGHTNING_STATS=1.
STATS = 'lightning_stats'


# Each rank's share of the final test, read back and removed by the parent.
FINAL_SHARE = 'final_test_rank{}.pt'


def _align(database, db_info):
    """Order the database's inputs and targets as the model expects, as hippynn's ``test_model`` does first.

    hippynn's development branch has ``Database.align``; release 0.1.3 checks and assigns them inline.
    """
    if hasattr(database, 'align'):
        database.align(**db_info)
        return
    for kind in ('inputs', 'targets'):
        if set(getattr(database, kind)) != set(db_info[kind]):
            raise ValueError(f"Database and model have different {kind}: "
                             f"database has {set(getattr(database, kind))}, model requires {set(db_info[kind])}")
        setattr(database, kind, db_info[kind])


def _final_splits(database, evaluator):
    """The splits hippynn's ``test_model`` evaluates, in its order: train, valid and test first."""
    _align(database, evaluator.db_info)
    required = set(database.inputs + database.targets)
    names = [name for name, split in database.splits.items() if all(key in split for key in required)]
    basic = [name for name in ('train', 'valid', 'test') if name in names]
    return basic + [name for name in names if name not in basic]


def _evaluate_share(module, database, evaluator, rank, n_devices, accelerator, local_rank=None):
    """This rank's share of the final test: the best model's predictions for every ``n_devices``-th batch of
    hippynn's evaluation batches, numbered so the parent can put them back in order. On the rank's own device,
    the one it trained on (Lightning may have moved the model back to the CPU at the end of training)."""
    import torch
    # Rank 0 writes best_model.pt during training and the record last, at the end of training.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        while not Path(RECORD).exists():
            time.sleep(0.5)
    local_rank = rank if local_rank is None else local_rank
    device = torch.device('cuda', local_rank) if accelerator == 'gpu' else torch.device('cpu')
    model = module.model.to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True))
    model.eval()
    batch_size = int(module.controller.eval_batch_size)
    shares = {}
    for name in _final_splits(database, evaluator):
        shares[name] = []
        for index, batch in enumerate(database.make_generator(name, 'eval', batch_size)):
            if index % n_devices != rank:
                continue
            batch = [item.to(device=device) for item in batch]
            predictions = model(*batch[:evaluator.n_inputs])
            shares[name].append((index, [value.detach().cpu() for value in predictions],
                                 [value.detach().cpu() for value in batch[-evaluator.n_targets:]]))
            del predictions
    torch.save(shares, FINAL_SHARE.format(rank))


def _final_test(evaluator, database, n_devices, metric_tracker, when='FinalTraining'):
    """hippynn's ``test_model`` from the ranks' shares: the same batches, put back in order, then hippynn's
    evaluation losses, plots and printout over all of them, as ``Evaluator.evaluate`` computes them."""
    import torch
    shares = [torch.load(FINAL_SHARE.format(rank), weights_only=False) for rank in range(n_devices)]
    metrics = {}
    for name in _final_splits(database, evaluator):
        batches = sorted((batch for share in shares for batch in share[name]), key=lambda batch: batch[0])
        predictions = [[batch[1][i] for batch in batches] for i in range(evaluator.n_outputs)]
        targets = [[batch[2][i] for batch in batches] for i in range(evaluator.n_targets)]
        predictions = [torch.cat(x, dim=0) if x[0].shape != () else x[0] for x in predictions]
        targets = [torch.cat(x, dim=0) for x in targets]
        losses = [x.mean().item() for x in evaluator.loss(*predictions, *targets)]
        metrics[name] = dict(zip(evaluator.loss_names, losses))
        if evaluator.plot_maker:
            evaluator.plot_maker.plot_phase(prediction_all_vals=predictions, target_all_vals=targets, when=when,
                                            eval_type=name)
    for rank in range(n_devices):
        Path(FINAL_SHARE.format(rank)).unlink(missing_ok=True)
    metric_tracker.register_metrics(metrics, when=when)
    metric_tracker.evaluation_print(metrics, quiet=False)
    return metrics


def _local_devices(n_devices, nodes, launcher):
    """Each node's share of a member's ``n_devices``, after checking ``nodes`` and ``launcher``."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError(f'member_nodes must be a positive integer, not {nodes!r}.')
    if n_devices % nodes:
        raise ValueError(f'gpus_per_member {n_devices} is a total over member_nodes {nodes}, so it must divide '
                         'evenly by it.')
    if nodes > 1 and (not launcher or isinstance(launcher, str)):
        raise ValueError('A member on several nodes needs member_launcher: the command, as a list of words, that '
                         'starts one process on each of its nodes, such as srun with one task per node.')
    return n_devices // nodes


def _share_splits(database):
    """Write the database's split arrays to ``SHARED``, leave the database without them, and return them."""
    import numpy as np
    shutil.rmtree(SHARED, ignore_errors=True)
    Path(SHARED).mkdir()
    layout = {}
    for name, split in database.splits.items():
        layout[name] = {}
        for key, tensor in split.items():
            layout[name][key] = file = f'{name}.{key}.npy'
            np.save(Path(SHARED) / file, tensor.detach().cpu().numpy())
    (Path(SHARED) / 'layout.json').write_text(json.dumps(layout))
    splits, database.splits = database.splits, {}
    return splits


def _attach_splits(database, directory):
    """Map the split arrays ``_share_splits`` wrote into ``database``, copy-on-write so they stay shared."""
    import numpy as np
    import torch
    layout = json.loads((Path(directory) / 'layout.json').read_text())
    database.splits = {name: {key: torch.from_numpy(np.load(Path(directory) / file, mmap_mode='c'))
                              for key, file in split.items()} for name, split in layout.items()}


@contextlib.contextmanager
def _node_data(payload):
    """The directory this node's ranks read shared split arrays from, or None if the setup holds them;
    a node-local copy, removed afterwards, when ALF_NODE_LOCAL_DIR is set."""
    shared = Path(payload).parent / SHARED
    local_root = os.environ.get('ALF_NODE_LOCAL_DIR')
    if not shared.exists():
        yield None
    elif not local_root:
        yield str(shared)
    else:
        local = Path(local_root) / f'alf-lightning-data-{os.getpid()}'
        try:
            started = time.perf_counter()
            shutil.copytree(shared, local)
            print(f'Copied the training arrays to {local} in {time.perf_counter() - started:.1f} s')
            sys.stdout.flush()
            yield str(local)
        finally:
            shutil.rmtree(local, ignore_errors=True)


def _free_port(host='127.0.0.1'):
    import socket
    with socket.socket() as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def node_rank(environ=None):
    """This node's number among a member's nodes, from the first of ``NODE_RANK_VARIABLES`` that is set."""
    environ = os.environ if environ is None else environ
    for name in NODE_RANK_VARIABLES:
        if environ.get(name, '') != '':
            return int(environ[name])
    raise RuntimeError('A multi-node member process needs its node number: the launcher must set one of '
                       + ', '.join(NODE_RANK_VARIABLES) + '.')


def _launch_nodes(launcher, nodes, local_devices):
    """Run ``launcher`` to start one node process on each of the member's nodes; wait for all of them."""
    import secrets
    import subprocess
    rendezvous = os.path.abspath(RENDEZVOUS)
    Path(rendezvous).unlink(missing_ok=True)
    command = [str(word) for word in launcher] + [
        sys.executable, '-m', 'alframework.ml_interfaces.hippynn_lightning', '--payload', os.path.abspath(PAYLOAD),
        '--nodes', str(nodes), '--local-devices', str(local_devices), '--rendezvous', rendezvous,
        '--token', secrets.token_hex(8)]
    # The node processes import the ALF this process runs, wherever the environment's own copy is.
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [source, environment.get('PYTHONPATH')]))
    print('Starting the member on', nodes, 'nodes:', ' '.join(command))
    sys.stdout.flush()
    try:
        finished = subprocess.run(command, env=environment)
    finally:
        Path(rendezvous).unlink(missing_ok=True)
    if finished.returncode != 0:
        raise RuntimeError(f'Lightning training on {nodes} nodes failed (launcher exit code {finished.returncode}); '
                           'the failing rank printed its error to the worker output above.')


def _meet(rendezvous, token, rank, timeout=None):
    """The first node's address and port: node 0 chooses and publishes them, the other nodes wait for them."""
    import socket
    timeout = float(os.environ.get('ALF_RENDEZVOUS_TIMEOUT', 900)) if timeout is None else timeout
    if rank == 0:
        address = os.environ.get('ALF_MASTER_ADDR') or socket.gethostname()
        found = dict(token=token, address=address, port=_free_port(''))
        temporary = f'{rendezvous}.{os.getpid()}'
        Path(temporary).write_text(json.dumps(found))
        os.replace(temporary, rendezvous)
        return found['address'], found['port']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(FileNotFoundError, ValueError):
            found = json.loads(Path(rendezvous).read_text())
            if found.get('token') == token:
                return found['address'], int(found['port'])
        time.sleep(0.5)
    raise TimeoutError(f'node {rank} waited {timeout:.0f} s for node 0 to write {rendezvous}')


def _run_ranks(local_devices, node=0, nodes=1, address='127.0.0.1', port=None):
    """Start one process per device of this node and wait for all of them; if one fails, stop the rest."""
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    port = _free_port() if port is None else port
    with _node_data(os.path.abspath(PAYLOAD)) as data:
        _start_and_wait(context, local_devices, node, nodes, address, port, data)


def _start_and_wait(context, local_devices, node, nodes, address, port, data):
    ranks = [context.Process(target=_rank_main,
                             args=(os.path.abspath(PAYLOAD), local, local_devices, node, nodes, address, port, data),
                             name=f'alf-lightning-rank-{node * local_devices + local}')
             for local in range(local_devices)]
    for process in ranks:
        process.start()
    try:
        while any(process.is_alive() for process in ranks):
            if any(process.exitcode not in (None, 0) for process in ranks):
                break
            time.sleep(1)
    finally:
        for process in ranks:
            if process.is_alive():
                process.terminate()
            process.join()
    failed = {node * local_devices + local: process.exitcode for local, process in enumerate(ranks)
              if process.exitcode != 0}
    if failed:
        # A negative exit code is the signal that ended the rank (-15: stopped here after another rank failed).
        codes = ', '.join(f'rank {rank} exit code {code}' for rank, code in failed.items())
        raise RuntimeError(f'Lightning training failed ({codes}); the failing rank printed its error to the '
                           'worker output above.')


def _pss_gib():
    """This process's proportional set size (shared pages split among the processes sharing them), if Linux
    reports it."""
    with contextlib.suppress(OSError, ValueError):
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
            if line.startswith('Pss:'):
                return int(line.split()[1]) / 2**20
    return None


class _RankStats:
    """With ALF_LIGHTNING_STATS=1, one rank's timings and peak memory, written as JSON when it finishes."""

    def __init__(self, rank, local_rank, node, world):
        self.enabled = os.environ.get('ALF_LIGHTNING_STATS', '') not in ('', '0')
        self.started = time.perf_counter()
        self.record = dict(rank=rank, local_rank=local_rank, node=node, world=world, steps=[], seconds={})
        self._mark = None

    def phase(self, name):
        if self.enabled:
            self.record['seconds'][name] = time.perf_counter() - self.started

    def step_start(self):
        if self.enabled:
            self._mark = time.perf_counter()

    def step_end(self, synchronize):
        if self.enabled and self._mark is not None:
            synchronize()
            self.record['steps'].append(time.perf_counter() - self._mark)

    def write(self, directory, device):
        if not self.enabled:
            return
        import resource
        import socket
        import torch
        steps = sorted(self.record.pop('steps'))
        pick = lambda q: steps[min(len(steps) - 1, int(q * len(steps)))] if steps else None
        self.record.update(host=socket.gethostname(), step_count=len(steps), step_total=sum(steps),
                           step_median=pick(0.5), step_p90=pick(0.9), step_max=steps[-1] if steps else None,
                           peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                           final_pss_gib=_pss_gib(),
                           gpu_peak_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                                         if device.type == 'cuda' else None))
        folder = Path(directory) / STATS
        folder.mkdir(exist_ok=True)
        (folder / f'rank-{self.record["rank"]:04d}.json').write_text(json.dumps(self.record, indent=1) + '\n')


def _rank_main(payload, local_rank, local_devices, node, nodes, address, port, data=None):
    """One device's training process. The rank variables make Lightning treat it as launched externally."""
    world, rank = local_devices * nodes, node * local_devices + local_rank
    os.environ.update(MASTER_ADDR=address, MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(local_rank), NODE_RANK=str(node))
    stats = _RankStats(rank, local_rank, node, world)
    import torch
    setup = torch.load(payload, weights_only=False)
    if data is not None:
        _attach_splits(setup['database'], data)
    stats.phase('payload_loaded')
    settings = setup['settings']
    module, datamodule = AlfLightningModule.from_train_setup(setup['training_modules'], setup['database'],
                                                             setup['controller'], setup['metric_tracker'])
    module.n_devices, module.parent_settings, module.rank_stats = world, settings, stats
    pl.seed_everything(settings['seed'], workers=True, verbose=False)
    accelerator = settings['accelerator']
    # Name the environment: inside a Slurm step (Parsl's SrunLauncher), Lightning would otherwise pick its
    # Slurm environment, read every rank as SLURM_PROCID 0, and have all of them bind MASTER_PORT.
    from pytorch_lightning.plugins.environments import LightningEnvironment
    trainer = pl.Trainer(accelerator=accelerator, devices=local_devices, num_nodes=nodes,
                         plugins=[LightningEnvironment()],
                         strategy=DDPStrategy(process_group_backend='nccl' if accelerator == 'gpu' else 'gloo'),
                         max_epochs=setup['controller'].max_epochs, use_distributed_sampler=True,
                         num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, default_root_dir=settings['directory'])
    trainer.fit(module, datamodule=datamodule)
    stats.phase('trained')
    _evaluate_share(module, setup['database'], setup['training_modules'].evaluator, rank, world, accelerator,
                    local_rank)
    stats.phase('final_share_written')
    device = torch.device('cuda', local_rank) if accelerator == 'gpu' else torch.device('cpu')
    stats.write(settings['directory'], device)


def _node_main(argv=None):
    """One node's process of a multi-node member: meet the other nodes, then train on this node's devices."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--payload', required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--local-devices', type=int, required=True)
    parser.add_argument('--rendezvous', required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args(argv)
    # A launcher stopping this node after another failed sends SIGTERM: exit through _run_ranks's cleanup, so
    # this node's device processes stop too instead of waiting on the others.
    import signal
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    node = node_rank()
    if not 0 <= node < args.nodes:
        raise RuntimeError(f'node number {node} is outside 0..{args.nodes - 1}: the launcher must start exactly one '
                           'process per node.')
    address, port = _meet(args.rendezvous, args.token, node)
    _run_ranks(args.local_devices, node, args.nodes, address, port)


if __name__ == '__main__':
    _node_main()
