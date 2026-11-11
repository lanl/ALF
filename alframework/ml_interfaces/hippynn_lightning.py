"""Train one HIPPYNN model on several devices of one node, with PyTorch Lightning data parallelism.

``train_data_parallel`` stands in for HIPPYNN's ``setup_and_train``, or ``setup_training`` followed by
``train_model``, when an ensemble member uses more than one GPU. It writes the files those write, so
ALF's loaders read a member trained this way unchanged: ``experiment_structure.pt``, ``best_model.pt``,
``best_checkpoint.pt``, ``training_metrics.pt``, and ``Training complete.`` in ``training_log.txt``, which
also gets the per-epoch output.
It also writes ``lightning_training.json``, which records the devices and batch sizes used.

The configured batch sizes are totals across devices: batch 64 on 4 GPUs gives each GPU 16. Each
device trains on its own slice of the training split every epoch, and validation losses are averaged
over the devices, so every rank makes the same early-stopping and batch-size decisions. The test
split is evaluated afterwards on one device, over the whole split, as in single-device training.

ALF starts one process per device itself, with ``spawn``, and sets the rank variables so Lightning
treats them as launched externally. Lightning's own launchers either rerun ``sys.argv``, which inside a
Parsl worker would start another worker, or pass the model and data to each process through shared
memory, which fails on Linux for empty tensors. Each process reads the setup from a file instead. Import this
module only after the member has chosen its devices, since it imports Torch.
"""
import contextlib
import copy
import json
import os
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
        from hippynn.experiment import serialization

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

    def on_train_end(self):
        if self.global_rank != 0:
            return
        import torch
        torch.save(self.metric_tracker, 'training_metrics.pt')
        record = dict(devices=self.n_devices, accelerator=self.parent_settings['accelerator'],
                      epochs=int(self.controller.current_epoch),
                      batch_size_per_device=int(self.controller.batch_size),
                      batch_size=int(self.controller.batch_size) * self.n_devices,
                      eval_batch_size_per_device=int(self.controller.eval_batch_size),
                      eval_batch_size=int(self.controller.eval_batch_size) * self.n_devices)
        Path(RECORD).write_text(json.dumps(record, indent=2) + '\n')


def train_data_parallel(training_modules, database, controller, *, n_devices, accelerator, seed):
    """Train ``training_modules`` on ``n_devices`` devices, in the current directory; return the metric tracker.

    ``accelerator`` is ``'gpu'`` or ``'cpu'``; CPU runs use the gloo backend and exist for testing.
    """
    require_lightning()
    import hippynn
    import torch
    from hippynn.experiment import SetupParams, serialization, setup_training, test_model

    if accelerator == 'gpu' and torch.cuda.device_count() < n_devices:
        raise RuntimeError(f'gpus_per_member is {n_devices}, but this worker sees {torch.cuda.device_count()} GPUs. '
                           'Give each ML worker that many GPUs; see the Parsl guide.')
    serialization.create_structure_file(training_modules, database, controller)
    _split_batches(controller, n_devices)
    training_modules, controller, metric_tracker = setup_training(
        training_modules, SetupParams(controller=controller, device='cpu'))
    settings = dict(
        custom_kernels=hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE or False,
        warn_low_distances=hippynn.settings.WARN_LOW_DISTANCES,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        directory=os.getcwd(), accelerator=accelerator, n_devices=n_devices, seed=seed)
    print(f'Training with PyTorch Lightning on {n_devices} {accelerator} devices: batch size '
          f'{controller.batch_size} per device, {controller.batch_size * n_devices} in total.')
    sys.stdout.flush()
    # The ranks read the setup from a file. Passing it as process arguments would share its tensors
    # through shared memory, which fails on Linux for empty tensors and can exhaust /dev/shm.
    torch.save(dict(training_modules=training_modules, database=database, controller=controller,
                    metric_tracker=metric_tracker, settings=settings), PAYLOAD)
    try:
        _run_ranks(n_devices)
    finally:
        Path(PAYLOAD).unlink(missing_ok=True)
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
    model.to('cuda' if accelerator == 'gpu' else 'cpu')
    print('Making plots over training time...')
    metric_tracker.plot_over_time()
    print('Testing model...')
    test_model(database, training_modules.evaluator, when='FinalTraining', batch_size=controller.eval_batch_size,
               metric_tracker=metric_tracker)
    print('Training complete.')
    return metric_tracker


PAYLOAD = 'lightning_setup.pt'


def _free_port():
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _run_ranks(n_devices):
    """Start one process per device and wait for all of them; if one fails, stop the rest."""
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    port = _free_port()
    ranks = [context.Process(target=_rank_main, args=(os.path.abspath(PAYLOAD), rank, n_devices, port),
                             name=f'alf-lightning-rank-{rank}') for rank in range(n_devices)]
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
    failed = [rank for rank, process in enumerate(ranks) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f'Lightning training failed on rank(s) {failed}; see the worker output for the error.')


def _rank_main(payload, rank, n_devices, port):
    """One device's training process. The rank variables make Lightning treat it as launched externally."""
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE=str(n_devices),
                      RANK=str(rank), LOCAL_RANK=str(rank), NODE_RANK='0')
    import torch
    setup = torch.load(payload, weights_only=False)
    settings = setup['settings']
    module, datamodule = AlfLightningModule.from_train_setup(setup['training_modules'], setup['database'],
                                                             setup['controller'], setup['metric_tracker'])
    module.n_devices, module.parent_settings = n_devices, settings
    pl.seed_everything(settings['seed'], workers=True, verbose=False)
    accelerator = settings['accelerator']
    trainer = pl.Trainer(accelerator=accelerator, devices=n_devices,
                         strategy=DDPStrategy(process_group_backend='nccl' if accelerator == 'gpu' else 'gloo'),
                         max_epochs=setup['controller'].max_epochs, use_distributed_sampler=True,
                         num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, default_root_dir=settings['directory'])
    trainer.fit(module, datamodule=datamodule)
