"""Train two UMA members and compare single-member NTK on a prepared public QM pool."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from ase import Atoms

from alframework.ml_interfaces.fine_tuning import fine_tune_member
from alframework.ml_interfaces.fine_tuning.uma import load_uma_batched_ensemble
from alframework.ml_interfaces.ntk.uma import load_uma_ntk
from alframework.samplers.engines.torchsim import TorchSimEngine
from alframework.tools.files import cached_sha256, write_json_atomic
from examples.atom_resolved_uncertainty.benchmark_ntk import (
    DATA_MANIFEST, _budget_metrics, _calibrate, _read_pool, _score_pool,
)


def train(dataset_dir, checkpoint, out, device, steps):
    """Use a node-local copy of the published split and train two matched LoRA members."""
    manifest = json.loads((dataset_dir / DATA_MANIFEST).read_text())
    source = dataset_dir / manifest['splits']['training']['file']
    if cached_sha256(source) != manifest['splits']['training']['sha256']:
        raise ValueError('Training shard differs from the public data manifest.')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shard = out / 'data.h5'
    shutil.copyfile(source, shard)
    settings = dict(adapter='alframework.ml_interfaces.fine_tuning.uma.UmaFineTuner',
                    foundation_model=str(checkpoint.resolve()), task='omol', E0s='average', seed=17,
                    steps=steps, epochs=None, batch_size=2, max_neighbors=100, lora=True)
    for member in range(2):
        settings['ntk'] = {'sketch_dim': 16} if member == 0 else None
        if not fine_tune_member(settings, [shard], out / 'models' / f'model-{member:02d}', member, device):
            raise RuntimeError(f'UMA benchmark member {member} did not finish.')


def _md_timing(model, frames, device, indices, *, use_ntk):
    width = len(frames)
    active = list(range(width))
    engine = TorchSimEngine([frame.copy() for frame in frames], model, 0.25, [300.] * width,
                            [np.random.RandomState(200 + i) for i in active], device=device)
    if use_ntk:
        engine.set_force_atom_indices({i: tuple(indices) for i in active})
    samples = []
    steps = 4
    for cycle in range(6):
        engine.synchronize()
        start = perf_counter()
        failures = engine.step(active, steps)
        engine.synchronize()
        stepped = perf_counter()
        if failures:
            raise RuntimeError(f'UMA benchmark trajectories failed: {failures}')
        for index in active:
            prediction = engine.observe(index)['prediction']
            if use_ntk:
                values = prediction['selected_force_uncertainty'][0]['ntk_force_selected_max']
            else:
                values = np.linalg.norm(np.std(prediction['forces'][0][:, indices], axis=0), axis=-1)
            if not np.isfinite(values.max()):
                raise FloatingPointError('Nonfinite UMA acquisition score.')
        engine.synchronize()
        checked = perf_counter()
        if cycle >= 2:
            samples.append((stepped - start, checked - stepped, checked - start))
    md, observation, cycle = np.median(samples, axis=0)
    return dict(trajectories=width, steps_per_check=steps, md_seconds_per_step=float(md / steps),
                check_seconds=float(observation), trajectory_steps_per_second=float(width * steps / cycle))


def measure(dataset_dir, model_dir, out, device, indices, width):
    """Replay two rankings and measure warmed TorchSim throughput with the same checkpoints."""
    import torch

    out.mkdir(parents=True, exist_ok=False)
    manifest_path = dataset_dir / DATA_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    calibration_pool = _read_pool(dataset_dir, manifest, 'calibration')
    test_pool = _read_pool(dataset_dir, manifest, 'test')
    if width > len(test_pool['positions']) or width < 1:
        raise ValueError('Select a positive trajectory count within the test pool.')
    if not indices or len(set(indices)) != len(indices) or not all(0 <= i < len(test_pool['numbers']) for i in indices):
        raise ValueError('Select distinct atom indices in the dataset atom order.')
    peak = None
    ensemble = load_uma_batched_ensemble(model_dir, task='omol', device=device)
    ntk = load_uma_ntk(model_dir, device, force_score='selected')
    cal, _ = _score_pool(calibration_pool, indices, 0, ensemble, ntk.estimator, 8, verify_predictions=True)
    stats = _calibrate(cal, None, None)
    records, times = _score_pool(test_pool, indices, 0, ensemble, ntk.estimator, 8)
    budget = _budget_metrics(records, stats)
    frames = [Atoms(numbers=test_pool['numbers'], positions=positions) for positions in test_pool['positions'][:width]]
    # The replay holds both providers. Measure each provider's memory and dynamics separately.
    basis_id = ntk.estimator.basis_id
    del ensemble, ntk
    results = {}
    for method in ('ensemble', 'ntk'):
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        if method == 'ensemble':
            model = load_uma_batched_ensemble(model_dir, task='omol', device=device)
        else:
            model = load_uma_ntk(model_dir, device, force_score='selected')
        results[method] = _md_timing(model, frames, device, indices, use_ntk=method == 'ntk')
        if torch.device(device).type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
        results[method]['peak_gpu_bytes'] = peak
        del model
    fit_log = (model_dir / 'model-00' / 'ntk_fit.log').read_text()
    fit_report, _ = json.JSONDecoder().raw_decode(fit_log[fit_log.index('\n{') + 1:] if '\n{' in fit_log else fit_log)
    write_json_atomic(out / 'summary.json', dict(
        dataset_manifest_sha256=cached_sha256(manifest_path), basis_id=basis_id, device=device, atoms=indices,
        environment={'fairchem_core': version('fairchem-core'), 'torch': str(torch.__version__),
                     'hip': torch.version.hip, 'cuda': torch.version.cuda,
                     'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None},
        model_fingerprints={path.parent.name: cached_sha256(path) for path in model_dir.glob('model-*/uma.pt')},
        basis_fit=fit_report, calibration=stats, replay_ms={name: float(np.median(samples)) * 1000
                                                         for name, samples in times.items()},
        packed_md=results, **budget))
    with (out / 'test.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--checkpoint', type=Path, required=True)
    training.add_argument('--steps', type=int, default=4)
    scoring = commands.add_parser('measure')
    scoring.add_argument('--model-dir', type=Path, required=True)
    scoring.add_argument('--indices', type=int, nargs='+', required=True)
    scoring.add_argument('--width', type=int, default=4)
    for command in (training, scoring):
        command.add_argument('--dataset-dir', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--device', default='cuda:0')
    options = vars(parser.parse_args())
    command = options.pop('command')
    (train if command == 'train' else measure)(**options)


if __name__ == '__main__':
    main()
