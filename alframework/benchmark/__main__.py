"""Command line for the GPU sizing benchmark; run ``python -m alframework.benchmark --help``."""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from alframework.benchmark import recommend
from alframework.benchmark.systems import load_system


def read_run(run_dir):
    """The settings a run directory gives the benchmark: its network, states, batch sizes, and structure."""
    run_dir = Path(run_dir).expanduser().resolve()
    master = json.loads((run_dir / 'master_config.json').read_text())
    ml = json.loads((run_dir / master['ML_config_path']).read_text())
    sampler = json.loads((run_dir / master['sampler_config_path']).read_text())
    states = sum(1 for key in master.get('properties_list', {}) if re.fullmatch(r'E\d+', key)) or 1
    controller = ml.get('controller_options') or {}
    reference = run_dir / 'reference.xyz'
    return dict(network=dict(network_choice=int(ml.get('network_choice', 0)), network_params=ml['network_params']),
                states=states, members=int(ml.get('n_models', 4)),
                batch_size=ml.get('batch_size', controller.get('batch_size')),
                gpus_per_member=ml.get('gpus_per_member', 1),
                sampler_batch=(sampler.get('batching') or {}).get('batch_size'),
                parallel_samplers=master.get('parallel_samplers'),
                structure=str(reference) if reference.exists() else None)


def parse(argv=None):
    parser = argparse.ArgumentParser(prog='python -m alframework.benchmark', description=__import__(
        'alframework.benchmark', fromlist=['__doc__']).__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', help='Read the network, states, and batch sizes from this run directory.')
    parser.add_argument('--system', help='water (the default), cppN from a clone, or a structure file.')
    parser.add_argument('--atoms', type=int, default=96, help='Atoms in the water system.')
    parser.add_argument('--profiles', nargs='+', help='Networks to measure: hipnn64, hipnn100, hiphop64, '
                        'hiphop100, hiphop145, or run (the run directory\'s). Default: run, or hipnn64 hipnn100.')
    parser.add_argument('--states', type=int, help='Electronic states per model. Default: the run\'s, or 1.')
    parser.add_argument('--members', type=int, help='Ensemble members loaded for sampling. Default: the run\'s, or 4.')
    parser.add_argument('--train-batches', nargs='+', type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument('--sample-batches', nargs='+', type=int, default=[1, 4, 16, 64, 256],
                        help='Trajectories per sampler GPU to try.')
    parser.add_argument('--device', default='cuda', help='cuda (the default), cuda:N, or cpu to check the script.')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=0.8, help='Share of GPU memory a setting may use.')
    parser.add_argument('--train-rows', type=int, default=40000, help='Training structures, for epoch times.')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--walltime-hours', type=float, default=16.0)
    parser.add_argument('--sampler-gpus', type=int, help='GPUs sampling runs on. Default: the GPUs visible here.')
    parser.add_argument('--hippynn-kernel', default='auto')
    parser.add_argument('--only', choices=['training', 'sampling'])
    parser.add_argument('--output', type=Path, default=Path('gpu_sizing.json'))
    parser.add_argument('--case', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_in_process(case):
    """Run one case in a fresh process, so running out of memory ends only that case."""
    done = subprocess.run([sys.executable, '-m', 'alframework.benchmark', '--case', json.dumps(case)],
                          capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
    if done.returncode == 0 and lines:
        return json.loads(lines[-1])
    text = (done.stderr + done.stdout).strip()
    record = {key: value for key, value in case.items() if key not in ('network_params', 'structure')}
    record.update(result='oom' if 'out of memory' in text.lower() else 'error',
                  error=text.splitlines()[-1] if text else f'exit {done.returncode}')
    return record


def gpu_memory_gib(device):
    if not str(device).startswith('cuda'):
        return None, 0
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU is visible; use --device cpu to check the script.')
    index = torch.device(device).index or 0
    return torch.cuda.get_device_properties(index).total_memory / 2**30, torch.cuda.device_count()


def table(title, header, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    print(f'\n{title}')
    for line in [header, ['-' * w for w in widths], *rows]:
        print('  '.join(str(cell).ljust(w) for cell, w in zip(line, widths)))


def _f(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def main(argv=None):
    args = parse(argv)
    if args.case:
        from alframework.benchmark.cases import run_case
        print(json.dumps(run_case(json.loads(args.case))))
        return 0
    from alframework.benchmark.cases import network_for
    run = read_run(args.run_dir) if args.run_dir else None
    atoms = load_system(args.system or (run or {}).get('structure') or 'water', args.atoms)
    states = args.states or (run or {}).get('states') or 1
    members = args.members or (run or {}).get('members') or 4
    profiles = args.profiles or (['run'] if run else ['hipnn64', 'hipnn100'])
    train_batches = sorted(set(args.train_batches) | ({run['batch_size']} if run and run['batch_size'] else set()))
    sample_batches = sorted(set(args.sample_batches) | ({run['sampler_batch']} if run and run['sampler_batch'] else set()))
    memory, visible = gpu_memory_gib(args.device)
    common = dict(device=args.device, steps=args.steps, warmup=args.warmup, states=states,
                  hippynn_kernel=args.hippynn_kernel)
    print(f'{len(atoms)} atoms ({atoms.get_chemical_formula()}), {states} state(s), device {args.device}'
          + (f', {memory:.1f} GiB' if memory else ' (CPU: only checks that the script works)'))
    records = []
    with tempfile.TemporaryDirectory(prefix='alf-benchmark-') as directory:
        structure = os.path.join(directory, 'structure.xyz')
        atoms.write(structure)
        for profile in profiles:
            choice, params = network_for(profile, atoms, (run or {}).get('network'))
            network = dict(common, profile=profile, network_choice=choice, network_params=params, structure=structure)
            if args.only != 'sampling':
                for batch in train_batches:
                    records.append(run_in_process(dict(network, kind='training', batch=batch)))
                    print(f'  training {profile} batch {batch}: {records[-1]["result"]}', flush=True)
            if args.only != 'training':
                for count in sample_batches:
                    records.append(run_in_process(dict(network, kind='sampling', trajectories=count, members=members)))
                    print(f'  sampling {profile} {count} trajectories: {records[-1]["result"]}', flush=True)
    report(records, args, memory, visible, run)
    return 0


def report(records, args, memory, visible, run):
    training = [r for r in records if r['kind'] == 'training']
    sampling = [r for r in records if r['kind'] == 'sampling']
    if training:
        table('Training (one GPU)', ['profile', 'batch', 'train s', 'eval s', 'structures/s', 'reserved GiB', 'result'],
              [[r['profile'], r['batch'], _f(r.get('train_step_s')), _f(r.get('eval_step_s')),
                _f(r['batch'] / r['train_step_s'], 1) if r.get('train_step_s') else '-', _f(r.get('peak_reserved_gib'), 2),
                r['result']] for r in training])
    if sampling:
        table('Sampling (one GPU, all members)', ['profile', 'trajectories', 'step s', 'structures/s', 'reserved GiB',
                                                  'result'],
              [[r['profile'], r['trajectories'], _f(r.get('step_s')),
                _f(r['trajectories'] / r['step_s'], 1) if r.get('step_s') else '-', _f(r.get('peak_reserved_gib'), 2),
                r['result']] for r in sampling])
    train = recommend.training_suggestions(records, memory_gib=memory, memory_fraction=args.memory_fraction,
                                           train_rows=args.train_rows, epochs=args.epochs,
                                           walltime_hours=args.walltime_hours, run_batch=(run or {}).get('batch_size'))
    sample = recommend.sampling_suggestions(records, memory_gib=memory, memory_fraction=args.memory_fraction,
                                            sampler_gpus=args.sampler_gpus or visible or 1)
    print('\nSuggestions' + ('' if memory else ' (from CPU timings, so only a check of the script)'))
    for profile, s in train.items():
        if s['batch_size'] is None:
            print(f'  {profile}: no training batch size fit in {args.memory_fraction:.0%} of GPU memory.')
            continue
        line = (f'  {profile} training: "batch_size": {s["batch_size"]}, about {s["epoch_minutes"]:.1f} min per epoch '
                f'for {args.train_rows} structures; {args.epochs} epochs take {s["training_hours"]:.1f} h, which '
                + ('fits' if s['fits_walltime'] else 'does not fit') + f' a {args.walltime_hours:g} h allocation.')
        print(line)
        if s.get('run_batch_size'):
            gpus = s['gpus_per_member']
            print(f'    The run\'s batch size {s["run_batch_size"]}: ' + (
                'did not fit when split over up to 8 GPUs.' if gpus is None else
                f'"gpus_per_member": {gpus} (an estimate, without the time GPUs spend exchanging gradients).'))
    for profile, s in sample.items():
        if s['trajectories_per_gpu'] is None:
            print(f'  {profile} sampling: no batch fit in {args.memory_fraction:.0%} of GPU memory.')
            continue
        print(f'  {profile} sampling: {s["trajectories_per_gpu"]} trajectories per GPU, about '
              f'{s["structures_per_second"]:.0f} MD steps/s in total; "batching": {{"batch_size": '
              f'{s["trajectories_per_gpu"]}}} and "parallel_samplers": {s["parallel_samplers"]} for '
              f'{s["sampler_gpus"]} sampler GPU(s).')
    environment = dict(python=platform.python_version(), platform=platform.platform())
    for package in ('torch', 'hippynn'):
        try:
            environment[package] = __import__(package).__version__
        except Exception:
            environment[package] = None
    args.output.write_text(json.dumps(dict(arguments={k: str(v) for k, v in vars(args).items() if k != 'case'},
                                           gpu_memory_gib=memory, environment=environment, records=records,
                                           training=train, sampling=sample), indent=2, default=str) + '\n')
    print(f'\nWrote every measurement to {args.output}')


if __name__ == '__main__':
    raise SystemExit(main())
