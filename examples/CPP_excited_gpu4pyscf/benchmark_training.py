"""Measure training and evaluation step time and GPU memory for each CPP network profile.

python benchmark_training.py --output training_benchmark.jsonl
python benchmark_training.py --profiles hiphop64 hipnn64 --batches 8 16 32 --output bench.jsonl

Each case builds the production S0-S3 energy and force graph from hippynn_config.json, with the
profile's network, and times AdamW updates on rattled copies of the minimized reference. Step time and
memory don't depend on the label values, so the targets are random. Each case runs in its own process,
so running out of memory in one doesn't affect the next. One JSON line per case goes to --output, then
a table with epoch projections for --train-rows training and --eval-rows evaluated structures.
Runs on CPU too (--device cpu), for a quick check that the script works; only GPU numbers mean anything.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = ('hiphop145', 'hiphop100', 'hiphop64', 'hipnn100', 'hipnn64')


def run_case(profile, batch, mode, device, warmup, steps, states):
    import numpy as np
    import torch
    from ase.io import read
    from hippynn.graphs import GraphModule, inputs

    from alframework.ml_interfaces.hippynn_interface import (_build_shared_network,
                                                             build_excited_state_training_graph)
    from prepare import apply_model_profile

    torch.manual_seed(0)
    atoms = read(HERE/'references/cpp9_s0.xyz')
    config = apply_model_profile(json.loads((HERE/'hippynn_config.json').read_text()), profile, len(atoms))
    species = inputs.SpeciesNode(db_name='species')
    positions = inputs.PositionsNode(db_name='coordinates')
    network = _build_shared_network(species, positions, network_choice=config['network_choice'],
                                    network_params=config['network_params'])
    table = [dict(state=state, energy_db_name=f'E{state}', force_db_name=f'F{state}') for state in range(states)]
    energies, forces, _, _ = build_excited_state_training_graph(
        network, positions, table, n_atoms=len(atoms), energy_weight=1.0, force_weight=1.0, l2_weight=0.0)
    model = GraphModule([species, positions], [node for _, node in energies] + [node for _, node in forces]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(0)
    z = torch.tensor(np.tile(atoms.numbers, (batch, 1)), dtype=torch.long, device=device)
    r = torch.tensor(atoms.positions[None] + rng.normal(scale=0.02, size=(batch, len(atoms), 3)),
                     dtype=torch.float32, device=device)
    e_true = torch.randn(batch, 1, device=device)
    f_true = torch.randn(batch, len(atoms), 3, device=device)
    cuda = str(device).startswith('cuda')
    model.train(mode == 'train')

    def step():
        rr = r.clone().requires_grad_(True)
        out = model(z, rr)
        if mode == 'train':
            optimizer.zero_grad()
            loss = sum(((e - e_true)**2).mean() for e in out[:states]) + \
                sum(((f - f_true)**2).mean() for f in out[states:])
            loss.backward()
            optimizer.step()
        if cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    record = dict(profile=profile, mode=mode, batch=batch, device=str(device), states=states,
                  parameters=sum(p.numel() for p in model.parameters()), result='passed',
                  median_step_s=float(np.median(times)), steps=steps, warmup=warmup)
    if cuda:
        record.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      gpu=torch.cuda.get_device_name())
    return record


def case_in_subprocess(profile, batch, mode, args):
    command = [sys.executable, str(Path(__file__).resolve()), '--case', profile, str(batch), mode,
               '--device', args.device, '--warmup', str(args.warmup), '--steps', str(args.steps),
               '--states', str(args.states)]
    done = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
    lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
    if done.returncode == 0 and lines:
        return json.loads(lines[-1])
    text = done.stderr + done.stdout
    result = 'oom' if 'out of memory' in text.lower() else 'error'
    return dict(profile=profile, mode=mode, batch=batch, device=args.device, result=result,
                error=text.strip().splitlines()[-1] if text.strip() else f'exit {done.returncode}')


def projections(records, train_rows, eval_rows, walltime_hours):
    """Minutes per epoch and epochs per allocation, pairing each training batch with the same eval batch."""
    steps = {(r['profile'], r['mode'], r['batch']): r for r in records if r['result'] == 'passed'}
    rows = []
    for (profile, mode, batch), record in sorted(steps.items()):
        if mode != 'train':
            continue
        evaluation = steps.get((profile, 'eval', batch))
        minutes = math.ceil(train_rows/batch)*record['median_step_s']/60
        if evaluation:
            minutes += math.ceil(eval_rows/batch)*evaluation['median_step_s']/60
        rows.append(dict(profile=profile, batch=batch, epoch_minutes=minutes, includes_eval=bool(evaluation),
                         epochs_per_allocation=walltime_hours*60/minutes,
                         peak_reserved_gib=record.get('peak_reserved_gib')))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profiles', nargs='+', default=list(DEFAULT_PROFILES))
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument('--eval-batches', nargs='+', type=int, default=[8, 16, 32])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--states', type=int, default=4)
    parser.add_argument('--train-rows', type=int, default=40000)
    parser.add_argument('--eval-rows', type=int, default=9000,
                        help='validation rows plus the 10%% training subset the trainer evaluates')
    parser.add_argument('--walltime-hours', type=float, default=16.0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', nargs=3, metavar=('PROFILE', 'BATCH', 'MODE'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case:
        profile, batch, mode = args.case
        print(json.dumps(run_case(profile, int(batch), mode, args.device, args.warmup, args.steps, args.states)))
        return 0
    if args.output is None:
        parser.error('--output is required')
    records = []
    with open(args.output, 'w') as handle:
        for profile in args.profiles:
            for mode, batches in (('train', args.batches), ('eval', args.eval_batches)):
                for batch in batches:
                    record = case_in_subprocess(profile, batch, mode, args)
                    records.append(record)
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    memory = record.get('peak_reserved_gib')
                    print(f"{profile:10s} {mode:5s} batch {batch:3d}: {record['result']:6s}"
                          + (f" {record['median_step_s']:.4f} s/step" if record['result'] == 'passed' else '')
                          + (f", {memory:.2f} GiB reserved" if memory is not None else ''), flush=True)
    print(f'\nProjected epoch at {args.train_rows} training and {args.eval_rows} evaluated rows:')
    for row in projections(records, args.train_rows, args.eval_rows, args.walltime_hours):
        print(f"{row['profile']:10s} batch {row['batch']:3d}: {row['epoch_minutes']:7.1f} min/epoch"
              f"{'' if row['includes_eval'] else ' (no eval at this batch)'}, "
              f"{row['epochs_per_allocation']:6.1f} epochs per {args.walltime_hours:g} h")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
