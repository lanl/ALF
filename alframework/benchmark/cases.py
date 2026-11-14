"""Run one benchmark case. ``__main__`` starts each case in its own process and reads its JSON line."""
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

# The CPP example's profiles (examples/CPP_excited_gpu4pyscf/prepare.py): (network_choice, n_features).
PROFILES = {'hipnn64': (0, 64), 'hipnn100': (0, 100), 'hiphop64': (1, 64), 'hiphop100': (1, 100),
            'hiphop145': (1, 145)}
# ALF's CPP defaults, used when no run directory gives a network.
DEFAULT_NETWORK = dict(n_features=64, n_sensitivities=20, n_interaction_layers=2, n_atom_layers=3,
                       dist_soft_min=0.85, dist_soft_max=5.5, dist_hard_max=8.0)


def network_for(profile, atoms, run_network=None):
    """Return ``(network_choice, network_params)`` for a profile, or the run's own network for ``run``."""
    if profile == 'run':
        choice, params = run_network['network_choice'], dict(run_network['network_params'])
    else:
        if profile not in PROFILES:
            raise ValueError(f'Unknown profile {profile!r}; choose from {sorted(PROFILES)} or run')
        choice, features = PROFILES[profile]
        base = dict((run_network or {}).get('network_params') or DEFAULT_NETWORK)
        params = dict(base, n_features=features)
        for key in ('l_max', 'n_max'):
            params.pop(key, None)
        if choice == 1:
            params.update(l_max=3, n_max=4)
    params['possible_species'] = sorted({0, *(int(z) for z in atoms.numbers), *params.get('possible_species', [])})
    return choice, params


def properties(states):
    return {f'{kind}{state}': [f'{kind}{state}', 'system' if kind == 'E' else 'atomic', 1.0]
            for state in range(states) for kind in 'EF'}


def _graph(atoms, choice, params, states):
    from hippynn.graphs import inputs
    from alframework.ml_interfaces.hippynn_interface import (
        _build_shared_network, build_excited_state_training_graph, validate_excited_state_training_properties)
    species = inputs.SpeciesNode(db_name='species')
    positions = inputs.PositionsNode(db_name='coordinates')
    positions.requires_grad = True
    network = _build_shared_network(species, positions, network_choice=choice, network_params=params)
    table = validate_excited_state_training_properties(properties(states))
    energies, forces, loss, validation = build_excited_state_training_graph(
        network, positions, table, n_atoms=len(atoms), energy_weight=1.0, force_weight=1.0, l2_weight=0.0)
    return species, positions, energies, forces, loss, validation


def _copies(atoms, count, seed=0):
    rng = np.random.default_rng(seed)
    copies = []
    for _ in range(count):
        copy = atoms.copy()
        copy.positions = copy.positions + rng.normal(scale=0.02, size=copy.positions.shape)
        copies.append(copy)
    return copies


def _memory(torch, cuda):
    if not cuda:
        return {}
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def _timed(step, steps, warmup, torch, cuda):
    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def training_case(case, atoms, torch):
    from hippynn.graphs import GraphModule
    choice, params = case['network_choice'], case['network_params']
    species, positions, energies, forces, _, _ = _graph(atoms, choice, params, case['states'])
    device, batch, states = torch.device(case['device']), case['batch'], case['states']
    model = GraphModule([species, positions], [n for _, n in energies] + [n for _, n in forces]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    z = torch.tensor(np.tile(atoms.numbers, (batch, 1)), dtype=torch.long, device=device)
    r = torch.tensor(np.stack([a.positions for a in _copies(atoms, batch)]), dtype=torch.float32, device=device)
    e_true, f_true = torch.randn(batch, 1, device=device), torch.randn(batch, len(atoms), 3, device=device)
    cuda = device.type == 'cuda'

    def train():
        out = model(z, r.clone().requires_grad_(True))
        optimizer.zero_grad()
        loss = sum(((e - e_true) ** 2).mean() for e in out[:states]) + sum(((f - f_true) ** 2).mean() for f in out[states:])
        loss.backward()
        optimizer.step()

    def evaluate():
        model(z, r.clone().requires_grad_(True))

    model.train()
    train_step = _timed(train, case['steps'], case['warmup'], torch, cuda)
    memory = _memory(torch, cuda)
    model.eval()
    eval_step = _timed(evaluate, case['steps'], case['warmup'], torch, cuda)
    return dict(train_step_s=train_step, eval_step_s=eval_step, parameters=sum(p.numel() for p in model.parameters()),
                **memory)


def write_untrained_members(directory, atoms, choice, params, states, members):
    """Write ``members`` untrained members that ALF's sampler loaders read like trained ones."""
    import hippynn
    import torch
    for member in range(members):
        torch.manual_seed(member)
        *_, loss, validation = _graph(atoms, choice, params, states)
        training_modules, _ = hippynn.experiment.assemble_for_training(loss, validation)
        out = Path(directory) / f'model-{member:02d}'
        out.mkdir(parents=True)
        torch.save({'training_modules': training_modules}, out / 'experiment_structure.pt')
        torch.save(training_modules.model.state_dict(), out / 'best_model.pt')
        (out / 'member_composition.json').write_text(json.dumps({'atomic_numbers': atoms.numbers.tolist()}))


def sampling_case(case, atoms, torch):
    from alframework.ml_interfaces import hippynn_interface as ml
    device, states = case['device'], case['states']
    with tempfile.TemporaryDirectory(prefix='alf-benchmark-') as directory:
        write_untrained_members(directory, atoms, case['network_choice'], case['network_params'], states,
                                case['members'])
        structures = _copies(atoms, case['trajectories'])
        if states == 1:
            # The packed-tensor call the batched dynamics engines make every step.
            ensemble = ml.load_hippynn_batched_ensemble(directory + '/', device=device, energy_key='E0', force_key='F0')
            numbers = torch.as_tensor(np.concatenate([a.numbers for a in structures]), device=device)
            positions = torch.as_tensor(np.concatenate([a.positions for a in structures]), dtype=torch.float32,
                                        device=device)
            cells = torch.zeros((len(structures), 3, 3), device=device)
            counts = [len(a) for a in structures]
            predict = lambda: ensemble.evaluate(numbers, positions, cells, [False] * 3, counts)
        else:
            ensemble = ml.load_hippynn_excited_state_ensemble(directory, device, properties(states),
                                                              case.get('hippynn_kernel', 'auto'))
            predict = lambda: ensemble.predict_states(structures)
        step = _timed(predict, case['steps'], case['warmup'], torch, torch.device(device).type == 'cuda')
        return dict(step_s=step, **_memory(torch, torch.device(device).type == 'cuda'))


def run_case(case):
    """Run one case and return its record, which includes the case's own settings."""
    import torch
    import hippynn
    from ase.io import read
    hippynn.custom_kernels.set_custom_kernels(case.get('hippynn_kernel', 'auto'))
    hippynn.settings.WARN_LOW_DISTANCES = False
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get('ALF_BENCHMARK_THREADS', '1')))
    atoms = read(case['structure'])
    record = {key: value for key, value in case.items() if key not in ('network_params', 'structure')}
    measure = training_case if case['kind'] == 'training' else sampling_case
    record.update(measure(case, atoms, torch), result='passed')
    if torch.device(case['device']).type == 'cuda':
        record['gpu'] = torch.cuda.get_device_name()
    return record
