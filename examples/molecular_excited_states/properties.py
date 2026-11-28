"""Fit and evaluate optional properties on a frozen HIP-HOP-NN representation."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from alframework.tools.elements import atomic_numbers_of
from alframework.tools.excited_state_schema import derive_state_property_table
from alframework.tools.files import cached_sha256
from alframework.tools.h5_parallel import read_shard
from alframework.tools.molecules import MOLECULE_ID

if __package__:
    from .evaluate import assert_independent, run_properties
    from .permanent_dipoles import permanent_dipoles
else:
    from evaluate import assert_independent, run_properties
    from permanent_dipoles import permanent_dipoles


@dataclass
class PropertyFrame:
    identity: str
    identifier: str
    numbers: np.ndarray
    positions: np.ndarray
    source: dict
    targets: np.ndarray | None
    excitation_eV: np.ndarray | None = None
    reference: tuple | None = None

    @property
    def geometry_sha256(self):
        digest = hashlib.sha256(np.asarray(self.numbers, dtype='<i8').tobytes())
        digest.update(np.asarray(self.positions, dtype='<f8').tobytes())
        return digest.hexdigest()


def shard_frames(shards, properties, *, optical):
    state_keys = {row['energy_key']: row['energy_db_name'] for row in derive_state_property_table(properties)}
    strength_key = properties['oscillator_strengths'][0] if optical else None
    for path in shards:
        path = Path(path).resolve()
        keys = {'species', 'coordinates', MOLECULE_ID, '_id', 'total_charge', 'spin_multiplicity'}
        if optical:
            keys.update([strength_key, *state_keys.values()])
        for group in read_shard(path, keys, group_keys={'coordinates'}):
            for key in keys - {'total_charge', 'spin_multiplicity'}:
                if key not in group:
                    raise ValueError(f'{path}:{group["path"]} is missing {key}.')
            positions = group['coordinates']
            count, n_atoms, _ = positions.shape
            species = np.broadcast_to(atomic_numbers_of(group['species']), (count, n_atoms))
            if (not np.isfinite(positions).all() or np.any(species <= 0)
                    or np.any(group.get('total_charge', 0)) or np.any(np.asarray(group.get('spin_multiplicity', 1)) != 1)):
                raise ValueError(f'{path}:{group["path"]} must contain finite neutral singlet geometries.')
            strengths, excitation = [None] * count, [None] * count
            if optical:
                strengths = group[strength_key] / float(properties['oscillator_strengths'][2])
                energies = np.column_stack([group[key] / float(properties[state][2]) for state, key in state_keys.items()])
                excitation = energies[:, 1:] - energies[:, :1]
                if (strengths.shape != excitation.shape or not np.isfinite(strengths).all()
                        or np.any(strengths < 0) or not np.isfinite(excitation).all()):
                    raise ValueError(f'{path}:{group["path"]} has invalid state-resolved optical labels.')
            for index in range(count):
                source = {'file': str(path), 'group': group['path'], 'frame': index}
                yield PropertyFrame(group[MOLECULE_ID][index], group['_id'][index], species[index], positions[index],
                                    source, strengths[index], excitation[index])


def write_report(path, report):
    with Path(path).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def label_dipoles(run, shards, output, *, frames_per_molecule=4, step=1.e-4, check_step=False):
    from pyscf import __version__ as version, lib

    from alframework.qm_interfaces.pyscf.gpu4pyscf import validate_gpu4pyscf_config

    if output.exists():
        raise FileExistsError(output)
    settings = validate_gpu4pyscf_config(json.loads((run / 'QM_config.json').read_text()))
    lib.num_threads(1)
    report = {'schema': 'alf-neutral-state-dipoles-v1', 'units': 'e Angstrom', 'states': [0, 1],
              'method': 'CPU PySCF conventional RKS/TDA, SCF S0 and finite-field S1',
              'xc': settings['xc'], 'basis': settings['basis'], 'pyscf_version': version,
              'step_atomic_units': step, 'frames': [], 'rejected': []}
    counts = Counter()
    for frame in shard_frames(shards, run_properties(run), optical=False):
        if counts[frame.identity] >= frames_per_molecule:
            continue
        counts[frame.identity] += 1
        key = {'id': frame.identifier, MOLECULE_ID: frame.identity,
               'source': frame.source, 'geometry_sha256': frame.geometry_sha256}
        try:
            dipoles, diagnostic = permanent_dipoles(Atoms(frame.numbers, positions=frame.positions),
                                                    settings, step, check_step=check_step)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            report['rejected'].append({**key, 'error': str(error)})
            continue
        report['frames'].append({**key, 'numbers': frame.numbers.tolist(), 'positions_A': frame.positions.tolist(),
                                 'dipoles_eA': dipoles.tolist(), 'diagnostics': diagnostic})
    write_report(output, report)
    return len(report['frames']), len(report['rejected'])


def dipole_frames(paths):
    methods = set()
    for path in paths:
        report = json.loads(Path(path).read_text())
        if (report.get('schema') != 'alf-neutral-state-dipoles-v1' or report.get('units') != 'e Angstrom'
                or report.get('states') != [0, 1]):
            raise ValueError(f'{path} does not contain the neutral S0/S1 dipole schema.')
        method = (report['xc'], report['basis'], report['method'], report['step_atomic_units'])
        methods.add(method)
        if len(methods) != 1:
            raise ValueError('Use one dipole reference method, basis and field step per property fit.')
        for row in report['frames']:
            frame = PropertyFrame(row[MOLECULE_ID], row['id'], np.asarray(row['numbers']),
                                  np.asarray(row['positions_A']), row['source'], np.asarray(row['dipoles_eA']),
                                  reference=method)
            if (frame.geometry_sha256 != row['geometry_sha256'] or frame.targets.shape != (2, 3)
                    or frame.numbers.ndim != 1 or not len(frame.numbers) or np.any(frame.numbers <= 0)
                    or frame.positions.shape != (len(frame.numbers), 3) or not np.isfinite(frame.positions).all()):
                raise ValueError(f'{path}: invalid geometry fingerprint or dipole shape for {frame.identifier}.')
            if not np.isfinite(frame.targets).all():
                raise ValueError(f'{path}: nonfinite dipoles for {frame.identifier}.')
            yield frame


class Readout(torch.nn.Module):
    """Predict system scalars or charges on a padded batch of frozen atomic features."""

    def __init__(self, features, outputs, kind):
        super().__init__()
        self.kind = kind
        self.linear = torch.nn.Linear(features, outputs)

    def forward(self, features, numbers, positions):
        mask = (numbers > 0).unsqueeze(-1)
        values = self.linear(features) * mask
        if self.kind == 'oscillator':
            return {'oscillator_strengths': torch.nn.functional.softplus(values.sum(dim=1))}
        charges = (values - values.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)) * mask
        dipoles = torch.einsum('bas,bac->bsc', charges, positions)
        return {'charges': charges, 'dipoles_eA': dipoles}


def feature_graph(member, device, energy_states=()):
    from hippynn.graphs import GraphModule, get_subgraph
    from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode
    from hippynn.graphs.nodes.tags import Network

    from alframework.ml_interfaces.hippynn.interface import _load_member, _target

    model = _load_member(str(member.resolve()), device)
    networks = [node for node in get_subgraph(model.nodes_to_compute) if isinstance(node, Network)]
    if len(networks) != 1:
        raise ValueError('Expected one shared HIP-HOP network.')
    outputs = [networks[0].feature_nodes[-1], *(_target(model, row['energy_db_name']) for row in energy_states)]
    graph = GraphModule(model.input_nodes, outputs).to(device).eval()
    roles = []
    for node in graph.input_nodes:
        node.requires_grad = False
        if isinstance(node, SpeciesNode):
            roles.append('species')
        elif isinstance(node, PositionsNode):
            roles.append('positions')
        else:
            raise ValueError('Expected a molecular species/positions checkpoint.')
    return graph, roles


def checkpoint_hashes(member):
    return {name: cached_sha256(member / name) for name in ('experiment_structure.pt', 'best_model.pt')}


def featurize(frames, member, device, batch_size, energy_properties=None):
    from alframework.ml_interfaces.hippynn.interface import _member_composition

    if not frames:
        raise ValueError('The property dataset is empty.')
    composition = _member_composition(member)
    trained = set(composition['atomic_numbers'])
    if composition.get('mixed_molecules', False):
        if any(not set(frame.numbers).issubset(trained) for frame in frames):
            raise ValueError('A molecule contains elements absent from dynamics training.')
    elif any(sorted(frame.numbers) != sorted(composition['atomic_numbers']) for frame in frames):
        raise ValueError('A molecule differs from the fixed composition of the dynamics checkpoint.')
    states = derive_state_property_table(energy_properties) if energy_properties else []
    graph, roles = feature_graph(member, device, states)
    dtype = next(parameter.dtype for parameter in graph.parameters() if parameter.is_floating_point())
    width = max(len(frame.numbers) for frame in frames)
    numbers = torch.zeros(len(frames), width, dtype=torch.long)
    positions = torch.zeros(len(frames), width, 3, dtype=dtype)
    for index, frame in enumerate(frames):
        count = len(frame.numbers)
        numbers[index, :count] = torch.as_tensor(frame.numbers.copy())
        positions[index, :count] = torch.as_tensor(frame.positions)
    chunks, energy_batches = [], []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            end = start + batch_size
            actual = max(len(item.numbers) for item in frames[start:end])
            species = numbers[start:end, :actual].to(device)
            coords = positions[start:end, :actual].to(device)
            inputs = {'species': species, 'positions': coords}
            packed, *energies = graph(*(inputs[role] for role in roles))
            padded = torch.zeros(len(species), width, packed.shape[-1], dtype=dtype)
            padded[numbers[start:end] > 0] = packed.cpu()
            chunks.append(padded)
            if states:
                energy_batches.append(torch.column_stack([
                    values.reshape(-1).cpu() / energy_properties[row['energy_key']][2]
                    for row, values in zip(states, energies)
                ]))
    features = torch.cat(chunks), numbers, positions
    return (features, torch.cat(energy_batches).numpy()) if states else features


def target_name(kind):
    return {'oscillator': 'oscillator_strengths', 'charge': 'dipoles_eA'}[kind]


def frame_key(frame):
    return {'id': frame.identifier, MOLECULE_ID: frame.identity, 'source': frame.source,
            'geometry_sha256': frame.geometry_sha256}


def load_frames(kind, paths, properties):
    return list(shard_frames(paths, properties, optical=True) if kind == 'oscillator' else dipole_frames(paths))


def fit(kind, member, run, training, validation, output, *, device='cpu', epochs=300, batch_size=32, seed=42):
    if output.exists():
        raise FileExistsError(output)
    properties = run_properties(run)
    train, valid = [load_frames(kind, paths, properties) for paths in (training, validation)]
    if {frame.identity for frame in train} & {frame.identity for frame in valid}:
        raise ValueError('Property training and validation must be molecule-disjoint.')
    references = {frame.reference for frame in [*train, *valid]}
    if len(references) > 1:
        raise ValueError('Property training and validation use different reference methods.')
    ensemble = member.parent
    assert_independent(ensemble, [frame_key(frame) for frame in valid], whole_molecules=True)
    train_data, valid_data = [featurize(frames, member, device, batch_size) for frames in (train, valid)]
    y_train, y_valid = [torch.as_tensor(np.stack([frame.targets for frame in frames]),
                                        dtype=train_data[0].dtype) for frames in (train, valid)]
    torch.manual_seed(seed)
    head = Readout(train_data[0].shape[-1], y_train.shape[1], kind).to(device=device, dtype=y_train.dtype)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(seed)
    best, weights = float('inf'), None

    def loss(data, y, ids):
        predicted = head(*(values[ids].to(device) for values in data))[target_name(kind)]
        return torch.nn.functional.mse_loss(predicted, y[ids].to(device))

    for _ in range(epochs):
        for ids in torch.randperm(len(train), generator=generator).split(batch_size):
            optimizer.zero_grad()
            objective = loss(train_data, y_train, ids)
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            objective = float(loss(valid_data, y_valid, slice(None)))
        if objective < best:
            best = objective
            weights = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if weights is None:
        raise FloatingPointError('No finite validation loss for the property head.')
    report = {'kind': kind, 'state_indices': list(range(1 if kind == 'oscillator' else 0,
                                                        y_train.shape[1] + (kind == 'oscillator'))),
              'units': 'dimensionless' if kind == 'oscillator' else 'e Angstrom',
              'features': head.linear.in_features, 'outputs': head.linear.out_features,
              'member': str(member.resolve()), 'parent_sha256': checkpoint_hashes(member),
              'reference': next(iter(references)),
              'training': [frame_key(frame) for frame in train], 'validation': [frame_key(frame) for frame in valid],
              'best_validation_mse': best, 'seed': seed, 'epochs': epochs,
              'properties_list': properties, 'weights': weights}
    with output.open('xb') as stream:
        torch.save(report, stream)
    return best


def spectra_overlap(reference, predicted, delta_ref, delta_pred, width=.3):
    start = float(min(delta_ref.min(), delta_pred.min())) - 4 * width
    stop = float(max(delta_ref.max(), delta_pred.max())) + 4 * width
    grid = np.linspace(start, stop, 401)

    def spectrum(strength, energy):
        return np.sum(strength[:, None] * np.exp(-((grid - energy[:, None]) / width) ** 2), axis=0)

    left, right = spectrum(reference, delta_ref), spectrum(predicted, delta_pred)
    total = np.maximum(left, right).sum()
    return float(np.minimum(left, right).sum() / total) if total else 1.


def load_readout(checkpoint, member):
    manifest = torch.load(checkpoint, map_location='cpu', weights_only=True)
    member = member or Path(manifest['member'])
    if checkpoint_hashes(member) != manifest['parent_sha256']:
        raise ValueError('Property checkpoint does not match this dynamics checkpoint.')
    return manifest, member


def predict_frames(manifest, member, frames, *, device, batch_size):
    kind = manifest['kind']
    if kind == 'oscillator':
        features, energies = featurize(frames, member, device, batch_size, manifest['properties_list'])
    else:
        features = featurize(frames, member, device, batch_size)
    head = Readout(manifest['features'], manifest['outputs'], kind).to(device=device, dtype=features[0].dtype)
    head.load_state_dict(manifest['weights'])
    output = {'kind': kind, 'states': manifest['state_indices'], 'units': manifest['units'], 'frames': []}
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            end = start + batch_size
            batch = head(*(values[start:end].to(device) for values in features))
            for i, frame in enumerate(frames[start:end]):
                row = {**frame_key(frame), target_name(kind): batch[target_name(kind)][i].cpu().tolist()}
                if kind == 'charge':
                    row['charges_e'] = batch['charges'][i, :len(frame.numbers)].cpu().tolist()
                else:
                    energy = energies[start + i]
                    row['excitation_eV'] = (energy[1:] - energy[0]).tolist()
                output['frames'].append(row)
    return output


def predict(checkpoint, structures, *, device='cpu', member=None, batch_size=32):
    from ase.io import iread

    manifest, member = load_readout(checkpoint, member)
    frames = []
    for i, atoms in enumerate(iread(structures, index=':')):
        charge = atoms.info.get('total_charge', atoms.info.get('charge', 0))
        if (atoms.pbc.any() or charge != 0 or atoms.info.get('spin_multiplicity', 1) != 1
                or not len(atoms) or not np.isfinite(atoms.positions).all() or np.any(atoms.numbers <= 0)):
            raise ValueError(f'{structures}:{i} must contain a finite, nonperiodic neutral singlet.')
        frames.append(PropertyFrame(atoms.info.get(MOLECULE_ID, atoms.get_chemical_formula()),
                                    str(atoms.info.get('frame_id', i)), atoms.numbers, atoms.positions,
                                    {'file': str(Path(structures).resolve()), 'frame': i}, None))
    return predict_frames(manifest, member, frames, device=device, batch_size=batch_size)


def score(checkpoint, labels, *, device='cpu', member=None, batch_size=32):
    manifest, member = load_readout(checkpoint, member)
    kind = manifest['kind']
    frames = load_frames(kind, labels, manifest['properties_list'])
    reserved = {row[MOLECULE_ID] for split in ('training', 'validation') for row in manifest[split]}
    if {frame.identity for frame in frames} & reserved:
        raise ValueError('Property test molecules overlap training or validation.')
    if any(frame.reference != manifest['reference'] for frame in frames):
        raise ValueError('Property test labels use a different reference method.')
    assert_independent(member.parent, [frame_key(frame) for frame in frames], whole_molecules=True)
    output = predict_frames(manifest, member, frames, device=device, batch_size=batch_size)
    errors = []
    for row, frame in zip(output['frames'], frames):
        prediction = np.asarray(row[target_name(kind)])
        row['target'] = frame.targets.tolist()
        row['prediction'] = prediction.tolist()
        error = prediction - frame.targets
        errors.append(np.abs(error))
        if kind == 'charge':
            row['dipole_change_error_eA'] = float(np.linalg.norm(error[1] - error[0]))
        else:
            row['spectrum_overlap'] = spectra_overlap(frame.targets, prediction, frame.excitation_eV,
                                                       np.asarray(row['excitation_eV']))
    output['MAE'] = np.mean(errors, axis=0).tolist()
    if kind == 'charge':
        output['dipole_change_MAE_eA'] = float(np.mean([row['dipole_change_error_eA'] for row in output['frames']]))
    else:
        output['mean_spectrum_overlap'] = float(np.mean([row['spectrum_overlap'] for row in output['frames']]))
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    dipoles = commands.add_parser('label-dipoles')
    dipoles.add_argument('run', type=Path)
    dipoles.add_argument('output', type=Path)
    dipoles.add_argument('shards', type=Path, nargs='+')
    dipoles.add_argument('--frames-per-molecule', type=int, default=4)
    dipoles.add_argument('--step', type=float, default=1.e-4)
    dipoles.add_argument('--check-step', action='store_true')
    train = commands.add_parser('fit')
    train.add_argument('kind', choices=('oscillator', 'charge'))
    train.add_argument('run', type=Path)
    train.add_argument('member', type=Path)
    train.add_argument('output', type=Path)
    train.add_argument('--training', type=Path, nargs='+', required=True)
    train.add_argument('--validation', type=Path, nargs='+', required=True)
    train.add_argument('--epochs', type=int, default=300)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--device', default='cpu')
    test = commands.add_parser('score')
    test.add_argument('checkpoint', type=Path)
    test.add_argument('labels', type=Path, nargs='+')
    test.add_argument('--device', default='cpu')
    test.add_argument('--member', type=Path)
    test.add_argument('--output', type=Path)
    inference = commands.add_parser('predict')
    inference.add_argument('checkpoint', type=Path)
    inference.add_argument('structures', type=Path)
    inference.add_argument('--device', default='cpu')
    inference.add_argument('--member', type=Path)
    inference.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'label-dipoles':
        if args['frames_per_molecule'] < 1:
            parser.error('--frames-per-molecule must be positive.')
        accepted, rejected = label_dipoles(**args)
        print(f'{accepted} accepted, {rejected} rejected')
        if not accepted:
            raise SystemExit(1)
    elif command == 'fit':
        if min(args['epochs'], args['batch_size']) < 1:
            parser.error('--epochs and --batch-size must be positive.')
        print(f'Validation MSE: {fit(**args):.6g}')
    else:
        output = args.pop('output')
        report = score(**args) if command == 'score' else predict(**args)
        if output:
            write_report(output, report)
        else:
            print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
