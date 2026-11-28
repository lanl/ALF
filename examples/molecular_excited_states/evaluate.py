"""Label quarantined SMILES seeds and evaluate state-resolved checkpoints."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from ase import Atoms
import numpy as np

from alframework.builders.external_geometry import build_external_geometries
from alframework.ml_interfaces.hippynn.interface import (
    load_excited_state_h5_arrays,
    load_hippynn_excited_state_ensemble,
)
from alframework.tools.data_shards import store_current_data
from alframework.tools.molecules import MOLECULE_ID

if __package__:
    from .prepare import PROPERTIES
else:
    from prepare import PROPERTIES


def panel(run):
    return json.loads((run / 'panel_manifest.json').read_text())['molecules']


def run_properties(run):
    return json.loads((Path(run) / 'master_config.json').read_text())['properties_list']


def label_holdout(run, output, *, frames_per_molecule=4, split='holdout'):
    from alframework.qm_interfaces.pyscf.gpu4pyscf import label_gpu4pyscf_excited_state_molecule

    if output.exists():
        raise FileExistsError(output)
    master = json.loads((run / 'master_config.json').read_text())
    training_directory = (run / Path(master['h5_path']).parent).resolve()
    if output.resolve().is_relative_to(training_directory):
        raise ValueError('Save independent labels outside the training HDF5 directory.')
    qm = json.loads((run / 'QM_config.json').read_text())
    properties = master['properties_list']
    holdout = [row for row in panel(run) if row['split'] == split]
    if not holdout:
        raise ValueError(f'This panel has no {split} molecules.')
    ids = [f'{split}-{i:010d}' for i in range(frames_per_molecule * len(holdout))]
    seeds = build_external_geometries(ids, {
        'sources': [row['seed_path'] for row in holdout], 'selection_mode': 'sequential_interleave',
    }, run)
    for molecule in seeds:
        label_gpu4pyscf_excited_state_molecule(molecule, QM_config=qm, properties_list=properties)
        if not molecule.check_convergence():
            raise RuntimeError(f'GPU4PySCF failed for {molecule.get_moleculeid()}.')
    store_current_data(str(output), seeds, properties)


def trajectory_id(identifier):
    return re.sub(r'(?:-s|:step-)\d+$', '', identifier)


def assert_independent(ensemble, test_frames, *, whole_molecules):
    field = MOLECULE_ID if whole_molecules else 'id'
    key = (lambda x: x) if whole_molecules else trajectory_id
    test = {key(row[field]) for row in test_frames}
    for member in sorted(ensemble.glob('model-*/')):
        manifest = json.loads((member / 'training_input_manifest.json').read_text())
        # Internal validation/model selection are also excluded from this paper benchmark.
        seen = {key(row[field]) for row in manifest['frames']}
        overlap = sorted(seen & test)
        if overlap:
            raise ValueError(f'External benchmark overlaps model data for {field}: {overlap}')


def evaluate(ensemble, shards, *, device, whole_molecules=True, batch_size=16, properties=PROPERTIES):
    arrays, summary = load_excited_state_h5_arrays(
        None, properties, mixed_molecules=True, shards=[str(path.resolve()) for path in shards])
    assert_independent(ensemble, summary['frames'], whole_molecules=whole_molecules)
    model = load_hippynn_excited_state_ensemble(ensemble, device, properties)
    states = summary['state_table']
    groups = defaultdict(list)
    for start in range(0, len(summary['frames']), batch_size):
        frames = summary['frames'][start:start + batch_size]
        structures = []
        for index in range(start, start + len(frames)):
            keep = arrays['species'][index] > 0
            structures.append(Atoms(numbers=arrays['species'][index, keep],
                                    positions=arrays['coordinates'][index, keep]))
        prediction = model.predict_states(structures)
        energies = prediction['state_energies'].mean(axis=0).T
        forces = prediction['state_forces']
        for i, (frame, atoms) in enumerate(zip(frames, structures)):
            index = start + i
            expected = np.array([arrays[row['energy_db_name']][index] / properties[row['energy_key']][2]
                                 for row in states])
            force_error = [np.abs(forces[i][:, state].mean(axis=0) - arrays[row['force_db_name']][index, :len(atoms)]
                                  / properties[row['force_key']][2]).mean() for state, row in enumerate(states)]
            groups[frame[MOLECULE_ID]].append((expected, energies[i], force_error))
    report = {}
    for identity, rows in sorted(groups.items()):
        reference, predicted, force_error = map(np.stack, zip(*rows))
        energy_error = predicted - reference
        relative_error = energy_error - energy_error[np.argmin(reference[:, 0])]
        report[identity] = {
            'frames': len(rows),
            'state_energy_MAE_eV': np.abs(energy_error).mean(axis=0).tolist(),
            'state_relative_energy_MAE_eV': np.abs(relative_error).mean(axis=0).tolist() if len(rows) > 1 else None,
            'state_force_MAE_eV_A': force_error.mean(axis=0).tolist(),
            'gap_MAE_eV': float(np.abs(energy_error[:, 1] - energy_error[:, 0]).mean()),
            'adjacent_gap_MAE_eV': np.abs(np.diff(energy_error, axis=1)).mean(axis=0).tolist(),
        }
    return report


def compare_at_equal_budget(models, shards, *, device, whole_molecules=True, timelines=None, properties=None):
    """Compare checkpoints trained with the same number of accepted QM labels."""
    from alframework.tools.task_timeline import read_timeline, stage_spans

    counts = {}
    for policy, directory in models.items():
        members = sorted(directory.glob('model-*/'))
        sizes = {len(json.loads((member / 'training_input_manifest.json').read_text())['frames'])
                 for member in members}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(f'{directory}: expected model members with one shared positive label budget')
        counts[policy] = sizes.pop()
    if len(set(counts.values())) != 1:
        raise ValueError(f'Choose checkpoints at equal accepted QM label counts. Found {counts}')
    report = {'accepted_qm_labels': next(iter(counts.values())), 'policies': {}}
    for policy, directory in models.items():
        options = {} if properties is None else {'properties': properties}
        holdout = evaluate(directory, shards, device=device, whole_molecules=whole_molecules, **options)
        result = {
            'holdout': holdout,
            'mean_over_molecules': {
                'S1_force_MAE_eV_A': float(np.mean([row['state_force_MAE_eV_A'][1] for row in holdout.values()])),
                'S1_S0_gap_MAE_eV': float(np.mean([row['gap_MAE_eV'] for row in holdout.values()])),
            },
        }
        if all('adjacent_gap_MAE_eV' in row for row in holdout.values()):
            result['mean_over_molecules'].update(
                state_force_MAE_eV_A=np.mean([row['state_force_MAE_eV_A'] for row in holdout.values()], axis=0).tolist(),
                adjacent_gap_MAE_eV=np.mean([row['adjacent_gap_MAE_eV'] for row in holdout.values()], axis=0).tolist(),
            )
        if timelines is not None:
            records = read_timeline(timelines[policy])
            if not records:
                raise ValueError(f'No task timing records in {timelines[policy]}')
            result['timing'] = {
                'stages': stage_spans(records),
                'ntk_fit_seconds_in_sampler': sum(record.get('phases', {}).get('ntk_fit', 0.) for record in records),
            }
        report['policies'][policy] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    label = commands.add_parser('label-holdout')
    label.add_argument('run', type=Path)
    label.add_argument('output', type=Path)
    label.add_argument('--frames-per-molecule', type=int, default=4)
    label.add_argument('--split', choices=('holdout', 'validation'), default='holdout')
    score = commands.add_parser('score')
    score.add_argument('model', type=Path)
    score.add_argument('shards', type=Path, nargs='+')
    score.add_argument('--device', default='cpu')
    score.add_argument('--run', type=Path, help='Prepared run with the state schema.')
    score.add_argument('--trajectory-holdout', action='store_true')
    compare = commands.add_parser('compare')
    compare.add_argument('shards', type=Path, nargs='+')
    for policy in ('ensemble', 'ntk', 'random'):
        compare.add_argument(f'--{policy}', type=Path, required=True)
    compare.add_argument('--device', default='cpu')
    compare.add_argument('--run', type=Path, help='Prepared run with the shared state schema.')
    compare.add_argument('--trajectory-holdout', action='store_true')
    compare.add_argument('--timelines', type=Path, nargs=3, metavar=('ENSEMBLE', 'NTK', 'RANDOM'))
    args = parser.parse_args()
    if args.command == 'label-holdout':
        if args.frames_per_molecule < 1:
            parser.error('--frames-per-molecule must be positive')
        label_holdout(args.run, args.output, frames_per_molecule=args.frames_per_molecule, split=args.split)
    elif args.command == 'score':
        print(json.dumps(evaluate(args.model, args.shards, device=args.device,
                                  whole_molecules=not args.trajectory_holdout,
                                  properties=PROPERTIES if args.run is None else run_properties(args.run)),
                         indent=2, allow_nan=False))
    else:
        policies = ('ensemble', 'ntk', 'random')
        report = compare_at_equal_budget(
            {policy: getattr(args, policy) for policy in policies}, args.shards, device=args.device,
            whole_molecules=not args.trajectory_holdout,
            properties=None if args.run is None else run_properties(args.run),
            timelines=None if args.timelines is None else dict(zip(policies, args.timelines)),
        )
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
