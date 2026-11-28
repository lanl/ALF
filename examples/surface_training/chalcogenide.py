"""Benchmark UMA OC25 runs for ligand adsorption on chalcogenides."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

from alframework.builders.surface import build_surface
from alframework.tools.molecules import MoleculesObject

MASTER = 'master_chalcogenide_uma.json'
LIGANDS = {'NC': 'methylamine', 'PC': 'methylphosphine'}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pose_observables(atoms):
    anchor = int(atoms.info['anchor'])
    metal = np.flatnonzero(atoms.symbols == atoms.info['site_metal'])
    carbon = np.flatnonzero((atoms.get_tags() == 2) & (atoms.numbers == 6))
    if len(carbon) != 1 or not len(metal):
        raise ValueError('Expected a methyl ligand and at least one substrate binding metal.')
    axis, distance = find_mic(atoms.positions[carbon[0]] - atoms.positions[anchor], atoms.cell, atoms.pbc)
    return {'metal_donor_A': float(min(atoms.get_distances(anchor, metal, mic=True))),
            'donor_tilt_deg': float(np.rad2deg(np.arccos(np.clip(axis[2] / distance, -1, 1))))}


def benchmark_frames(run, pool, count=None):
    """Build a deterministic benchmark panel using the standard surface inputs."""
    master = read_json(run / MASTER)
    training = read_json(run / master['builder_config_path'])
    configs = [training]
    if pool == 'holdout':
        configs.append(read_json(run / 'builder_chalcogenide_holdout.json'))
    pairs = [
        (config, material, adsorbate)
        for config in configs
        for material in config['materials']
        for adsorbate in material['adsorbates']
    ] * 8
    if count is not None and not 1 <= count <= len(pairs):
        raise ValueError(f'Choose a frame count from 1 to {len(pairs)}.')
    frames = []
    for index, (config, material, adsorbate) in enumerate(pairs[:count]):
        identifier = f'{pool}-{index:010d}'
        settings = {**config, 'materials': [{**material, 'adsorbates': [adsorbate]}]}
        frame = build_surface(identifier, settings, run)
        placement = frame.get_metadata()['surface_builder']
        composition = placement['material_id']
        ligand = LIGANDS[placement['adsorbate_smiles']]
        atoms = frame.get_atoms()
        atoms.info.update(frame_id=identifier, material=f'{composition}_{ligand}', composition=composition,
                          ligand=ligand, site_metal=adsorbate['surface_elements'][0],
                          anchor=placement['anchor_atom_index'], partition=pool)
        atoms.info.update(pose_observables(atoms))
        frames.append(frame)
    return frames


def label(run, pool, output, *, soc=False, count=None):
    """Use the ALF VASP task to label a benchmark or fixed pool."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task
    from alframework.tools.data_shards import store_current_data

    if output.exists():
        raise FileExistsError(output)
    master = read_json(run / MASTER)
    training_directory = (run / Path(master['h5_path']).parent).resolve()
    if output.resolve().is_relative_to(training_directory):
        raise ValueError('Save benchmark and fixed-pool labels outside the active training directory.')
    qm = read_json(run / master['QM_config_path'])
    if soc:
        qm['input'].update(lsorbit=True, isym=-1)
    frames = benchmark_frames(run, pool, count)
    for frame in frames:
        VASP_ase_calculator_task.func(frame, qm, str(run / 'runs' / ('soc' if soc else 'benchmark')),
                                     master['properties_list'])
        if not frame.check_convergence():
            raise RuntimeError(f'VASP failed for {frame.get_moleculeid()}.')
        result = frame.get_results()
        atoms = frame.get_atoms()
        atoms.calc = SinglePointCalculator(atoms, energy=result['free_energy'], forces=result['forces'])
    if output.suffix == '.h5':
        store_current_data(str(output), frames, master['properties_list'])
    else:
        with output.open('x') as stream:
            write(stream, [frame.get_atoms() for frame in frames], format='extxyz')


def uma_calculator(checkpoint, device):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from ase.calculators.mixing import AverageCalculator

    paths = sorted(checkpoint.glob('model-*/uma.pt')) if checkpoint.is_dir() else [checkpoint]
    if not paths:
        raise ValueError(f'No UMA members under {checkpoint}.')
    models = [FAIRChemCalculator(load_predict_unit(str(path), device=device), task_name='oc25') for path in paths]
    return AverageCalculator(models)


def score(labels, checkpoint, device):
    if __package__:
        from .perovskite_benchmark import score as score_frames
    else:
        from perovskite_benchmark import score as score_frames

    return score_frames(read(labels, index=':'), uma_calculator(checkpoint, device))


def matched_trajectory(run, output, *, checkpoint, frame, seed, steps, temperature, device):
    from ase.calculators.vasp import Vasp
    from alframework.qm_interfaces.ase.vasp import vasp_input_for_structure
    if __package__:
        from .perovskite_benchmark import run_trajectory
    else:
        from perovskite_benchmark import run_trajectory

    if steps < 1 or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Choose positive steps and temperature.')
    start = benchmark_frames(run, 'holdout')[frame]
    atoms = start.get_atoms()
    if checkpoint is None:
        master = read_json(run / MASTER)
        qm = read_json(run / master['QM_config_path'])
        calc = Vasp(directory=run / 'runs' / f'reference-md-{frame}-{seed}',
                    command=qm['QM_run_command'],
                    **vasp_input_for_structure(atoms, qm, start.get_metadata()))
    else:
        calc = uma_calculator(checkpoint, device)
    with output.open('x') as stream:
        run_trajectory(atoms, stream, calc, steps=steps, dt=.5, temperature=temperature, seed=seed)


def paired_sampling(run, checkpoint, output, count, udd_weight):
    """Propagate ordinary and biased batches from identical poses and random streams."""
    from alframework.ml_interfaces.fine_tuning.uma import load_uma_batched_ensemble
    from alframework.samplers.batched_mlmd import sample_trajectories

    if not np.isfinite(udd_weight) or udd_weight >= 0:
        raise ValueError('Use a finite negative UDD weight to lower energy in high-spread regions.')
    if count < 2:
        raise ValueError('Choose a batch size of at least two.')
    frames = benchmark_frames(run, 'holdout', count)
    output.mkdir(parents=True, exist_ok=False)
    ensemble = load_uma_batched_ensemble(checkpoint, 'oc25', device='cuda:0')
    master = read_json(run / MASTER)
    baseline = read_json(run / master['sampler_config_path'])
    report = {}
    for name, options in (('ordinary', {'well_params': None}),
                          ('udd', {'well_params': None, 'udd_bias_weight': udd_weight})):
        starts = []
        for frame in frames:
            start = MoleculesObject(frame.get_atoms().copy(), frame.get_moleculeid())
            start.update_metadata(frame.get_metadata())
            starts.append(start)
        meta = output / name
        meta.mkdir()
        sampler = {**baseline, 'meta_dir': str(meta), 'trajectory_frequency': 1.,
                   'trajectory_interval': 1, 'MLMD_calculator_options': options}
        result = sample_trajectories(starts, ensemble, sampler, device='cuda:0')
        report[name] = {frame.get_moleculeid(): frame.get_metadata().get('trajectory_outcome')
                        for frame in result}
        for frame in frames:
            atoms = frame.get_atoms()
            identifier = frame.get_moleculeid()
            trajectory = next(meta.rglob(f'metadata-{identifier}.traj'), None)
            if trajectory is not None:
                images = read(trajectory, index=':')
                for image in images:
                    image.info.update(atoms.info)
                    image.set_tags(atoms.get_tags())
                write(meta / f'{identifier}.extxyz', images, write_results=False)
    write_json(output / 'paired_sampling.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    labels = commands.add_parser('label')
    labels.add_argument('run', type=Path)
    labels.add_argument('pool', choices=('holdout', 'fixed'))
    labels.add_argument('output', type=Path)
    labels.add_argument('--soc', action='store_true')
    labels.add_argument('--count', type=int)
    scoring = commands.add_parser('score')
    scoring.add_argument('labels', type=Path)
    scoring.add_argument('checkpoint', type=Path)
    scoring.add_argument('--device', default='cuda')
    matched = commands.add_parser('md')
    matched.add_argument('run', type=Path)
    matched.add_argument('output', type=Path)
    matched.add_argument('--checkpoint', type=Path, help='Run VASP when a model checkpoint is absent.')
    matched.add_argument('--frame', type=int, default=0)
    matched.add_argument('--steps', type=int, default=100)
    matched.add_argument('--seed', type=int, default=42)
    matched.add_argument('--temperature', type=float, default=300.)
    matched.add_argument('--device', default='cuda')
    paired = commands.add_parser('sample')
    paired.add_argument('run', type=Path)
    paired.add_argument('checkpoint', type=Path)
    paired.add_argument('output', type=Path)
    paired.add_argument('--count', type=int, default=4)
    paired.add_argument('--udd-weight', type=float, required=True)
    trajectory = commands.add_parser('trajectory')
    trajectory.add_argument('structures', type=Path, nargs='+')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'label':
        label(**args)
    elif command == 'score':
        print(json.dumps(score(**args), indent=2, allow_nan=False))
    elif command == 'md':
        matched_trajectory(**args)
    elif command == 'sample':
        paired_sampling(**args)
    else:
        observations = [pose_observables(atoms) for path in args['structures'] for atoms in read(path, index=':')]
        print(json.dumps({key: np.percentile([row[key] for row in observations], [10, 50, 90]).tolist()
                          for key in observations[0]}, indent=2))


if __name__ == '__main__':
    main()
