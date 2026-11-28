"""Prepare and benchmark UMA OC25 runs for ligand adsorption on chalcogenides."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from ase.build import bulk
from ase.geometry import find_mic
from ase.io import read, write

from alframework.builders.surface import build_surface
from alframework.tools.molecules import MoleculesObject
from alframework.tools.tasks import alf_task

HERE = Path(__file__).resolve().parent
LATTICES = {'CdSe': 6.05, 'CdTe': 6.48, 'HgTe': 6.46, 'HgSe': 6.09}
LIGANDS = {'methylamine': 'NC', 'methylphosphine': 'PC'}
PROPERTIES = {'free_energy': ['energy', 'system', 1.0], 'forces': ['forces', 'atomic', 1.0]}
SLAB_METADATA = {'surface_builder': True, 'vacuum_axes': ['c']}
SLAB = {'miller_indices': [[1, 1, 0]], 'vacuum_A': 20., 'min_slab_thickness_A': 10.,
        'max_atoms': 200}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prototype_slabs(destination):
    """Create starting slabs for previews and bulk/surface convergence work."""
    with tempfile.TemporaryDirectory() as source:
        for material, lattice in LATTICES.items():
            crystal = Path(source) / f'{material}.extxyz'
            write(crystal, bulk(material, 'zincblende', a=lattice, cubic=True))
            settings = {**SLAB, 'bulk_file': str(crystal), 'min_in_plane_A': 8.,
                        'target_a_A': 10., 'target_b_A': 10., 'slab_symmetry': 'require'}
            slab = build_surface(f'{material}-clean', settings).get_atoms()
            write(destination / f'{material}.extxyz', slab)


def pose_pairs():
    return [(material, ligand) for material in LATTICES for ligand in LIGANDS]


def relax_slabs(run, output, *, fmax=.03, steps=150):
    """Relax clean prototype slabs at fixed cell with the configured reference."""
    from ase.calculators.vasp import Vasp
    from ase.optimize import BFGS
    from alframework.qm_interfaces.ase.vasp import vasp_input_for_structure

    if not np.isfinite(fmax) or fmax <= 0 or steps < 1:
        raise ValueError('Set positive relaxation steps and force tolerance.')
    qm = read_json(run / 'vasp_config.json')
    qm['slab_dipole_correction'] = False
    output.mkdir(parents=True, exist_ok=False)
    for material in LATTICES:
        atoms = read(run / 'slabs' / f'{material}.extxyz')
        inputs = vasp_input_for_structure(atoms, qm, SLAB_METADATA)
        atoms.calc = Vasp(directory=output / f'vasp-{material}', command=qm['QM_run_command'], **inputs)
        if (not BFGS(atoms, logfile=str(output / f'{material}.log')).run(fmax=fmax, steps=steps)
                or not atoms.calc.converged):
            raise RuntimeError(f'{material} did not relax to {fmax} eV/Å in {steps} steps.')
        write(output / f'{material}.extxyz', atoms, write_results=False)


def pose_geometry(identifier, prefix, material, ligand, root, seed):
    metal = material[:2]
    settings = {
        **SLAB, 'seed': seed, 'slab_file': f'slabs/{material}.extxyz',
        'adsorbates': [{'smiles': LIGANDS[ligand], 'relax_smiles': True, 'anchor_indices': [0],
                        'mode': 'binding_sites', 'site_types': ['ontop'], 'surface_elements': [metal],
                        'max_tilt_deg': 35., 'max_lateral_offset_A': .5, 'max_additional_height_A': .5}],
    }
    frame = build_surface(identifier, settings, root)
    atoms = frame.get_atoms()
    placement = frame.get_metadata()['surface_builder']
    anchor = placement['anchor_atom_index']
    atoms.info.update(frame_id=identifier, material=f'{material}_{ligand}', composition=material,
                      ligand=ligand, site_metal=metal, anchor=anchor, partition=prefix)
    atoms.info.update(pose_observables(atoms))
    return atoms


def pose_observables(atoms):
    anchor = int(atoms.info['anchor'])
    metal = np.flatnonzero(atoms.symbols == atoms.info['site_metal'])
    carbon = np.flatnonzero((atoms.get_tags() == 2) & (atoms.numbers == 6))
    if len(carbon) != 1 or not len(metal):
        raise ValueError('Expected a methyl ligand and at least one substrate binding metal.')
    axis, distance = find_mic(atoms.positions[carbon[0]] - atoms.positions[anchor], atoms.cell, atoms.pbc)
    return {'metal_donor_A': float(min(atoms.get_distances(anchor, metal, mic=True))),
            'donor_tilt_deg': float(np.rad2deg(np.arccos(np.clip(axis[2] / distance, -1, 1))))}


def pose_pool(pairs, count, prefix, root, seed):
    return [pose_geometry(f'{prefix}-{i:010d}', prefix, material, ligand, root, seed)
            for i, (material, ligand) in enumerate(pairs * count)]


@alf_task('builder')
def chalcogenide_builder_task(moleculeid, builder_config, master_directory=None, current_h5_id=0):
    """Interleave the finite bootstrap across pairs, then replay its reserved starts."""
    root = Path(master_directory or '.').resolve()
    source = root / builder_config['source']
    if current_h5_id:
        import hashlib

        digest = hashlib.sha256(f'{builder_config["seed"]}|{moleculeid}'.encode()).digest()
        index = int.from_bytes(digest[:8], 'big') % builder_config['n_frames']
    else:
        index = 0 if moleculeid == 'test_builder' else int(moleculeid.rsplit('-', 1)[-1])
    atoms = read(source, index=index, parallel=False)
    frame = MoleculesObject(atoms, moleculeid)
    frame.update_metadata({**SLAB_METADATA, 'geometry_source_id': atoms.info['frame_id'],
                           'surface_material': atoms.info['material']})
    return frame


def prepare(output, *, slabs=None, checkpoint=None, preview=False, vasp_command=None):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if preview == (slabs is not None):
        raise ValueError('Supply either --preview or a directory of relaxed --slabs.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.chalcogenide-', dir=output.parent) as temporary:
        root = Path(temporary) / 'run'
        slab_dir = root / 'slabs'
        slab_dir.mkdir(parents=True)
        if preview:
            prototype_slabs(slab_dir)
        else:
            for material in LATTICES:
                shutil.copyfile(slabs / f'{material}.extxyz', slab_dir / f'{material}.extxyz')
        for material in LATTICES:
            counts = Counter(read(slab_dir / f'{material}.extxyz').get_chemical_symbols())
            if set(counts) != {material[:2], material[2:]} or len(set(counts.values())) != 1:
                raise ValueError(f'Expected a stoichiometric {material} slab, received {counts}.')
        training = [(material, ligand) for material, ligand in pose_pairs() if material != 'HgSe']
        pools = {
            'seeds': pose_pool(training, 10, 'seed', root, 17),
            'fixed': pose_pool(training, 8, 'fixed', root, 18),
            'holdout': pose_pool(pose_pairs(), 8, 'holdout', root, 19),
        }
        # Use one cell per substrate across all poses. Preserve at least 20 Å of vacuum.
        for material in LATTICES:
            frames = [atoms for pool in pools.values() for atoms in pool if atoms.info['composition'] == material]
            height = max(atoms.cell[2, 2] for atoms in frames)
            for atoms in frames:
                atoms.cell[2, 2] = height
                atoms.center(axis=2)
        for name, frames in pools.items():
            write(root / f'{name}.extxyz', frames)
        master = read_json(HERE / 'master_perovskite_uma.json')
        master.update(builder_task='chalcogenide.chalcogenide_builder_task', builder_config_path='builder_config.json',
                      QM_config_path='vasp_config.json', ML_config_path='uma_config.json',
                      sampler_config_path='sampler_config.json', status_path='status.json',
                      h5_path='runs/h5/data-{:04d}.h5', model_path='runs/models/model-{:04d}',
                      QM_scratch_dir='runs/vasp', bootstrap_spares=12)
        qm = read_json(HERE / 'vasp_perovskite.json')
        if vasp_command:
            qm['QM_run_command'] = vasp_command
        ml = read_json(HERE / 'uma_perovskite.json')
        ml['fine_tune']['E0s'] = 'foundation'
        if checkpoint is not None:
            ml['fine_tune']['foundation_model'] = str(checkpoint.expanduser().resolve(strict=True))
        sampler = read_json(HERE / 'sampler_perovskite_uma.json')
        sampler['meta_dir'] = 'runs/sampling'
        for name, config in (
            ('master_config', master), ('vasp_config', qm), ('uma_config', ml), ('sampler_config', sampler),
            ('builder_config', {'source': 'seeds.extxyz', 'n_frames': len(pools['seeds']), 'seed': 17}),
            ('preparation', {'prototype_slabs': preview, 'bootstrap_frames': 48,
                             'pools': {name: [dict(atoms.info) for atoms in frames] for name, frames in pools.items()}}),
        ):
            write_json(root / f'{name}.json', config)
        for file in ('parsl_configs.py', 'chalcogenide.py'):
            shutil.copyfile(HERE / file, root / file)
        root.rename(output)
    return output


def label(run, pool, output, *, soc=False, count=None):
    """Label a saved pool using the ALF VASP task and write a benchmark or fixed-pool shard."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task
    from alframework.tools.data_shards import store_current_data

    if output.exists():
        raise FileExistsError(output)
    master = read_json(run / 'master_config.json')
    training_directory = (run / Path(master['h5_path']).parent).resolve()
    if output.resolve().is_relative_to(training_directory):
        raise ValueError('Save benchmark and fixed-pool labels outside the active training directory.')
    qm = read_json(run / 'vasp_config.json')
    if soc:
        qm['input'].update(lsorbit=True, isym=-1)
    atoms_list = read(run / f'{pool}.extxyz', index=':')
    if count is not None:
        if not 1 <= count <= len(atoms_list):
            raise ValueError(f'Choose a label count from 1 to {len(atoms_list)}.')
        atoms_list = atoms_list[:count]
    frames = []
    for atoms in atoms_list:
        frame = MoleculesObject(atoms, atoms.info['frame_id'])
        frame.update_metadata(SLAB_METADATA)
        VASP_ase_calculator_task.func(frame, qm, str(run / 'runs' / ('soc' if soc else 'benchmark')), PROPERTIES)
        if not frame.check_convergence():
            raise RuntimeError(f'VASP failed for {frame.get_moleculeid()}.')
        result = frame.get_results()
        atoms.calc = SinglePointCalculator(atoms, energy=result['free_energy'], forces=result['forces'])
        frames.append(frame)
    if output.suffix == '.h5':
        store_current_data(str(output), frames, PROPERTIES)
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
    atoms = read(run / 'holdout.extxyz', index=frame)
    if checkpoint is None:
        qm = read_json(run / 'vasp_config.json')
        calc = Vasp(directory=run / 'runs' / f'reference-md-{frame}-{seed}',
                    command=qm['QM_run_command'], **vasp_input_for_structure(atoms, qm, SLAB_METADATA))
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
    frames = read(run / 'holdout.extxyz', index=':')
    if not 2 <= count <= len(frames):
        raise ValueError(f'Choose a batch size from 2 to {len(frames)}.')
    output.mkdir(parents=True, exist_ok=False)
    ensemble = load_uma_batched_ensemble(checkpoint, 'oc25', device='cuda:0')
    baseline = read_json(run / 'sampler_config.json')
    report = {}
    for name, options in (('ordinary', {'well_params': None}),
                          ('udd', {'well_params': None, 'udd_bias_weight': udd_weight})):
        starts = [MoleculesObject(atoms.copy(), atoms.info['frame_id']) for atoms in frames[:count]]
        for start in starts:
            start.update_metadata(SLAB_METADATA)
        meta = output / name
        meta.mkdir()
        sampler = {**baseline, 'meta_dir': str(meta), 'trajectory_frequency': 1.,
                   'trajectory_interval': 1, 'MLMD_calculator_options': options}
        result = sample_trajectories(starts, ensemble, sampler, device='cuda:0')
        report[name] = {frame.get_moleculeid(): frame.get_metadata().get('trajectory_outcome')
                        for frame in result}
        for atoms in frames[:count]:
            identifier = atoms.info['frame_id']
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
    prep = commands.add_parser('prepare')
    prep.add_argument('output', type=Path)
    inputs = prep.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--slabs', type=Path)
    inputs.add_argument('--preview', action='store_true')
    prep.add_argument('--checkpoint', type=Path)
    prep.add_argument('--vasp-command')
    relaxation = commands.add_parser('relax')
    relaxation.add_argument('run', type=Path)
    relaxation.add_argument('output', type=Path)
    relaxation.add_argument('--fmax', type=float, default=.03)
    relaxation.add_argument('--steps', type=int, default=150)
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
    if command == 'prepare':
        print(prepare(**args))
    elif command == 'relax':
        relax_slabs(**args)
    elif command == 'label':
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
