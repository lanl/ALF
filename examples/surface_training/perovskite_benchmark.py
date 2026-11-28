"""Prepare and score independent H2O/NH3 perovskite slab geometries."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import iread, read, write

from alframework.builders.surface import build_surface
from alframework.tools.molecules import MoleculesObject
from alframework.tools.velocities import has_fixed_atoms

if __package__:
    from .perovskite_bulk import ALLOY_ORDERS, DEFAULT_OUTPUT
else:
    from perovskite_bulk import ALLOY_ORDERS, DEFAULT_OUTPUT


HERE = Path(__file__).resolve().parent
SLAB_METADATA = {'surface_builder': True, 'vacuum_axes': ['c']}
METALS = ('Pb', 'Sn')


def config(name):
    return json.loads((HERE / name).read_text())


def termination(atoms):
    metal = np.isin(atoms.get_chemical_symbols(), METALS)
    metal_height = atoms.positions[metal, 2].max()
    cesium_height = atoms.positions[atoms.symbols == 'Cs', 2].max()
    return 'MX2' if metal_height > cesium_height else 'CsX'


def prepare_heldout(per_site=2, *, bulk_dir=DEFAULT_OUTPUT, partition='end-members'):
    """Balance end-member pose groups or alloy metal-atop groups."""
    builder = config('builder_perovskite.json')
    bulk_dir = Path(bulk_dir).resolve()
    if partition == 'end-members':
        for material in builder['materials']:
            material['bulk_file'] = str(bulk_dir / Path(material['bulk_file']).name)
            material['adsorbates'][0]['site_types'] = ['bridge', 'hollow']
        expected = {(material['id'], face, site) for material in builder['materials']
                    for face in ('CsX', 'MX2') for site in ('bridge', 'hollow')}
    elif partition == 'alloy':
        from copy import deepcopy

        ligands = [entry for entry in builder['materials'] if entry['id'].startswith('CsPbI3_')]
        builder['materials'] = [
            dict(deepcopy(entry), id=f'CsPbSnI3-{order}_{entry["id"].split("_")[1]}',
                 bulk_file=str(bulk_dir / f'CsPbSnI3-{order}.extxyz'))
            for order in ALLOY_ORDERS for entry in ligands
        ]
        expected = {(material['id'], 'MX2', metal) for material in builder['materials'] for metal in METALS}
    else:
        raise ValueError(f'Unknown benchmark partition {partition!r}')
    builder['seed'] += 1
    selected = defaultdict(list)
    for i in range(len(expected) * per_site * 128):
        candidate = build_surface(f'{partition}-{i:08d}', builder, HERE)
        atoms = candidate.get_atoms()
        metadata = candidate.get_metadata()['surface_builder']
        face = termination(atoms)
        site, site_metal = metadata['site_type'], None
        if partition == 'alloy':
            if face != 'MX2':
                continue
            exposed = np.flatnonzero(atoms.get_tags() == 1)
            _, distances = find_mic(atoms.positions[exposed] - metadata['site'], atoms.cell, atoms.pbc)
            site_metal = str(atoms.symbols[exposed[np.argmin(distances)]])
            if site_metal not in METALS or distances.min() > 1e-5:
                continue
        key = (metadata['material_id'], face, site_metal or site)
        if key not in expected:
            continue
        if len(selected[key]) >= per_site:
            continue
        atoms.info.update(frame_id=candidate.get_moleculeid(), material=key[0], termination=key[1],
                          site=site, anchor=int(metadata['anchor_atom_index']), partition=partition)
        if site_metal is not None:
            atoms.info['site_metal'] = site_metal
        selected[key].append(atoms)
        if sum(map(len, selected.values())) == len(expected) * per_site:
            return [atoms for key in sorted(selected) for atoms in selected[key]]
    missing = sorted(key for key in expected if len(selected[key]) < per_site)
    raise RuntimeError(f'Could not generate enough frames for held-out pose groups {missing}')


def label_heldout(frames, qm_config, scratch):
    from alframework.qm_interfaces.ase.vasp import VASP_ase_calculator_task

    properties = config('master_perovskite_uma.json')['properties_list']
    for atoms in frames:
        frame = MoleculesObject(atoms, atoms.info['frame_id'])
        # Apply the same per-slab reciprocal mesh and dipole treatment as during ALF.
        frame.update_metadata(SLAB_METADATA)
        frame = VASP_ase_calculator_task.func(frame, qm_config, str(scratch), properties)
        if not frame.check_convergence():
            raise RuntimeError(f'VASP failed to converge for {frame.get_moleculeid()}.')
        results = frame.get_results()
        atoms.calc = SinglePointCalculator(atoms, energy=results['free_energy'], forces=results['forces'])
        yield atoms


def score(frames, calculator, near_radius=3.5):
    """Compare forces and relative energies within fixed compositions."""
    groups = defaultdict(list)
    for reference in frames:
        predicted = reference.copy()
        predicted.calc = calculator
        error = np.abs(predicted.get_forces() - reference.get_forces())
        adsorbate = np.flatnonzero(reference.get_tags() == 2)
        if not len(adsorbate):
            raise ValueError('The held-out frames must identify adsorbate atoms with ASE tag 2.')
        substrate = np.flatnonzero(reference.get_tags() != 2)
        close = np.zeros(len(reference), dtype=bool)
        for anchor in adsorbate:
            close[substrate] |= reference.get_distances(anchor, substrate, mic=True) <= near_radius
        if not np.any(close):
            raise ValueError('The near-adsorbate region contains no substrate atoms.')
        key = (reference.info['material'], reference.get_chemical_formula())
        groups[key].append(dict(
            reference_eV=reference.get_potential_energy(), model_eV=predicted.get_potential_energy(),
            site_metal=reference.info.get('site_metal'),
            adsorbate_error_eV_A=error[adsorbate].mean(),
            near_slab_error_eV_A=error[close].mean(),
            all_force_error_eV_A=error.mean(),
        ))
    report = {}
    for key, cases in groups.items():
        reference_site = min(cases, key=lambda case: case['reference_eV'])
        for case in cases:
            case['relative_energy_error_eV'] = (None if case is reference_site else
                                                abs((case['model_eV'] - reference_site['model_eV'])
                                                    - (case['reference_eV'] - reference_site['reference_eV'])))
        summary = score_subset(cases)
        groups_by_metal = {metal: [case for case in cases if case['site_metal'] == metal] for metal in METALS}
        if any(groups_by_metal.values()):
            summary['by_site_metal'] = {metal: score_subset(subset) for metal, subset in groups_by_metal.items()
                                        if subset}
        report['/'.join(key)] = summary
    return report


def score_subset(cases):
    energy_errors = [case['relative_energy_error_eV'] for case in cases
                     if case['relative_energy_error_eV'] is not None]
    return dict(frames=len(cases), relative_energy_MAE_eV=float(np.mean(energy_errors)) if energy_errors else None,
                **{key: float(np.mean([case[key] for case in cases]))
                   for key in ('adsorbate_error_eV_A', 'near_slab_error_eV_A', 'all_force_error_eV_A')})


def summarize_trajectory(frames):
    """Report structural observables from a trajectory with the adsorbate tags retained."""
    distances, tilts = defaultdict(list), []
    for atoms in frames:
        anchor = int(atoms.info['anchor'])
        hydrogen = np.flatnonzero((atoms.get_tags() == 2) & (atoms.numbers == 1))
        for metal in METALS:
            indices = np.flatnonzero(atoms.symbols == metal)
            if len(indices):
                distances[metal].append(float(min(atoms.get_distances(anchor, indices, mic=True))))
        axis = atoms.get_distances(anchor, hydrogen, vector=True, mic=True).mean(axis=0)
        tilts.append(float(np.degrees(np.arccos(np.clip(axis[2] / np.linalg.norm(axis), -1., 1.)))))
    if not distances:
        raise ValueError('The trajectory contains no frames.')
    return {'frames': len(tilts), 'percentiles': [10, 50, 90],
            **{f'{metal}_donor_distance_A': np.percentile(values, [10, 50, 90]).tolist()
               for metal, values in distances.items()},
            'donor_axis_tilt_deg': np.percentile(tilts, [10, 50, 90]).tolist()}


def calculator(backend, checkpoint, device, *, uma_task=None):
    if backend == 'uma':
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        task = uma_task or config('uma_perovskite.json')['fine_tune']['task']
        return FAIRChemCalculator(load_predict_unit(str(checkpoint), device=device), task_name=task)
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(checkpoint), device=device, head='oc20_usemppbe')


def run_trajectory(atoms, stream, calc, *, steps, dt, temperature, seed):
    from ase import units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = atoms.copy()
    atoms.calc = calc
    if not has_fixed_atoms(atoms):
        atoms.set_constraint(FixCom())
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    dynamics = Langevin(atoms, dt * units.fs, temperature_K=temperature, friction=.01 / units.fs,
                        fixcm=False, rng=rng)

    def save_frame():
        if getattr(calc, 'converged', True) is False:
            raise RuntimeError(f'SCF failed to converge at step {dynamics.nsteps}')
        atoms.info['md_step'] = dynamics.nsteps
        write(stream, atoms, format='extxyz', write_results=False)

    dynamics.attach(save_frame, interval=1)
    dynamics.run(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('output', type=Path)
    prepare.add_argument('--per-site', type=int, default=2)
    prepare.add_argument('--partition', choices=('end-members', 'alloy'), default='end-members')
    prepare.add_argument('--bulk-dir', type=Path, default=DEFAULT_OUTPUT)
    label = commands.add_parser('label')
    label.add_argument('structures', type=Path)
    label.add_argument('output', type=Path)
    label.add_argument('--scratch', type=Path, required=True)
    test = commands.add_parser('score')
    test.add_argument('structures', type=Path)
    test.add_argument('backend', choices=('uma', 'mace'))
    test.add_argument('checkpoint', type=Path)
    test.add_argument('--device', default='cuda')
    test.add_argument('--uma-task', choices=('oc20', 'oc25'), help='UMA task, defaults to the training config.')
    dynamics = commands.add_parser('trajectory')
    dynamics.add_argument('structures', type=Path)
    md = commands.add_parser('md')
    md.add_argument('structures', type=Path)
    md.add_argument('backend', choices=('uma', 'mace', 'vasp'))
    md.add_argument('output', type=Path)
    md.add_argument('--checkpoint', type=Path)
    md.add_argument('--scratch', type=Path)
    md.add_argument('--device', default='cuda')
    md.add_argument('--uma-task', choices=('oc20', 'oc25'), help='UMA task, defaults to the training config.')
    md.add_argument('--frame', type=int, default=0)
    md.add_argument('--steps', type=int, default=100)
    md.add_argument('--dt-fs', type=float, default=.5)
    md.add_argument('--temperature-K', type=float, default=300)
    md.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.per_site < 1:
            parser.error('--per-site must be positive')
        frames = prepare_heldout(args.per_site, bulk_dir=args.bulk_dir, partition=args.partition)
        with args.output.open('x') as stream:
            write(stream, frames, format='extxyz')
    elif args.command == 'label':
        frames = label_heldout(iread(args.structures), config('vasp_perovskite.json'), args.scratch)
        with args.output.open('x') as stream:
            for atoms in frames:
                write(stream, atoms, format='extxyz', append=True)
    elif args.command == 'score':
        calc = calculator(args.backend, args.checkpoint, args.device, uma_task=args.uma_task)
        print(json.dumps(score(iread(args.structures), calc), indent=2, allow_nan=False))
    elif args.command == 'md':
        if args.steps < 1 or not np.isfinite([args.dt_fs, args.temperature_K]).all() or min(args.dt_fs, args.temperature_K) <= 0:
            parser.error('Provide positive steps, time step, and temperature.')
        atoms = read(args.structures, index=args.frame)
        if args.backend == 'vasp':
            if args.scratch is None:
                parser.error('VASP MD requires --scratch.')
            from ase.calculators.vasp import Vasp
            from alframework.qm_interfaces.ase.vasp import vasp_input_for_structure

            qm = config('vasp_perovskite.json')
            calc = Vasp(directory=str(args.scratch), command=qm['QM_run_command'],
                        **vasp_input_for_structure(atoms, qm, SLAB_METADATA))
        else:
            if args.checkpoint is None:
                parser.error('Model MD requires --checkpoint.')
            calc = calculator(args.backend, args.checkpoint, args.device, uma_task=args.uma_task)
        with args.output.open('x') as stream:
            run_trajectory(atoms, stream, calc, steps=args.steps, dt=args.dt_fs,
                           temperature=args.temperature_K, seed=args.seed)
    else:
        print(json.dumps(summarize_trajectory(iread(args.structures)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
