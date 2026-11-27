"""Prepare cubic reference cells for the four end members and the alloy holdout."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.io import read, write
from scipy.optimize import minimize_scalar

from alframework.qm_interfaces.ase.vasp import vasp_input_for_structure


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / 'perovskite_bulk_oc25'
LEAD_PROTOTYPES = {'Br': 'mp-600089', 'I': 'mp-1069538'}
ALLOY_ORDERS = ('checkerboard', 'stripe')


def end_members():
    for halide, identifier in LEAD_PROTOTYPES.items():
        path = HERE.parent / 'surface_adsorption' / f'{identifier}.extxyz'
        prototype = read(path)
        lengths = prototype.cell.lengths()
        if (len(prototype) != 5 or not np.allclose(lengths, lengths[0])
                or not np.allclose(prototype.cell.angles(), 90)):
            raise ValueError(f'{path} must contain a five-atom cubic cell')
        for metal in ('Pb', 'Sn'):
            atoms = prototype.copy()
            atoms.set_cell(np.eye(3) * lengths[0], scale_atoms=True)
            atoms.symbols[atoms.symbols == 'Pb'] = metal
            atoms.info.clear()
            yield f'Cs{metal}{halide}3', atoms, {
                'prototype': identifier, 'prototype_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'prototype_metal': 'Pb', 'metal': metal,
            }


def relax_lattice(atoms, qm, scratch):
    """Minimize the static VASP free energy along the isotropic cubic lattice parameter."""
    from ase.calculators.vasp import Vasp

    a0 = atoms.cell.lengths()[0]
    lower, upper = a0 * np.array([.94, 1.06])
    trial = atoms.copy()
    trial.set_cell(np.eye(3) * lower, scale_atoms=True)
    options = vasp_input_for_structure(trial, {**qm, 'slab_dipole_correction': False}, {})
    if 'kpts' not in options:
        raise ValueError('Bulk relaxation needs kpoint_spacing_A_inv or fixed input.kpts.')
    # Hold the mesh fixed across volumes. The smallest cell sets the densest
    # mesh needed in the bracket under the same reciprocal-spacing rule.
    samples = []

    def energy(a):
        trial.set_cell(np.eye(3) * a, scale_atoms=True)
        trial.calc = Vasp(directory=str(scratch / f'point-{len(samples):03d}'),
                          command=qm['QM_run_command'], **options)
        value = float(trial.get_potential_energy(force_consistent=True))
        if not trial.calc.converged or not np.isfinite(value):
            raise RuntimeError(f'Bulk SCF failed at a={a:.6f} Å in {scratch}')
        samples.append({'a_A': float(a), 'free_energy_eV': value})
        return value

    edges = [energy(a) for a in (lower, upper)]
    result = minimize_scalar(energy, bounds=(lower, upper), method='bounded',
                             options={'xatol': .002, 'maxiter': 30})
    if (not result.success or not lower + .002 < result.x < upper - .002 or result.fun >= min(edges)):
        raise RuntimeError(f'Bulk energy minimum was not bracketed in {scratch}')
    relaxed = atoms.copy()
    relaxed.set_cell(np.eye(3) * result.x, scale_atoms=True)
    return relaxed, {'a_A': float(result.x), 'kpts': options['kpts'], 'samples': samples}


def alloy_cell(lead, tin, ordering):
    """Order B-site metals in the (100) plane of a pseudocubic 1×2×2 supercell."""
    a = .5 * (lead.cell.lengths()[0] + tin.cell.lengths()[0])
    primitive = lead.copy()
    primitive.set_cell(np.eye(3) * a, scale_atoms=True)
    mixed = primitive.repeat((1, 2, 2))
    metals = np.flatnonzero(mixed.symbols == 'Pb')
    rows, columns = np.floor(2 * mixed.get_scaled_positions()[metals, 1:] + 1e-8).astype(int).T
    if ordering == 'checkerboard':
        substitute = (rows + columns) % 2 == 1
    elif ordering == 'stripe':
        substitute = rows % 2 == 1
    else:
        raise ValueError(f'Unknown alloy ordering {ordering}')
    if len(metals) != 4 or np.count_nonzero(substitute) != 2:
        raise ValueError('Expected two Pb and two Sn sites in the alloy supercell')
    mixed.symbols[metals[substitute]] = 'Sn'
    return mixed, a


def prepare_bulk(output, *, qm, scratch, preview=False):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview and output == DEFAULT_OUTPUT:
        raise ValueError('Give preview cells a separate output directory.')
    report = {'geometry': 'cubic prototype preview' if preview else 'DFT-relaxed cubic end members',
              'vasp': None if preview else qm, 'training_end_members': {}, 'alloy_holdout': {}}
    cells = {}
    with TemporaryDirectory(dir=output.parent, prefix='.perovskite-bulk-') as temporary:
        stage = Path(temporary)
        for composition, atoms, provenance in end_members():
            if preview:
                relaxation = {'a_A': float(atoms.cell.lengths()[0])}
            else:
                atoms, relaxation = relax_lattice(atoms, qm, scratch / composition)
            write(stage / f'{composition}.extxyz', atoms)
            cells[composition] = atoms
            report['training_end_members'][composition] = {**provenance, **relaxation}
        for ordering in ALLOY_ORDERS:
            mixed, a = alloy_cell(cells['CsPbI3'], cells['CsSnI3'], ordering)
            write(stage / f'CsPbSnI3-{ordering}.extxyz', mixed)
            report['alloy_holdout'][ordering] = {'composition': 'CsPb0.5Sn0.5I3', 'pseudocubic_a_A': a,
                                                 'supercell': [1, 2, 2]}
        (stage / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scratch', type=Path, default=HERE / 'runs/perovskite_oc25/bulk')
    parser.add_argument('--qm-config', type=Path, default=HERE / 'vasp_perovskite.json')
    parser.add_argument('--preview', action='store_true', help='Use the unrelaxed prototypes for software checks.')
    args = parser.parse_args()
    qm = json.loads(args.qm_config.read_text())
    report = prepare_bulk(args.output, qm=qm, scratch=args.scratch.resolve(), preview=args.preview)
    print(json.dumps({name: row['a_A'] for name, row in report['training_end_members'].items()}, indent=2))


if __name__ == '__main__':
    main()
