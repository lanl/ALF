"""Bundle a converged CPPn S0 optimization as references/cppN_s0.{xyz,json}.

python references/import_s0_reference.py /path/to/campaign/cppN

The source is one ring directory written by the TDDFT ``optimize_cpp_s0.py``
driver: ``ground_state.json`` (full-precision coordinates, gradient and energy)
beside the campaign's ``effective_config.json`` one level up. The optimization
must be converged with the theory in this example's ``QM_config.json``. The
script is CPU-only, runs no QM, and never overwrites a bundled reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
# Same constants as the bundled cpp6/cpp9 references.
HARTREE_TO_EV = 27.21138602
BOHR_TO_ANGSTROM = 0.52917721092
THEORY_KEYS = ('xc', 'basis', 'charge', 'multiplicity', 'density_fit', 'auxbasis',
               'grids_level', 'scf_conv_tol', 'scf_max_cycle')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_reference(source, destination=HERE):
    sys.path.insert(0, str(EXAMPLE))
    from initial_sampling import DEFAULT_SETTINGS, infer_topology, screen_geometry

    source = Path(source).expanduser().resolve(strict=True)
    record_path = source/'ground_state.json'
    record = json.loads(record_path.read_text())
    campaign = json.loads((source.parent/'effective_config.json').read_text())['qm']
    qm = json.loads((EXAMPLE/'QM_config.json').read_text())
    if not (record['scf_converged'] and record['optimizer_converged'] and record['force_criteria_passed']):
        raise ValueError(f'{source}: S0 optimization is not converged')
    for key in THEORY_KEYS:
        if campaign[key] != qm[key]:
            raise ValueError(f'{source}: {key}={campaign[key]!r} differs from QM_config.json ({qm[key]!r})')
    source_atoms = read(source/'optimized_reference.xyz', parallel=False)
    positions = np.asarray(record['coords_angstrom'], dtype=np.float64)
    gradient = np.asarray(record['gradient_hartree_per_bohr'], dtype=np.float64)
    if positions.shape != (len(source_atoms), 3) or gradient.shape != positions.shape:
        raise ValueError('Saved coordinates or gradient have the wrong shape')
    if not (np.isfinite(positions).all() and np.isfinite(gradient).all()):
        raise ValueError('Saved coordinates or gradient are not finite')
    # The extxyz export is rounded; the JSON coordinates are the full-precision ones.
    if not np.allclose(positions, source_atoms.positions, rtol=0, atol=1e-7):
        raise ValueError('ground_state.json coordinates disagree with optimized_reference.xyz')
    atoms = Atoms(numbers=source_atoms.numbers, positions=positions)
    topology = infer_topology(atoms, DEFAULT_SETTINGS['screening'])
    ring = topology['nring']
    if screen_geometry(positions, topology, DEFAULT_SETTINGS['screening']):
        raise ValueError(f'cpp{ring} reference fails the sampling geometry screen')
    xyz, info = Path(destination)/f'cpp{ring}_s0.xyz', Path(destination)/f'cpp{ring}_s0.json'
    if xyz.exists() or info.exists():
        raise FileExistsError(f'Refusing to overwrite the bundled cpp{ring} reference')

    write(xyz, atoms, format='xyz')
    if not np.allclose(read(xyz, parallel=False).positions, positions, rtol=0, atol=1e-12):
        raise RuntimeError('Written reference does not round-trip')
    # Component-wise max/RMS, the convention of the bundled cpp6/cpp9 records.
    force = gradient*HARTREE_TO_EV/BOHR_TO_ANGSTROM
    energy = float(record['energy_hartree'])
    # Same keys as the bundled cpp6/cpp9 theory blocks.
    theory = {key: qm[key] for key in sorted(qm) if key not in {'solvent', 'compute_nacr'}}
    theory.update(energy_offset_eV=0.0, **{key: campaign[key] for key in THEORY_KEYS})
    reference = {
        'ring_size': ring,
        'reference': xyz.name,
        'reference_sha256': _sha256(xyz),
        'source': '/'.join(source.parts[source.parts.index('ml_4_excited_states'):])
                  if 'ml_4_excited_states' in source.parts else str(source),
        'source_result_sha256': _sha256(record_path),
        'success': True,
        'optimizer_converged': True,
        'selected_state': 0,
        'ground_energy_hartree': energy,
        'hartree_to_ev': HARTREE_TO_EV,
        'energy_offset_eV': energy*HARTREE_TO_EV,
        'gradient_max_ev_per_angstrom': float(np.abs(force).max()),
        'gradient_rms_ev_per_angstrom': float(np.sqrt(np.mean(force**2))),
        'theory': theory,
    }
    info.write_text(json.dumps(reference, indent=2)+'\n')
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='+', type=Path, help='cppN directories from optimize_cpp_s0.py')
    parser.add_argument('--destination', type=Path, default=HERE)
    args = parser.parse_args(argv)
    for source in args.source:
        reference = import_reference(source, args.destination)
        print(f"cpp{reference['ring_size']}: {reference['ground_energy_hartree']} Ha, "
              f"offset {reference['energy_offset_eV']} eV")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
