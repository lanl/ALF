"""Qualify singlet TDA roots and gradients in two bases on fixed geometries."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter

import numpy as np
from ase import Atoms, units
from ase.optimize import BFGS
from scipy.optimize import linear_sum_assignment

from alframework.qm_interfaces.pyscf.calculator import PySCFCalculator
from alframework.qm_interfaces.pyscf.gpu4pyscf import validate_gpu4pyscf_config


BASES = ('6-31g*', '6-31+g*')
NROOTS = 3


def reference_geometries(reference, seed_path, settings, *, stretch_A, fmax):
    """Relax the ground state and take a precomputed distortion or a C=O stretch."""
    import h5py
    from rdkit import Chem

    molecule = Chem.MolFromMolFile(str(reference), removeHs=False)
    if molecule is None:
        raise ValueError(f'Could not read the prepared reference {reference}')
    sites = molecule.GetSubstructMatches(Chem.MolFromSmarts('[#6]=[#8]'))
    atoms = Atoms(numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                  positions=molecule.GetConformer().GetPositions())
    keys = ('xc', 'charge', 'multiplicity', 'num_threads', 'grids_level', 'scf_conv_tol', 'scf_max_cycle')
    atoms.calc = PySCFCalculator(basis=BASES[0], density_fit=False,
                                 **{key: settings[key] for key in keys})
    if not BFGS(atoms, logfile=None).run(fmax=fmax, steps=150):
        raise RuntimeError(f'Ground-state relaxation did not reach {fmax} eV/Å')
    equilibrium = atoms.copy()
    if not sites:
        with h5py.File(seed_path) as handle:
            group = handle['molecule']
            index = int(np.argmax(group['strata'][:, 0]))
            displaced = Atoms(symbols=[item.decode() for item in group['species'][:]],
                              positions=group['coordinates'][index])
        return (), {'equilibrium': equilibrium, 'builder_distortion': displaced}
    displaced = equilibrium.copy()
    carbon, oxygen = sites[0]
    # Move directly along the bond and keep the midpoint and remaining atoms fixed.
    displaced.set_distance(carbon, oxygen, stretch_A, add=True, fix=0.5)
    return (carbon, oxygen), {'equilibrium': equilibrium, 'carbonyl_stretch': displaced}


def solve_tda(atoms, basis, settings, nroots, field=None):
    """Run conventional CPU RKS/TDA, optionally under an electric field in atomic units."""
    from pyscf import dft, gto

    molecule = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis=basis,
                     unit='Angstrom', charge=0, spin=0, verbose=0)
    scf = dft.RKS(molecule, xc=settings['xc'])
    scf.grids.level = settings['grids_level']
    scf.conv_tol = settings['scf_conv_tol']
    scf.max_cycle = settings['scf_max_cycle']
    scf.chkfile = None
    if field is not None:
        electronic = np.einsum('a,aij->ij', field, molecule.intor_symmetric('int1e_r', comp=3))
        hcore = scf.get_hcore() + electronic
        scf.get_hcore = lambda *args: hcore
    scf.kernel()
    if not scf.converged:
        raise RuntimeError(f'SCF did not converge in {basis}')
    solver = scf.TDA()
    solver.singlet = True
    solver.nstates = nroots
    solver.conv_tol = settings['tda_conv_tol']
    solver.max_cycle = settings['tda_max_cycle']
    # CPU Davidson uses a different orthogonalization implementation than the GPU solver.
    solver.lindep = max(solver.lindep, settings['tda_lindep'])
    solver.kernel()
    if np.shape(solver.e) != (nroots,) or not np.all(solver.converged) or not np.isfinite(solver.e).all():
        raise RuntimeError(f'Expected {nroots} converged singlet roots in {basis}. '
                           f'Returned {len(solver.e)}, convergence flags {np.asarray(solver.converged).tolist()}.')
    return molecule, scf, solver


def root_data(molecule, scf, solver):
    energies = np.asarray(solver.e) * units.Hartree
    strengths = np.asarray(solver.oscillator_strength())
    if not np.isfinite(strengths).all():
        raise RuntimeError(f'Nonfinite oscillator strengths in {molecule.basis}')
    return molecule, scf.mo_coeff[:, scf.mo_occ > 0], scf.mo_coeff[:, scf.mo_occ == 0], np.array(
        [x / np.linalg.norm(x) for x, _ in solver.xy]), energies, strengths


def compare_roots(left, right, *, min_overlap, gap_shift_eV):
    from pyscf import gto

    mol_l, occ_l, vir_l, amplitudes_l, energies_l, strengths_l = left
    mol_r, occ_r, vir_r, amplitudes_r, energies_r, strengths_r = right
    # Contract normalized TDA transition densities through the cross-basis AO
    # overlap. The absolute value removes the arbitrary phase of each root.
    ao = gto.intor_cross('int1e_ovlp', mol_l, mol_r)
    occ_overlap = occ_l.T @ ao @ occ_r
    vir_overlap = vir_l.T @ ao @ vir_r
    overlap = np.abs(np.einsum('nia,ij,mjb,ab->nm', amplitudes_l, occ_overlap,
                               amplitudes_r, vir_overlap, optimize=True))
    if not np.isfinite(overlap).all():
        raise RuntimeError('Nonfinite transition-density overlaps')
    indices, matched = linear_sum_assignment(overlap, maximize=True)
    overlaps = overlap[indices, matched]
    shifts = energies_r[matched] - energies_l
    flags = []
    if np.any(indices != matched):
        flags.append('root_order_changed')
    if np.any(overlaps < min_overlap):
        flags.append('transition_density_overlap_low')
    if np.any(np.abs(shifts) > gap_shift_eV):
        flags.append('vertical_energy_shift_large')
    return dict(energies_eV={BASES[0]: energies_l.tolist(), BASES[1]: energies_r.tolist()},
                oscillator_strengths={BASES[0]: strengths_l.tolist(), BASES[1]: strengths_r.tolist()},
                root_matches_1_based=(matched + 1).tolist(), overlap_matrix=overlap.tolist(),
                matched_energy_shifts_eV=shifts.tolist(), flags=flags,
                s1_basis_sensitive=bool(matched[0] != 0 or overlaps[0] < min_overlap
                                        or abs(shifts[0]) > gap_shift_eV))


def qualify(run, *, all_molecules=False, threads=1, stretch_A=.08, fmax=.03, min_overlap=.8,
            gap_shift_eV=.2, nroots=NROOTS, gradients=False, gpu_labels=False):
    from pyscf import __version__ as pyscf_version, lib

    entries = json.loads((run / 'panel_manifest.json').read_text())['molecules']
    selected = [row for row in entries if row['split'] == 'train' and (all_molecules or row.get('qualification'))]
    if not selected:
        raise ValueError('Mark panel representatives with qualification=true or pass --all.')
    qm = json.loads((run / 'QM_config.json').read_text())
    settings = validate_gpu4pyscf_config({**qm, 'num_threads': threads})
    lib.num_threads(threads)
    report = dict(xc=settings['xc'], bases=BASES, n_singlet_roots=nroots, density_fit=False,
                  pyscf_version=pyscf_version, stretch_A=stretch_A, equilibrium_fmax_eV_A=fmax,
                  min_overlap=min_overlap, max_matched_energy_shift_eV=gap_shift_eV,
                  passed=[], flagged=[], failed=[])
    for entry in selected:
        row = {key: entry[key] for key in ('name', 'family', 'molecule_id', 'split') if key in entry}
        print(f'Checking {row.get("name", row["molecule_id"])}', file=sys.stderr, flush=True)
        try:
            bond, geometries = reference_geometries(run / entry['reference_path'], run / entry['seed_path'],
                                                    settings, stretch_A=stretch_A, fmax=fmax)
        except (RuntimeError, ValueError) as exc:
            report['failed'].append(dict(row, stage='geometry', error=str(exc)))
            continue
        for name, atoms in geometries.items():
            geometry = dict(row, geometry=name, carbonyl_atom_indices=list(bond),
                            symbols=atoms.get_chemical_symbols(), positions_A=atoms.positions.tolist())
            try:
                start = perf_counter()
                solutions = [solve_tda(atoms, basis, settings, nroots) for basis in BASES]
                result = compare_roots(*(root_data(*items) for items in solutions),
                                       min_overlap=min_overlap, gap_shift_eV=gap_shift_eV)
                if gradients:
                    derivatives = [np.stack([td.nuc_grad_method().kernel(state=i)
                                             for i in range(1, nroots + 1)]) for _, _, td in solutions]
                    right = derivatives[1][np.asarray(result['root_matches_1_based']) - 1]
                    result['matched_gradient_rms_difference_eV_A'] = (
                        np.sqrt(np.mean((derivatives[0] - right) ** 2, axis=(1, 2)))
                        * units.Hartree / units.Bohr).tolist()
                result['cpu_seconds'] = perf_counter() - start
                if gpu_labels:
                    result['gpu_labels'] = {basis: time_gpu_label(atoms, basis, qm, nroots) for basis in BASES}
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                report['failed'].append(dict(geometry, stage='roots', error=str(exc)))
                continue
            if gpu_labels and any(not record['converged'] for record in result['gpu_labels'].values()):
                report['failed'].append(dict(geometry, stage='gpu_label', **result))
            else:
                report['flagged' if result['flags'] else 'passed'].append(dict(geometry, **result))
    return report


def time_gpu_label(atoms, basis, qm, nroots):
    from alframework.qm_interfaces.pyscf.gpu4pyscf import label_gpu4pyscf_excited_state_molecule
    from alframework.tools.molecules import MoleculesObject

    if __package__:
        from .prepare import state_properties
    else:
        from prepare import state_properties
    stopped = Event()
    samples = []
    executable = shutil.which('nvidia-smi')

    def sample_memory():
        while not stopped.is_set():
            try:
                result = subprocess.run([executable, '--query-compute-apps=pid,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True,
                                        text=True, check=True, timeout=3)
            except (OSError, subprocess.SubprocessError):
                return
            for row in result.stdout.splitlines():
                parts = [item.strip() for item in row.split(',')]
                if len(parts) == 2 and parts[0] == str(os.getpid()) and parts[1].isdigit():
                    samples.append(int(parts[1]))
            stopped.wait(.25)

    start = perf_counter()
    frame = MoleculesObject(atoms.copy(), 'qualification')
    worker = Thread(target=sample_memory, daemon=True) if executable else None
    if worker is not None:
        worker.start()
    try:
        label_gpu4pyscf_excited_state_molecule(frame, QM_config={**qm, 'basis': basis, 'nroots': nroots},
                                             properties_list=state_properties(nroots, optical=True))
        elapsed = perf_counter() - start
    finally:
        stopped.set()
        if worker is not None:
            worker.join()
    meta = frame.get_metadata()
    return {'converged': bool(frame.check_convergence()), 'seconds': elapsed,
            'observed_gpu_peak_MiB': max(samples, default=None),
            **{key: value for key, value in meta.items()
               if key.startswith('qm_') and ('seconds' in key or 'peak' in key or 'error' in key)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--all', action='store_true', dest='all_molecules',
                        help='Qualify every training molecule instead of the marked representatives.')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--nroots', type=int, choices=range(1, 6), default=NROOTS)
    parser.add_argument('--gradients', action='store_true', help='Compare CPU gradients for the matched roots.')
    parser.add_argument('--gpu-labels', action='store_true', help='Time complete GPU4PySCF labels in both bases.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.output.exists():
        raise FileExistsError(args.output)
    report = qualify(args.run, all_molecules=args.all_molecules, threads=args.threads,
                     nroots=args.nroots, gradients=args.gradients, gpu_labels=args.gpu_labels)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(', '.join(f'{len(report[key])} {key}' for key in ('passed', 'flagged', 'failed')))
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
