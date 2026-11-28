"""Neutral S0/S1 permanent dipoles from SCF and finite-field TDA."""

import numpy as np
from ase import units

if __package__:
    from .qualify import compare_roots, root_data, solve_tda
else:
    from qualify import compare_roots, root_data, solve_tda


def permanent_dipoles(atoms, settings, step=1.e-4, *, check_step=False):
    """Return state dipoles in e Å and the diagnostics for this geometry."""
    if step <= 0 or not np.isfinite(step):
        raise ValueError('The electric field step must be finite and positive.')
    if settings['charge'] or settings['multiplicity'] != 1 or settings.get('solvent'):
        raise ValueError('This reference pilot requires a neutral, gas-phase singlet.')
    basis = settings['basis']
    # Extra roots let the matching follow S1 even if the field changes state order.
    nroots = max(3, settings['nroots'])
    settings = {**settings, 'scf_conv_tol': min(settings['scf_conv_tol'], 1.e-11),
                'tda_conv_tol': min(settings['tda_conv_tol'], 1.e-9)}
    molecule, scf, td = solve_tda(atoms, basis, settings, nroots)
    zero = root_data(molecule, scf, td)
    ground = np.asarray(scf.dip_moment(unit='AU', verbose=0))
    nuclear = molecule.atom_charges() @ molecule.atom_coords()

    def derivative(h):
        energies, overlaps = [], []
        for direction in np.eye(3):
            pair = []
            for field in (-h * direction, h * direction):
                perturbed = solve_tda(atoms, basis, settings, nroots, field=field)
                roots = root_data(*perturbed)
                match = compare_roots(zero, roots, min_overlap=.8, gap_shift_eV=.2)
                root = match['root_matches_1_based'][0] - 1
                overlap = match['overlap_matrix'][0][root]
                if overlap < .8:
                    raise ValueError(f'S1 overlap {overlap:.3f} under applied field is below 0.8.')
                overlaps.append(overlap)
                ground_energy = perturbed[1].e_tot - field @ nuclear
                pair.append([ground_energy, ground_energy + perturbed[2].e[root]])
            energies.append((np.asarray(pair[0]) - pair[1]) / (2 * h))
        return np.asarray(energies).T, min(overlaps)

    dipoles, overlap = derivative(step)
    checks = {'minimum_s1_overlap': overlap,
              'finite_field_s0_error_eA': float(np.linalg.norm(dipoles[0] - ground) * units.Bohr)}
    if checks['finite_field_s0_error_eA'] > 1.e-3:
        raise ValueError(f'Ground-state finite-field check failed: {checks["finite_field_s0_error_eA"]:.3g} e Å.')
    if check_step:
        small, _ = derivative(step / 2)
        checks['halved_field_s1_change_eA'] = float(np.linalg.norm(small[1] - dipoles[1]) * units.Bohr)
        if checks['halved_field_s1_change_eA'] > 1.e-3:
            raise ValueError(f'S1 finite-field step check failed: {checks["halved_field_s1_change_eA"]:.3g} e Å.')
    return np.stack((ground, dipoles[1])) * units.Bohr, checks
