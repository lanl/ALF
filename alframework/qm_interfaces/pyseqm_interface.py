"""Ground-state PySEQM labeling with native multi-structure batches.

PySEQM evaluates a batch of independent molecules in one call. These tasks
accept ``molecule_objects`` and are selected with ``"mode": "native"`` in the
QM configuration's ``batching`` block (see :mod:`alframework.tools.labeling`).

PySEQM requires each molecule's atoms sorted by non-increasing atomic number,
with zero-padding to a common length. Sorting, padding, and the restoration of
ALF's atom order all happen here; ALF never sees PySEQM's layout.

Units: PySEQM returns ``Etot`` in eV and ``force`` (``-dE/dR``) in eV/Angstrom.
They are stored as ``energy`` and ``forces``; use ``properties_list`` unit
factors as with any other labeler. A molecule whose SCF does not converge is
marked unconverged and is skipped by ALF storage; its batch-mates are kept.

Example ``QM_config``::

    {"method": "AM1", "scf_eps": 1e-8, "device": "cuda",
     "batching": {"mode": "native", "batch_size": 64}}

For GPU labeling, place ``alf_QM_executor`` on GPU nodes and set ``device``.

Periodic systems and open-shell/charged states beyond ``charge``/``mult``
metadata are not supported by this adapter.
"""

import numpy as np
from parsl import python_app

from alframework.tools.molecules_class import MoleculesObject

DEFAULT_PARAMETERS = {"method": "AM1", "scf_eps": 1.0e-8, "scf_converger": [2]}


def pyseqm_parameters(QM_config):
    """Build PySEQM ``seqm_parameters`` from ALF's QM configuration."""
    parameters = dict(DEFAULT_PARAMETERS)
    for key in ("method", "scf_eps", "scf_converger"):
        if key in QM_config:
            parameters[key] = QM_config[key]
    parameters.update(QM_config.get("pyseqm_parameters", {}))
    if parameters.get("excited_states"):
        raise ValueError("pyseqm_interface labels ground states only; remove 'excited_states'.")
    return parameters


def pack_molecules(molecule_objects):
    """Sort each molecule heaviest-first and zero-pad the batch.

    Returns:
        (tuple): ``coordinates [batch, atoms, 3]``, ``species [batch, atoms]``,
        and ``orders``, the per-molecule permutations such that
        ``sorted = original[order]``.
    """
    width = max(len(molecule.get_atoms()) for molecule in molecule_objects)
    coordinates = np.zeros((len(molecule_objects), width, 3))
    species = np.zeros((len(molecule_objects), width), dtype=np.int64)
    orders = []
    for index, molecule in enumerate(molecule_objects):
        atoms = molecule.get_atoms()
        if any(atoms.get_pbc()):
            raise ValueError("pyseqm_interface does not support periodic systems ({}).".format(
                molecule.get_moleculeid()))
        numbers = atoms.get_atomic_numbers()
        order = np.argsort(-numbers, kind="stable")
        coordinates[index, :len(atoms)] = atoms.get_positions()[order]
        species[index, :len(atoms)] = numbers[order]
        orders.append(order)
    return coordinates, species, orders


def restore_atom_order(sorted_values, order):
    """Return per-atom values in the molecule's original order."""
    restored = np.empty((len(order),) + sorted_values.shape[1:], dtype=sorted_values.dtype)
    restored[order] = sorted_values[:len(order)]
    return restored


def run_pyseqm(coordinates, species, parameters, device="cpu", charges=0, mult=1):
    """Run one PySEQM batch; return energies (eV), forces (eV/A), and convergence flags."""
    import torch
    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        device = torch.device(device)
        # Molecule records the batch's elements in the parameters that Electronic_Structure reads.
        parameters = dict(parameters)
        constants = Constants().to(device)
        molecule = Molecule(constants, parameters, torch.as_tensor(coordinates, device=device),
                            torch.as_tensor(species, device=device),
                            charges=torch.as_tensor(charges, dtype=torch.float64, device=device),
                            mult=torch.as_tensor(mult, dtype=torch.float64, device=device)).to(device)
        driver = Electronic_Structure(parameters).to(device)
        driver(molecule)
        notconverged = getattr(driver, "notconverged", None)
        if notconverged is None:
            converged = np.zeros(len(coordinates), dtype=bool)
        else:
            converged = ~np.asarray(notconverged.detach().cpu().numpy(), dtype=bool).reshape(len(coordinates))
        return (molecule.Etot.detach().cpu().numpy().reshape(len(coordinates)),
                molecule.force.detach().cpu().numpy(), converged)
    finally:
        torch.set_default_dtype(previous_dtype)


def label_pyseqm_batch(molecule_objects, QM_config, runner=run_pyseqm):
    """Label a batch of molecules with ground-state PySEQM energies and forces."""
    assert all(isinstance(molecule, MoleculesObject) for molecule in molecule_objects), \
        'molecule_objects must contain only MoleculesObject instances'
    molecule_objects = list(molecule_objects)
    coordinates, species, orders = pack_molecules(molecule_objects)
    charges = np.array([molecule.get_metadata().get("charge", 0) for molecule in molecule_objects], dtype=float)
    mult = np.array([molecule.get_metadata().get("mult", 1) for molecule in molecule_objects], dtype=float)
    try:
        energies, forces, converged = runner(coordinates, species, pyseqm_parameters(QM_config),
                                             device=QM_config.get("device", "cpu"), charges=charges, mult=mult)
    except Exception as exc:
        if len(molecule_objects) == 1:
            raise
        # One bad member can fail the whole batched solve; isolate it by labeling members individually.
        return [labeled for molecule in molecule_objects
                for labeled in _label_alone(molecule, QM_config, runner, exc)]

    for index, molecule in enumerate(molecule_objects):
        member_forces = restore_atom_order(forces[index], orders[index])
        ok = bool(converged[index] and np.isfinite(energies[index]) and np.all(np.isfinite(member_forces)))
        molecule.store_results({"energy": float(energies[index]), "forces": member_forces})
        molecule.set_converged_flag(ok)
        if not ok:
            molecule.update_metadata({"labeling_error": "PySEQM SCF did not converge"
                                      if not converged[index] else "PySEQM returned non-finite results"})
    return molecule_objects


def _label_alone(molecule, QM_config, runner, batch_exception):
    try:
        return label_pyseqm_batch([molecule], QM_config, runner)
    except Exception as exc:
        molecule.set_converged_flag(False)
        molecule.update_metadata({"labeling_error": "{}: {} (batch error: {})".format(
            type(exc).__name__, exc, batch_exception)})
        return [molecule]


@python_app(executors=['alf_QM_executor'])
def pyseqm_task(molecule_objects, QM_config):
    """Native batched PySEQM labeling on ALF's QM executor.

    Args:
        molecule_objects (list): MoleculesObject instances to label together.
        QM_config (dict): PySEQM settings (``method``, ``scf_eps``, ``device``,
            optional ``pyseqm_parameters``) and the ``batching`` block.

    Returns:
        (list): The labeled MoleculesObject instances, in input order.
    """
    return label_pyseqm_batch(molecule_objects, QM_config)


# Excited-state entrypoints are separate so ground-state configurations and
# their native batching retain the original contract.
def run_pyseqm_cis(coordinates, species, parameters, device):
    """One native AM1/CIS solve on current upstream PySEQM (float64).

    Returns absolute state energies [molecules, states], negative gradients
    [molecules, states, padded_atoms, 3], and explicit convergence flags.
    """
    import torch
    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.set_num_threads(int(parameters.pop('_num_threads', 1)))
        dev = torch.device(device)
        mol = Molecule(Constants().to(dev), parameters,
                       torch.as_tensor(coordinates, dtype=torch.float64, device=dev),
                       torch.as_tensor(species, dtype=torch.int64, device=dev)).to(dev)
        driver = Electronic_Structure(parameters).to(dev)
        driver(mol)
        width = len(coordinates)
        def flag(obj, name):
            value = getattr(obj, name, None)
            if value is None:
                raise RuntimeError(f'PySEQM omitted required convergence flag {name}')
            arr = np.asarray(value.detach().cpu().numpy(), dtype=bool)
            if arr.shape != (width,):
                raise RuntimeError(f'Invalid PySEQM {name} shape {arr.shape}')
            return arr
        ok = ~flag(driver, 'notconverged') & flag(mol, 'cis_converged')
        ok &= ~flag(mol, 'cis_unstable')
        energies = torch.cat((mol.Etot[:, None], mol.Etot[:, None] + mol.cis_energies), dim=1)
        return (energies.detach().cpu().numpy(), mol.all_forces.detach().cpu().numpy(), ok)
    finally:
        torch.set_default_dtype(previous)


def _cis_child(connection, coordinates, species, parameters, device, runner):
    try:
        connection.send(('ok', runner(coordinates, species, parameters, device)))
    except BaseException as exc:
        connection.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def _stop_cis_child(process):
    """Always reap the owned child, escalating only when necessary."""
    process.join(timeout=1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    if process.is_alive():
        raise RuntimeError('PySEQM child did not terminate')


def run_pyseqm_cis_isolated(coordinates, species, parameters, device, timeout=300,
                            runner=run_pyseqm_cis):
    """Spawn one child for the entire native solve, with bounded lifetime."""
    import multiprocessing as mp
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_cis_child,
        args=(sender, coordinates, species, parameters, device, runner))
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise TimeoutError(f'PySEQM CIS native solve exceeded {timeout:g} seconds')
        try:
            status, result = receiver.recv()
        except EOFError as exc:
            raise RuntimeError('PySEQM CIS child exited without results') from exc
        if status != 'ok':
            raise RuntimeError(result)
        return result
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            _stop_cis_child(process)
            process.close()


def run_pyseqm_ground_as_states(coordinates, species, parameters, device):
    """A ground-state solve in the CIS runner's shapes: energies [molecules, 1], forces [molecules, 1, atoms, 3]."""
    energies, forces, converged = run_pyseqm(coordinates, species, parameters, device=device)
    return np.asarray(energies)[:, None], np.asarray(forces)[:, None], np.asarray(converged)


def label_pyseqm_excited_batch(molecule_objects, QM_config, properties_list,
                               sampler_config=None, gpus_per_node=None, runner=None):
    """Native CIS labeling with per-member fallback, stable IDs and atom order.

    Current support: neutral, restricted AM1/CIS molecules without PBC. Every
    requested state must have a force and converge for the frame to be stored.
    ``qm_backend_width`` is nonzero only for an actual successful native solve.
    With only ``E0``/``F0`` requested, it runs a ground-state AM1 solve instead of CIS.
    """
    import time
    from alframework.tools.excited_state_tools import derive_state_property_table, derive_gap_property_table
    from alframework.tools.gpu import sampling_device

    molecules = list(molecule_objects)
    if not molecules:
        return []
    if not all(isinstance(m, MoleculesObject) for m in molecules):
        raise TypeError('molecule_objects must contain MoleculesObject instances')
    start = time.monotonic()
    config = dict(QM_config or {})
    widths = []
    batch_errors = []
    compatibility_groups = []
    retries = 0
    ground = False
    offset = config.get('energy_offset_eV', (sampler_config or {}).get('energy_offset_eV', 0.))
    try:
        table = derive_state_property_table(properties_list, require_forces=True)
        gaps = derive_gap_property_table(properties_list)
        ground = len(table) == 1  # Only E0/F0: a ground-state solve, with no CIS.
        supported = {row[k] for row in table for k in ('energy_key', 'force_key')}
        supported.update(g['gap_key'] for g in gaps)
        if set(properties_list) - supported:
            raise ValueError('PySEQM CIS supports only E#/F#/dE# properties')
        for row in table:
            for key, scope in ((row['energy_key'], 'system'), (row['force_key'], 'atomic')):
                schema = properties_list[key]
                if len(schema) != 3 or schema[1] != scope or not np.isfinite(float(schema[2])):
                    raise ValueError(f'Invalid CIS property schema for {key}')
        scales = [float(properties_list[r['energy_key']][2]) for r in table]
        if len(set(scales)) != 1:
            raise ValueError('State energies must use identical property scales')
        if config.get('method', 'AM1').upper() != 'AM1':
            raise ValueError('Excited-state PySEQM currently supports AM1/CIS only')
        if any(m.get_metadata().get('charge', 0) != 0 or
               m.get_metadata().get('mult', 1) != 1 for m in molecules):
            raise ValueError('PySEQM CIS requires neutral closed-shell molecules')
        if config.get('charge', 0) != 0 or config.get('multiplicity', 1) != 1:
            raise ValueError('PySEQM CIS requires neutral closed-shell molecules')
        offset = float(offset)
        if not np.isfinite(offset):
            raise ValueError('energy_offset_eV must be finite')
        tolerance = float(config.get('cis_tol', 1e-8))
        scf_eps = float(config.get('scf_eps', 1e-10))
        timeout = float(config.get('timeout_seconds', 300))
        if not all(np.isfinite(x) and x > 0 for x in (tolerance, scf_eps, timeout)):
            raise ValueError('CIS/SCF tolerances and timeout must be finite and positive')
        if int(config.get('num_threads', 1)) < 1:
            raise ValueError('num_threads must be positive')
        if ground:
            parameters = dict(DEFAULT_PARAMETERS, scf_eps=scf_eps)
        else:
            parameters = {'method': 'AM1', 'scf_eps': min(scf_eps, .1*tolerance),
                          'scf_converger': [2], 'analytical_gradient': [True],
                          'do_all_forces': True,
                          'excited_states': {'method': 'cis', 'n_states': len(table)-1,
                                             'tolerance': tolerance},
                          '_num_threads': int(config.get('num_threads', 1))}
        # Injected runners permit dependency-free tests. Real GPU selection is
        # local to the inherited mask; this never edits visibility variables.
        device = sampling_device({'device': config.get('device', 'cpu')}, gpus_per_node)
        def evaluate(group):
            nonlocal retries
            try:
                xyz, numbers, orders = pack_molecules(group)
                if not np.isfinite(xyz).all():
                    raise ValueError('PySEQM coordinates must be finite')
                widths.append(len(group))
                if runner is None and ground:
                    energy, force, ok = run_pyseqm_ground_as_states(xyz, numbers, dict(parameters), device)
                elif runner is None:
                    energy, force, ok = run_pyseqm_cis_isolated(
                        xyz, numbers, dict(parameters), device, timeout)
                else:
                    energy, force, ok = runner(xyz, numbers, dict(parameters), device)
                energy, force, ok = np.asarray(energy), np.asarray(force), np.asarray(ok)
                if (energy.shape != (len(group), len(table)) or
                    force.shape != (len(group), len(table), numbers.shape[1], 3) or
                    ok.shape != (len(group),)):
                    raise ValueError('Malformed PySEQM CIS batch output')
                for i, m in enumerate(group):
                    member_forces = np.stack([restore_atom_order(f, orders[i]) for f in force[i]])
                    valid = bool(ok[i] and np.isfinite(energy[i]).all() and np.isfinite(member_forces).all())
                    if not valid:
                        fail(m, 'SCF/CIS failed or returned nonfinite results')
                        continue
                    results = {r['energy_key']: float(energy[i,j]-offset) for j,r in enumerate(table)}
                    results.update({r['force_key']: member_forces[j] for j,r in enumerate(table)})
                    results.update({g['gap_key']: results[g['upper_energy_key']]-results[g['lower_energy_key']] for g in gaps})
                    m.store_results(results)
                    m.set_converged_flag(True)
                    m.update_metadata({'qm_backend_width': len(group), 'qm_scf_converged': True,
                                       'qm_cis_converged': None if ground else True, 'qm_device': device,
                                       'energy_offset_eV': offset})
            except Exception as exc:
                batch_errors.append({'width': len(group), 'error': f'{type(exc).__name__}: {exc}'})
                if len(group) == 1:
                    fail(group[0], f'{type(exc).__name__}: {exc}')
                else:
                    retries += len(group)
                    for m in group:
                        evaluate([m])
        def fail(m, message):
            for key in properties_list:
                m.get_results().pop(key, None)
            m.set_converged_flag(False)
            m.update_metadata({'qm_error': message, 'qm_backend_width': 0,
                               'qm_scf_converged': False, 'qm_cis_converged': False})
        # Upstream all-state analytic CIS gradients require identical sorted
        # species in a solve, even though its ground-state API accepts mixed
        # sizes. Partition before the backend call; this is not an error retry.
        groups = {}
        for m in molecules:
            key = tuple(sorted(map(int, m.get_atoms().numbers)))
            groups.setdefault(key, []).append(m)
        compatibility_groups = [[m.get_moleculeid() for m in group] for group in groups.values()]
        for group in groups.values():
            evaluate(group)
    except Exception as exc:
        for m in molecules:
            for key in properties_list:
                m.get_results().pop(key, None)
            m.set_converged_flag(False)
            m.update_metadata({'qm_error': f'{type(exc).__name__}: {exc}', 'qm_backend_width': 0})
    for m in molecules:
        m.update_metadata({'qm_backend': 'pyseqm' if ground else 'pyseqm_cis', 'qm_submitted_width': len(molecules),
                           'qm_solve_widths': list(widths), 'qm_retry_count': retries,
                           'qm_batch_errors': batch_errors, 'qm_compatibility_groups': compatibility_groups,
                           'qm_failed_ids': [x.get_moleculeid() for x in molecules if not x.check_convergence()],
                           'qm_elapsed_seconds': time.monotonic()-start})
    return molecules


@python_app(executors=['alf_QM_executor'])
def pyseqm_excited_state_task(molecule_object, QM_config, properties_list,
                              sampler_config=None, gpus_per_node=None):
    return label_pyseqm_excited_batch([molecule_object], QM_config, properties_list,
                                     sampler_config, gpus_per_node)[0]


@python_app(executors=['alf_QM_executor'])
def pyseqm_excited_state_batch_task(molecule_objects, QM_config, properties_list,
                                    sampler_config=None, gpus_per_node=None):
    return label_pyseqm_excited_batch(molecule_objects, QM_config, properties_list,
                                     sampler_config, gpus_per_node)
