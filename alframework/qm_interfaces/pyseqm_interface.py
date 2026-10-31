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
