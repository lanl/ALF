"""Explicit topology identity; never infer same-element atom identities."""

import numpy as np

TOPOLOGY_ATOM_IDS = "topology_atom_ids"


def validated_atom_ids(values, n_atoms, *, context="topology_atom_ids"):
    ids = np.asarray(values)
    if (ids.shape != (n_atoms,) or ids.dtype.kind not in "iu"
            or not np.array_equal(np.sort(ids), np.arange(n_atoms))):
        raise ValueError(f"{context} must be an integer permutation of 0..{n_atoms - 1}.")
    return ids.astype(np.int64, copy=False)


def canonical_orders(values, n_frames, n_atoms, *, context="topology_atom_ids"):
    ids = np.asarray(values)
    if ids.shape == (n_atoms,):
        ids = np.broadcast_to(ids, (n_frames, n_atoms))
    if ids.shape != (n_frames, n_atoms):
        raise ValueError(f"{context} must have shape [atoms] or [frames, atoms].")
    for row in ids:
        validated_atom_ids(row, n_atoms, context=context)
    return np.argsort(ids, axis=1, kind="stable")


def canonicalize_tagged_atoms(atoms):
    """Return a canonical copy when a real identity mapping is present."""
    if TOPOLOGY_ATOM_IDS not in atoms.arrays:
        return atoms
    ids = validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
    return atoms[np.argsort(ids, kind="stable")]


def attach_topology_ids(atoms, reference_numbers):
    """Tag an already reference-ordered structure in a topology-aware extension."""
    if not np.array_equal(atoms.numbers, reference_numbers):
        raise ValueError("Cannot attach topology IDs outside reference atomic-number order.")
    if TOPOLOGY_ATOM_IDS in atoms.arrays:
        ids = validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
        if not np.array_equal(ids, np.arange(len(atoms))):
            raise ValueError("Normalize supplied topology IDs before attaching canonical IDs.")
    else:
        atoms.set_array(TOPOLOGY_ATOM_IDS, np.arange(len(atoms), dtype=np.int64))
