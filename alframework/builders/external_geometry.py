"""Import unlabeled geometries without inheriting a source calculation."""

import hashlib
import re
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from parsl import python_app

from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, validated_atom_ids
from alframework.tools.molecules_class import MoleculesObject


def _text(value):
    return value.decode('utf8') if isinstance(value, bytes) else str(value)


def geometry_manifest(builder_config, master_directory=None):
    """Index files in configured order, HDF5 groups in sorted order."""
    sources = builder_config.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('builder_config.sources must be a nonempty list of files')
    root = Path(master_directory or '.').expanduser().resolve()
    entries = []
    for source in sources:
        path = Path(source).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix.lower() in {'.xyz', '.extxyz'}:
            frames = read(str(path), index=':', parallel=False)
            entries.extend((str(path), None, i, fingerprint) for i in range(len(frames)))
        elif path.suffix.lower() in {'.h5', '.hdf5'}:
            with h5py.File(path, 'r') as handle:
                groups = []
                if 'coordinates' in handle and 'species' in handle:
                    groups.append('/')
                handle.visititems(lambda name, obj: groups.append(name)
                                  if isinstance(obj, h5py.Group) and 'coordinates' in obj
                                  and 'species' in obj else None)
                for name in sorted(groups):
                    coordinates = handle[name]['coordinates']
                    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
                        raise ValueError(f'{path}:{name}/coordinates must be [frames, atoms, 3]')
                    entries.extend((str(path), name, i, fingerprint)
                                   for i in range(len(coordinates)))
        else:
            raise ValueError(f'Unsupported geometry file: {path}')
    if not entries:
        raise ValueError('No geometry frames in configured sources')
    return entries


def _h5_frame(group, index):
    positions = np.asarray(group['coordinates'][index], dtype=float)
    species = np.asarray(group['species'])
    if species.ndim == 2:
        species = species[0 if species.shape[0] == 1 else index]
    if species.ndim != 1 or len(species) != len(positions):
        raise ValueError('Geometry species must describe each atom')
    if species.dtype.kind in 'SUO':
        numbers = [atomic_numbers[_text(v)] for v in species]
    elif species.dtype.kind in 'iu':
        numbers = species
    else:
        raise ValueError('Species must be chemical symbols or integer atomic numbers')
    options = {}
    for key, static_shape in [('cell', (3, 3)), ('pbc', (3,))]:
        if key in group:
            values = np.asarray(group[key])
            options[key] = values if values.shape == static_shape else values[index]
    atoms = Atoms(numbers=numbers, positions=positions, **options)
    if TOPOLOGY_ATOM_IDS in group:
        ids = np.asarray(group[TOPOLOGY_ATOM_IDS])
        ids = ids if ids.ndim == 1 else ids[index]
        atoms.set_array(TOPOLOGY_ATOM_IDS, validated_atom_ids(ids, len(atoms)))
    source_id = _text(group['_id'][index]) if '_id' in group else None
    return atoms, source_id


def build_external_geometries(moleculeids, builder_config, master_directory=None):
    """Stateless ID-to-frame mapping; returned objects contain no labels.

    ``sequential`` uses the trailing integer of an ID and raises on exhaustion.
    ``deterministic_hash`` samples with replacement using the ID and seed.
    Positions are in Angstrom, as in ASE and ALF HDF5 storage.
    """
    ids = list(moleculeids)
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError('Nonempty string molecule IDs are required')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate molecule IDs in one builder task')
    manifest = geometry_manifest(builder_config, master_directory)
    mode = builder_config.get('selection_mode', 'deterministic_hash')
    if mode not in {'sequential', 'deterministic_hash'}:
        raise ValueError('selection_mode must be sequential or deterministic_hash')
    outputs = []
    for moleculeid in ids:
        if mode == 'sequential':
            match = re.search(r'(\d+)$', moleculeid)
            if match is None:
                raise ValueError('Sequential selection requires a trailing integer in each ID')
            selected = int(match.group(1))
            if selected >= len(manifest):
                raise IndexError(f'Geometry source exhausted at {selected}; {len(manifest)} frames available')
        else:
            key = f'{moleculeid}|{int(builder_config.get("selection_seed", 0))}'.encode()
            selected = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(manifest)
        path, group, frame, fingerprint = manifest[selected]
        if group is None:
            source = read(path, index=frame, parallel=False)
            source_id = source.info.get('_id', source.info.get('id'))
        else:
            with h5py.File(path, 'r') as handle:
                source, source_id = _h5_frame(handle[group], frame)
        if (not len(source) or not np.isfinite(source.positions).all()
                or np.any(source.numbers <= 0)):
            raise ValueError(f'Invalid geometry in {path}, frame {frame}')
        # Copy only geometry and explicitly supplied identity, never calculators,
        # source info/labels, velocities, or other per-atom calculation outputs.
        atoms = Atoms(numbers=source.numbers, positions=source.positions,
                      cell=source.cell, pbc=source.pbc)
        if TOPOLOGY_ATOM_IDS in source.arrays:
            atoms.set_array(TOPOLOGY_ATOM_IDS,
                            validated_atom_ids(source.arrays[TOPOLOGY_ATOM_IDS], len(source)))
        molecule = MoleculesObject(atoms, moleculeid)
        molecule.update_metadata(dict(geometry_source=path, geometry_source_sha256=fingerprint,
                                      geometry_group=group, geometry_frame=frame,
                                      geometry_source_id=source_id, geometry_index=selected))
        outputs.append(molecule)
    return outputs


@python_app(executors=['alf_sampler_executor'])
def external_geometry_builder_task(moleculeid=None, moleculeids=None,
                                   builder_config=None, master_directory=None):
    """Builder compatible with ALF's single and multi-structure dispatch."""
    # ALF supplies both arguments for multi-structure builders; plural wins.
    plural = moleculeids is not None
    result = build_external_geometries(moleculeids if plural else [moleculeid],
                                       builder_config or {}, master_directory)
    return result if plural else result[0]
