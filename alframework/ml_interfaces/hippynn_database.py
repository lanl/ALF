"""ALF schema support for small molecules in HIPPYNN's ANI database reader.

Imported only inside training, after its worker's device setup. The upstream
reader infers atom axes from dimensions, which are ambiguous for small systems
(three atoms versus three Cartesian components, for example).
"""
import numpy as np
from hippynn.databases.h5_pyanitools import PyAniDirectoryDB


class ALFPyAniDirectoryDB(PyAniDirectoryDB):
    def __init__(self, *args, coordinates_key='coordinates', atom_keys=(), **kwargs):
        self.coordinates_key = coordinates_key
        self.atom_keys = tuple(dict.fromkeys(('coordinates', 'forces', 'charges', coordinates_key,
                                             *(key for key in atom_keys if key is not None))))
        super().__init__(*args, **kwargs)
        # HIPPYNN persists this restarter in checkpoints; retain custom schema.
        self.restarter.kwargs.update(coordinates_key=coordinates_key, atom_keys=self.atom_keys,
                                    allow_unfound=kwargs.get('allow_unfound', False))

    def determine_key_structure(self, batches, sys_count, n_atoms_max, species_key='species'):
        if not batches:
            raise ValueError('No structure groups in the ALF training dataset')
        if all(batch[species_key].shape[1] >= 7 for batch in batches):
            return super().determine_key_structure(batches, sys_count, n_atoms_max, species_key=species_key)

        keys = set(batches[0])
        atom_keys = set(self.atom_keys) | {species_key}
        shapes = {}
        padding = {}
        for key, value in batches[0].items():
            shapes[key] = [sys_count, *value.shape[1:]]
            padding[key] = [1] if key in atom_keys else []
            if padding[key]:
                if value.ndim < 2:
                    raise ValueError(f'Atom property {key!r} has no atom axis')
                shapes[key][1] = n_atoms_max
        for batch in batches:
            if set(batch) != keys:
                raise ValueError('Inconsistent property keys across ALF structure groups')
            coordinates = batch[self.coordinates_key]
            n_frames, n_atoms = coordinates.shape[:2]
            if coordinates.shape != (n_frames, n_atoms, 3):
                raise ValueError('ALF coordinates must have shape (frames, atoms, 3)')
            if batch[species_key].shape != (1, n_atoms):
                raise ValueError('ALF species must describe one fixed composition per group')
            for key, value in batch.items():
                expected = list(shapes[key][1:])
                if key in atom_keys:
                    expected[0] = n_atoms
                if list(value.shape[1:]) != expected:
                    raise ValueError(f'Unsupported trailing shape for {key!r}; configure its atom-property key')
                if key != species_key and value.shape[0] != n_frames:
                    raise ValueError(f'Frame count mismatch for {key!r}')
        padding['sys_number'] = []
        return padding, shapes, self.coordinates_key
