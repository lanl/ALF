"""Store labeled structures, and the sources of their labels, in ALF's HDF5 training shards."""

import os

import numpy as np
from ase.geometry import complete_cell

from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, validated_atom_ids
from alframework.tools.files import replaced_atomically
from alframework.tools.label_hdf5 import GROUP_BY, LabelH5Writer, label_schema_attributes, many_groups_warning, split_group_by
from alframework.tools.label_sources import LabelSourceCollector, write_shard_label_sources
from alframework.tools.label_validation import validate_label
from alframework.tools.molecules import MOLECULE_ID, MoleculesObject

PAIR_INDEX_METADATA_KEY = "nac_pairs"
PAIR_INDEX_DATASET = "nac_pairs"
PARTIAL_SUFFIX = '.partial'
# Molecule metadata keys and the shard datasets they're stored in.
STATE_METADATA = ('charge', 'mult')
STRUCTURE_STATE_DATASETS = ('total_charge', 'spin_multiplicity')


def compute_empirical_formula(S):
    """Computes an empirical formula to describe a chemical system.

    Args:
        S (list): List of strings where each strings represents an atom in the system.

    Returns:
        (str): An empircal formula representing the system.

    """
    uniques = np.unique(S, return_counts=True)
    arg_sort = np.argsort(uniques[0])

    return "_".join([i+str(j).zfill(2) for i,j in zip(uniques[0][arg_sort], uniques[1][arg_sort])])


def group_key(atoms, group_by='formula'):
    """The shard group a structure is stored in.

    ``"formula"`` groups by sorted species list (:func:`compute_empirical_formula`). ``"atom_count"`` groups by atom
    count, and keeps periodic, partly periodic and nonperiodic frames, and frames with and without topology atom
    IDs, in groups of their own, as a group's columns must agree: ``atoms_0200``, ``atoms_0200_periodic``,
    ``atoms_0012_pbc110``, ``atoms_0042_tagged``.
    """
    if group_by == 'formula':
        return compute_empirical_formula(atoms.get_chemical_symbols())
    if group_by != 'atom_count':
        raise ValueError(f'Unknown label grouping {group_by!r}.')
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    name = f'atoms_{len(atoms):04d}'
    if pbc.all():
        name += '_periodic'
    elif pbc.any():
        name += '_pbc' + ''.join('1' if value else '0' for value in pbc)
    if TOPOLOGY_ATOM_IDS in atoms.arrays:
        name += '_tagged'
    return name


def pair_atomic_property(values, atom_index, property_key):
    """Reorder one ``[pairs, atoms, 3]`` property into shard atom order.

    ``pair_atomic`` exists because the atom permutation applies to axis 1. The
    ``atomic`` branch of :func:`store_current_data` indexes axis 0, which would
    silently permute the pair axis instead.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != len(atom_index) or array.shape[2] != 3:
        raise RuntimeError(
            f"pair_atomic property {property_key!r} must have shape "
            f"[pairs, {len(atom_index)}, 3]; received {array.shape}."
        )
    return array[:, atom_index]


def store_pair_index(group, system, properties):
    """Record the pair index for ``pair_atomic`` properties once per group.

    The mapping is constant for a given state count, so it is stored like
    ``species`` rather than appended per frame. A shard therefore describes its
    own pair ordering and analysis code never has to recompute it.
    """

    if not any(
        str(schema[1]).lower() == "pair_atomic"
        for schema in properties.values()
        if len(schema) >= 2
    ):
        return
    pairs = system.get_metadata().get(PAIR_INDEX_METADATA_KEY)
    if pairs is None:
        raise RuntimeError(
            "A pair_atomic property requires the QM interface to record "
            f"{PAIR_INDEX_METADATA_KEY!r} metadata describing its pair order."
        )
    raw_pairs = np.asarray(pairs)
    if raw_pairs.dtype.kind not in "iu" or np.any(raw_pairs < 0):
        raise ValueError("nac_pairs must contain nonnegative integer state indices.")
    pair_array = raw_pairs.astype(np.int64, copy=False)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise RuntimeError(
            f"{PAIR_INDEX_METADATA_KEY!r} must have shape [pairs, 2]; "
            f"received {pair_array.shape}."
        )
    if (np.any(pair_array[:, 0] >= pair_array[:, 1])
            or len({tuple(row) for row in pair_array}) != len(pair_array)):
        raise ValueError("nac_pairs must contain unique ordered state pairs.")
    for key, schema in properties.items():
        if str(schema[1]).lower() == "pair_atomic":
            if len(system.get_results()[key]) != len(pair_array):
                raise ValueError("nac_pairs length differs from pair_atomic property.")
    stored = group.get(PAIR_INDEX_DATASET)
    if stored is None:
        group[PAIR_INDEX_DATASET] = pair_array
    elif not np.array_equal(stored, pair_array):
        # Mixing pair orderings inside one group would make every stored
        # pair_atomic slice ambiguous.
        raise RuntimeError(
            "Molecules in one shard group reported different "
            f"{PAIR_INDEX_METADATA_KEY!r} orderings."
        )


def store_current_data(h5path, system_data, properties, attributes=None, *, h5_options=None, label_units=None):
    """Stores the key results of the QM calculations in the database.

    Args:
        h5path (str): Path to store the .h5 files.
        system_data (iterable): MoleculesObjects; read once, so a generator works (one piece at a time).
        properties (dict): Dictionary defined in master.json whose keys are the properties that we want to retrieve
                           from the QM calculations and store in the database.
        attributes (dict): Optional strings to record as attributes on the shard.
        h5_options (dict): Optional HDF5 file creation settings, and the optional ``group_by``
                           (``"formula"``, the default, or ``"atom_count"``; see :func:`group_key`).
        label_units (dict): Explicit units of the stored property datasets.

    When any converged structure has ``charge`` or ``mult`` metadata, every structure in the shard
    gets ``total_charge`` and ``spin_multiplicity`` datasets, which default to 0 and 1.
    When supplied, ``molecule_id`` records the molecular identity independently of the geometry's ``_id``.

    Returns:
        (int): The number of converged structures saved.

    The shard is written to ``<h5path>.partial`` and renamed when complete, so a crash never leaves
    a partial shard under the final name. Readers ignore ``.partial`` files, because they don't end
    in ``.h5``.

    """
    if os.path.exists(h5path):
        # A restart moves past shards stored before a crash, so this means something else wrote here.
        raise FileExistsError(f'{h5path} already exists, and ALF never replaces a shard. Check whether another '
                              'process is writing to this directory.')
    group_by, writer_options = split_group_by(h5_options)
    # With atom-count groups, frames of one group can differ in species, so species is stored for each frame.
    per_frame_species = group_by == 'atom_count'
    # system data is a list of [mol-id(string), atoms, properties dictionary]
    data_dict = {}
    # Total charge and spin multiplicity, stored for the whole shard when any structure records them.
    states, has_state = {}, False
    identities, has_identity = {}, False
    print("Saving h5 file: " + h5path)
    total_number = 0
    sources = LabelSourceCollector()
    saved_number = 0
    nan_number = 0
    unconverged_number = 0
    for system in system_data:
        total_number += 1
        assert isinstance(system, MoleculesObject), 'system must be an instance of MoleculesObject'

        cur_moliculeid = system.get_moleculeid()
        cur_atoms = system.get_atoms()
        cur_properties = system.get_results()
        molkey = group_key(cur_atoms, group_by)

        # Ensure system converged before saving
        if system.check_convergence():
            # Labels are checked when they're held; this catches callers that store directly.
            validate_label(system, properties)
            if molkey in data_dict and (('cell' in data_dict[molkey]) != bool(cur_atoms.pbc.all())):
                raise ValueError(f'{molkey}: cannot mix periodic and nonperiodic frames in one shard group.')
            saved_number += 1
            sources.add(system)
            # Main's species contract: every row in a formula group is sorted.
            atom_index = np.argsort(cur_atoms.get_atomic_numbers(),
                                    kind="stable" if TOPOLOGY_ATOM_IDS in cur_atoms.arrays else None)
            topology_ids = None
            if TOPOLOGY_ATOM_IDS in cur_atoms.arrays:
                topology_ids = validated_atom_ids(
                    cur_atoms.arrays[TOPOLOGY_ATOM_IDS], len(cur_atoms),
                    context=f"{cur_moliculeid}:topology_atom_ids")[atom_index]
            if molkey in data_dict and ((topology_ids is not None) !=
                                         (TOPOLOGY_ATOM_IDS in data_dict[molkey])):
                raise ValueError("Cannot mix tagged and untagged topology layouts in one shard group.")
            state = system.get_metadata()
            identity = state.get(MOLECULE_ID, '')
            if not isinstance(identity, str) or (MOLECULE_ID in state and not identity.strip()):
                raise ValueError(f'{cur_moliculeid}: molecule_id must be a nonempty string.')
            has_identity = has_identity or bool(identity)
            identities.setdefault(molkey, []).append(identity)
            has_state = has_state or STATE_METADATA[0] in state or STATE_METADATA[1] in state
            row_state = (int(state.get(STATE_METADATA[0], 0)), int(state.get(STATE_METADATA[1], 1)))
            # If there is already a molecule with the same formula, append
            if molkey in data_dict:
                states[molkey].append(row_state)
                data_dict[molkey]["_id"].append(cur_moliculeid)
                if per_frame_species:
                    data_dict[molkey]["species"].append(np.array(cur_atoms.get_chemical_symbols())[atom_index])
                data_dict[molkey]["coordinates"].append(cur_atoms.get_positions()[atom_index])
                if topology_ids is not None:
                    data_dict[molkey][TOPOLOGY_ATOM_IDS].append(topology_ids)
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"].append(complete_cell(cur_atoms.get_cell()))
                for prop in properties:
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]].append(cur_properties[prop] * properties[prop][2])
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]].append(np.array(cur_properties[prop])[atom_index] * properties[prop][2])
                    elif properties[prop][1].lower() == "pair_atomic":
                        # [pairs, atoms, 3]: the atom reordering applies to axis 1,
                        # so the atomic branch above would index the pair axis.
                        data_dict[molkey][properties[prop][0]].append(
                            pair_atomic_property(cur_properties[prop], atom_index, prop)
                            * properties[prop][2]
                        )
                    else:
                        raise RuntimeError('Unknown property format')
                store_pair_index(data_dict[molkey], system, properties)
            # If there is not already a molecule with this empirical formula, make a new one
            else:
                states[molkey] = [row_state]
                data_dict[molkey] = {}
                species = np.array(cur_atoms.get_chemical_symbols())[atom_index]
                data_dict[molkey]["species"] = [species] if per_frame_species else species
                data_dict[molkey]["_id"] = [cur_moliculeid]
                data_dict[molkey]["coordinates"] = [cur_atoms.get_positions()[atom_index]]
                if topology_ids is not None:
                    data_dict[molkey][TOPOLOGY_ATOM_IDS] = [topology_ids]
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"] = [complete_cell(cur_atoms.get_cell())]
                for prop in properties.keys():
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]] = [cur_properties[prop] * properties[prop][2]]
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]] = [np.array(cur_properties[prop])[atom_index] * properties[prop][2]]
                    elif properties[prop][1].lower() == "pair_atomic":
                        data_dict[molkey][properties[prop][0]] = [
                            pair_atomic_property(cur_properties[prop], atom_index, prop)
                            * properties[prop][2]
                        ]
                    else:
                        raise RuntimeError('Unknown property format')
                store_pair_index(data_dict[molkey], system, properties)
        elif not isinstance(system, MoleculesObject): # code never enter in this line, but leaving for now to avoid problems
            nan_number += 1
        elif not system.check_convergence():
            unconverged_number = unconverged_number + 1
        else:
            print("Warning: molecule not saved for unknown reason")
    print("Total Systems: " + str(total_number))
    print("Saved Systems: " + str(saved_number))
    print("NAN Systems: " + str(nan_number))
    print("Unconverged Systems: " + str(unconverged_number))

    if has_state:
        for isokey, rows in states.items():
            data_dict[isokey][STRUCTURE_STATE_DATASETS[0]] = [charge for charge, _ in rows]
            data_dict[isokey][STRUCTURE_STATE_DATASETS[1]] = [mult for _, mult in rows]
    if has_identity:
        for formula, rows in identities.items():
            data_dict[formula][MOLECULE_ID] = rows
    for isokey in data_dict:
        frames = len(data_dict[isokey]['_id'])
        for key, values in data_dict[isokey].items():
            if key not in ('species', PAIR_INDEX_DATASET) and len(values) != frames:
                raise ValueError(f'{isokey}:{key} is not aligned with the {frames} stored frames.')
        # print('isokeys:',isokey)
        for propkey in data_dict[isokey]:
            if per_frame_species and propkey == 'species':
                data_dict[isokey][propkey] = np.char.encode(np.asarray(data_dict[isokey][propkey]), 'utf-8')
            elif propkey.lower() in ('species', '_id', MOLECULE_ID):
                data_dict[isokey][propkey] = [el.encode('utf-8') for el in list(data_dict[isokey][propkey])]
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
            else:
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
                # print("encoding species")
#                    if type(data_dict[isokey][propkey]) is 'numpy.ndarray':
#                        data_dict[isokey][propkey] = np.stack(data_dict[isokey][propkey])
#                    else:
#                        data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
#                    print('propkey:', propkey,data_dict[isokey][propkey].shape)
#
    with replaced_atomically(h5path, suffix=PARTIAL_SUFFIX) as partial:
        # Mode 'w' overwrites a .partial left by a crash.
        dpack = LabelH5Writer(str(partial), mode='w', **writer_options)
        try:
            for key in data_dict:
                dpack.store_data(key, **data_dict[key])
            write_shard_label_sources(dpack.store, sources.sources)
            dpack.store.attrs.update({**(attributes or {}), **label_schema_attributes(properties, label_units)})
            if per_frame_species:
                dpack.store.attrs[GROUP_BY] = group_by
        finally:
            dpack.cleanup()
    many_groups_warning(h5path, len(data_dict), saved_number, group_by)
    return saved_number
