"""Store labeled structures and provenance in ALF's HDF5 training shards."""

import glob
import json
import os
from pathlib import Path

import numpy as np
from ase.geometry import complete_cell

from alframework.tools import pyanitools as pyt
from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, validated_atom_ids
from alframework.tools.files import fsync_file
from alframework.tools.label_validation import validate_label
from alframework.tools.molecules_class import MoleculesObject

PAIR_INDEX_METADATA_KEY = "nac_pairs"
PAIR_INDEX_DATASET = "nac_pairs"
PARTIAL_SUFFIX = '.partial'
# Molecule metadata keys and the shard datasets they're stored in.
STATE_METADATA = ('charge', 'mult')
STRUCTURE_STATE_DATASETS = ('total_charge', 'spin_multiplicity')
LABEL_PROVENANCE = 'label_provenance'


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
            "Molecules in one empirical-formula group reported different "
            f"{PAIR_INDEX_METADATA_KEY!r} orderings."
        )


def store_current_data(h5path, system_data, properties, attributes=None):
    """Stores the key results of the QM calculations in the database.

    Args:
        h5path (str): Path to store the .h5 files.
        system_data (iterable): MoleculesObjects; read once, so a generator works (one piece at a time).
        properties (dict): Dictionary defined in master.json whose keys are the properties that we want to retrieve
                           from the QM calculations and store in the database.
        attributes (dict): Optional strings to record as attributes on the shard.

    When any converged structure has ``charge`` or ``mult`` metadata, every structure in the shard
    gets ``total_charge`` and ``spin_multiplicity`` datasets, which default to 0 and 1.

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
    # system data is a list of [mol-id(string), atoms, properties dictionary]
    data_dict = {}
    # Total charge and spin multiplicity, stored for the whole shard when any structure records them.
    states, has_state = {}, False
    print("Saving h5 file: " + h5path)
    total_number = 0
    sources = SourceCollector()
    saved_number = 0
    nan_number = 0
    unconverged_number = 0
    for system in system_data:
        total_number += 1
        assert isinstance(system, MoleculesObject), 'system must be an instance of MoleculesObject'

        cur_moliculeid = system.get_moleculeid()
        cur_atoms = system.get_atoms()
        cur_properties = system.get_results()
        molkey = compute_empirical_formula(cur_atoms.get_chemical_symbols())

        # Ensure system converged before saving
        if system.check_convergence():
            # Labels are checked when they're held; this catches callers that store directly.
            validate_label(system, properties)
            if molkey in data_dict and (('cell' in data_dict[molkey]) != bool(cur_atoms.pbc.all())):
                raise ValueError(f'{molkey}: cannot mix periodic and nonperiodic frames in one formula group.')
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
                raise ValueError("Cannot mix tagged and untagged topology layouts in one formula group.")
            state = system.get_metadata()
            has_state = has_state or STATE_METADATA[0] in state or STATE_METADATA[1] in state
            row_state = (int(state.get(STATE_METADATA[0], 0)), int(state.get(STATE_METADATA[1], 1)))
            # If there is already a molecule with the same formula, append
            if molkey in data_dict:
                states[molkey].append(row_state)
                data_dict[molkey]["_id"].append(cur_moliculeid)
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
                data_dict[molkey]["species"] = np.array(cur_atoms.get_chemical_symbols())[atom_index]
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
    for isokey in data_dict:
        frames = len(data_dict[isokey]['_id'])
        for key, values in data_dict[isokey].items():
            if key not in ('species', PAIR_INDEX_DATASET) and len(values) != frames:
                raise ValueError(f'{isokey}:{key} is not aligned with the {frames} stored frames.')
        # print('isokeys:',isokey)
        for propkey in data_dict[isokey]:
            if propkey.lower() in ['species','_id']:
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
    partial = h5path + PARTIAL_SUFFIX
    try:
        # Mode 'w' overwrites a .partial left by a crash.
        dpack = pyt.datapacker(partial, mode='w')
        try:
            for key in data_dict:
                dpack.store_data(key, **data_dict[key])
            if sources.sources:
                dpack.store.attrs[LABEL_PROVENANCE] = json.dumps(sources.sources)
            for name, value in (attributes or {}).items():
                dpack.store.attrs[name] = value
        finally:
            dpack.cleanup()
        fsync_file(partial)
        os.replace(partial, h5path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return saved_number


class SourceCollector:
    """The distinct ``label_provenance`` records of labeled structures, in first-seen order, added one at a time."""

    def __init__(self):
        self.sources, self._seen = [], set()

    def add(self, structure):
        provenance = structure.get_metadata().get(LABEL_PROVENANCE)
        if provenance is None:
            return
        key = json.dumps(provenance, sort_keys=True, default=str)
        if key not in self._seen:
            self._seen.add(key)
            self.sources.append(json.loads(key))


def label_sources(structures):
    """Return the distinct ``label_provenance`` records of labeled structures, in first-seen order."""
    collector = SourceCollector()
    for structure in structures:
        collector.add(structure)
    return collector.sources


def read_label_provenance(h5path):
    """Return the label sources recorded on an HDF5 shard, or an empty list if none were recorded.

    QM labelers record nothing; foundation-model teachers record themselves. Reads only the
    shard's attributes, not its data.
    """
    import h5py
    with h5py.File(h5path, 'r') as handle:
        value = handle.attrs.get(LABEL_PROVENANCE)
    if value is None:
        return []
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def write_label_sources(ensemble_dir, h5_dir):
    """Record which label sources a trained ensemble learned from, as ``label_sources.json``.

    A failure here is reported, never raised, so it can't cost a finished training.
    """
    try:
        shards = sorted(glob.glob(os.path.join(str(h5_dir), '*.h5')))
        teachers, seen, unrecorded = [], set(), []
        for shard in shards:
            sources = read_label_provenance(shard)
            if not sources:
                unrecorded.append(os.path.basename(shard))
            for source in sources:
                key = json.dumps(source, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    teachers.append(source)
        Path(ensemble_dir).mkdir(parents=True, exist_ok=True)
        (Path(ensemble_dir) / 'label_sources.json').write_text(json.dumps(
            {'teachers': teachers, 'shards_without_label_provenance': unrecorded, 'shards': len(shards)}, indent=2) + '\n')
    except Exception as exc:
        print('Could not record label sources in {}: {}: {}'.format(ensemble_dir, type(exc).__name__, exc), flush=True)
