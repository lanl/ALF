"""Record what produced each label, so a shard and a trained model can say what they learned from.

A labeler puts a description of itself in each label's ``label_source`` metadata: a foundation-model teacher
its checkpoint and package, an ASE task its calculator and method settings. Storage writes the distinct
sources onto the HDF5 shard as its ``label_sources`` attribute, and after training ALF writes the sources of
every shard it learned from to ``label_sources.json`` beside the ensemble. Labelers that record nothing, such
as the legacy QM tasks, leave their shards without sources.

Labels and shards written before the rename used ``label_provenance`` for both; they are still read.
"""

import glob
import json
import os
from pathlib import Path

# A label's metadata key, and a shard's attribute.
LABEL_SOURCE = 'label_source'
SHARD_LABEL_SOURCES = 'label_sources'
# The earlier name for both.
LEGACY_LABEL_PROVENANCE = 'label_provenance'


def label_source(structure):
    """The source recorded in a labeled structure's metadata, or None."""
    metadata = structure.get_metadata()
    source = metadata.get(LABEL_SOURCE)
    return metadata.get(LEGACY_LABEL_PROVENANCE) if source is None else source


class LabelSourceCollector:
    """The distinct sources of labeled structures, in first-seen order, added one structure at a time."""

    def __init__(self):
        self.sources, self._seen = [], set()

    def add(self, structure):
        source = label_source(structure)
        if source is None:
            return
        key = json.dumps(source, sort_keys=True, default=str)
        if key not in self._seen:
            self._seen.add(key)
            self.sources.append(json.loads(key))


def label_sources(structures):
    """The distinct sources of labeled structures, in first-seen order."""
    collector = LabelSourceCollector()
    for structure in structures:
        collector.add(structure)
    return collector.sources


def write_shard_label_sources(store, sources):
    """Record ``sources`` on an open HDF5 shard; a shard whose labels have none gets no attribute."""
    if sources:
        store.attrs[SHARD_LABEL_SOURCES] = json.dumps(sources)


def read_label_sources(h5path):
    """The label sources recorded on an HDF5 shard, or an empty list. Reads only the shard's attributes."""
    import h5py
    with h5py.File(h5path, 'r') as handle:
        value = handle.attrs.get(SHARD_LABEL_SOURCES)
        if value is None:
            value = handle.attrs.get(LEGACY_LABEL_PROVENANCE)
    if value is None:
        return []
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def write_label_sources(ensemble_dir, h5_dir):
    """Record which label sources a trained ensemble learned from, as ``label_sources.json``.

    The file lists the distinct sources of every shard in ``h5_dir``, the shards that recorded none, and the
    number of shards. A failure here is reported, never raised, so it can't cost a finished training.
    """
    try:
        shards = sorted(glob.glob(os.path.join(str(h5_dir), '*.h5')))
        sources, seen, unrecorded = [], set(), []
        for shard in shards:
            recorded = read_label_sources(shard)
            if not recorded:
                unrecorded.append(os.path.basename(shard))
            for source in recorded:
                key = json.dumps(source, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    sources.append(source)
        Path(ensemble_dir).mkdir(parents=True, exist_ok=True)
        (Path(ensemble_dir) / 'label_sources.json').write_text(json.dumps(
            {'label_sources': sources, 'shards_without_label_sources': unrecorded, 'shards': len(shards)},
            indent=2) + '\n')
    except Exception as exc:
        print(f'Could not record label sources in {ensemble_dir}: {type(exc).__name__}: {exc}', flush=True)
