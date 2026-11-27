"""Label sources: what produced each label, recorded on the label, its shard, and the models trained on it."""
import json

import h5py
import numpy as np

from alframework.tools import label_sources
from alframework.tools.data_shards import store_current_data
from tests.helpers.labels import PROPERTIES, labeled_h2

SOURCE = {"labeler": "example", "method": "PBE"}


def labeled(source=None, key=label_sources.LABEL_SOURCE):
    molecule = labeled_h2("h2")
    if source is not None:
        molecule.update_metadata({key: source})
    return molecule


def test_a_shard_records_each_distinct_source_once(tmp_path):
    other = dict(SOURCE, method="B3LYP")
    store_current_data(str(tmp_path / "data.h5"), [labeled(SOURCE), labeled(SOURCE), labeled(other)], PROPERTIES)
    assert label_sources.read_label_sources(tmp_path / "data.h5") == [SOURCE, other]


def test_a_shard_whose_labels_have_no_source_records_none(tmp_path):
    store_current_data(str(tmp_path / "data.h5"), [labeled()], PROPERTIES)
    with h5py.File(tmp_path / "data.h5") as handle:
        assert label_sources.SHARD_LABEL_SOURCES not in handle.attrs
    assert label_sources.read_label_sources(tmp_path / "data.h5") == []


def test_a_label_held_under_the_old_metadata_key_keeps_its_source(tmp_path):
    molecule = labeled(SOURCE, key=label_sources.LEGACY_LABEL_PROVENANCE)
    assert label_sources.label_source(molecule) == SOURCE
    store_current_data(str(tmp_path / "data.h5"), [molecule], PROPERTIES)
    assert label_sources.read_label_sources(tmp_path / "data.h5") == [SOURCE]


def test_a_shard_written_under_the_old_attribute_is_still_read(tmp_path):
    with h5py.File(tmp_path / "old.h5", "w") as handle:
        handle.attrs[label_sources.LEGACY_LABEL_PROVENANCE] = json.dumps([SOURCE])
    assert label_sources.read_label_sources(tmp_path / "old.h5") == [SOURCE]


def test_a_trained_model_lists_its_shards_sources_and_the_shards_without_any(tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    store_current_data(str(h5 / "data-0000.h5"), [labeled(SOURCE)], PROPERTIES)
    store_current_data(str(h5 / "data-0001.h5"), [labeled()], PROPERTIES)
    label_sources.write_label_sources(tmp_path / "model", h5)
    assert json.loads((tmp_path / "model" / "label_sources.json").read_text()) == {
        "label_sources": [SOURCE], "shards_without_label_sources": ["data-0001.h5"], "shards": 2}


def test_label_provenance_uses_the_frozen_inventory(tmp_path):
    from alframework.tools.ensemble_training import training_shards

    h5 = tmp_path / "h5"
    h5.mkdir()
    seed = h5 / "seed.hdf5"
    store_current_data(str(seed), [labeled(SOURCE)], PROPERTIES)
    shards = training_shards(h5)
    store_current_data(str(h5 / "later.h5"), [labeled()], PROPERTIES)
    label_sources.write_label_sources(tmp_path / "model", shards=shards)
    record = json.loads((tmp_path / "model" / "label_sources.json").read_text())
    assert shards == record["training_shards"] == [str(seed)]
    assert record["label_sources"] == [SOURCE]


def test_sources_are_compared_by_value_whatever_their_key_order():
    collector = label_sources.LabelSourceCollector()
    for source in ({"a": 1, "b": np.float64(2.0)}, {"b": 2.0, "a": 1}):
        collector.add(labeled(source))
    assert collector.sources == [{"a": 1, "b": 2.0}]
