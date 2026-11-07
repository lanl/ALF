"""Restarting after a crash: whatever a crash leaves on disk must let the same command continue.

Status and HDF5 shards are written to a temporary name and renamed when complete, so a crash leaves
either the old file or the new one, never half of one.
"""

import ast
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from alframework.tools import files, pyanitools
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import find_empty_directory, store_current_data, write_label_sources, write_status
from tests.helpers.driver import run_alf_driver, stored_ids
from tests.helpers.molecules import WATER

PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}
MAIN = Path(__file__).resolve().parents[1] / "alframework" / "__main__.py"


def labeled(name="water-0", energy=-1.0):
    molecule = MoleculesObject(WATER.copy(), name)
    molecule.store_results({"energy": energy, "forces": np.zeros((len(WATER), 3))})
    molecule.set_converged_flag(True)
    return molecule


def crash_inside_store_data(monkeypatch):
    """Fail after the first group is written, as a crash partway through a store would."""
    original = pyanitools.datapacker.store_data

    def store_then_crash(self, store_loc, **kwargs):
        original(self, store_loc, **kwargs)
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(pyanitools.datapacker, "store_data", store_then_crash)


def test_write_status_replaces_the_file(tmp_path):
    path = tmp_path / "status.json"
    write_status(str(path), {"current_h5_id": 1})
    write_status(str(path), {"current_h5_id": 2})
    assert json.loads(path.read_text()) == {"current_h5_id": 2}
    assert os.listdir(tmp_path) == ["status.json"]


def test_write_status_keeps_the_old_status_when_a_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    write_status(str(path), {"current_h5_id": 1})

    def write_half_then_fail(values, handle, **kwargs):
        handle.write('{"current_h5_id": ')
        raise OSError("node lost")

    monkeypatch.setattr(files.json, "dump", write_half_then_fail)
    with pytest.raises(OSError, match="node lost"):
        write_status(str(path), {"current_h5_id": 2})
    assert json.loads(path.read_text()) == {"current_h5_id": 1}


def test_coordinator_writes_status_only_through_write_status():
    """A direct ``open(..., "w")`` on the status file would bring back truncated status after a crash."""
    direct = [node.lineno for node in ast.walk(ast.parse(MAIN.read_text()))
              if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"
              and "status_path" in ast.unparse(node.args[0])
              and "w" in ast.unparse(node.args[1:2] or [ast.Constant("r")])]
    assert direct == []
    assert MAIN.read_text().count("write_status(master_config['status_path'], status)") == 6


def test_store_current_data_leaves_nothing_when_a_store_fails(tmp_path, monkeypatch):
    crash_inside_store_data(monkeypatch)
    with pytest.raises(OSError, match="disk quota"):
        store_current_data(str(tmp_path / "data-0000.h5"), [labeled()], PROPERTIES)
    assert os.listdir(tmp_path) == []


def test_store_current_data_refuses_to_replace_a_shard(tmp_path):
    path = tmp_path / "data-0000.h5"
    store_current_data(str(path), [labeled(energy=-1.0)], PROPERTIES)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="stopped between storing this shard and saving status"):
        store_current_data(str(path), [labeled(energy=-2.0)], PROPERTIES)
    assert path.read_bytes() == before
    assert os.listdir(tmp_path) == ["data-0000.h5"]


def test_store_current_data_overwrites_a_partial_shard_left_by_a_crash(tmp_path):
    path = tmp_path / "data-0000.h5"
    Path(str(path) + ".partial").write_bytes(b"half a shard")
    assert store_current_data(str(path), [labeled()], PROPERTIES) == 1
    assert os.listdir(tmp_path) == ["data-0000.h5"]
    with h5py.File(path) as handle:
        np.testing.assert_allclose(handle["H02_O01"]["energy"][()], [-1.0])


def test_readers_ignore_a_partial_shard(tmp_path):
    h5 = tmp_path / "h5"
    h5.mkdir()
    store_current_data(str(h5 / "data-0000.h5"), [labeled()], PROPERTIES)
    (h5 / "data-0001.h5.partial").write_bytes(b"half a shard")

    assert find_empty_directory(str(h5 / "data-{:04d}.h5")) == 1
    write_label_sources(str(tmp_path / "model"), str(h5))
    assert json.loads((tmp_path / "model" / "label_sources.json").read_text()) == {
        "teachers": [], "shards_without_label_provenance": ["data-0000.h5"], "shards": 1}

    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB, _h5_shards

    assert _h5_shards(str(h5)) == [str(h5 / "data-0000.h5")]
    database = ALFPyAniDirectoryDB(directory=str(h5), inputs=["species", "coordinates"],
                                   targets=["energy", "forces"], quiet=True, seed=0)
    assert len(database.arr_dict["energy"]) == 1


def coordinator_tasks():
    from parsl import python_app

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(WATER.copy(), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        molecule_object.store_results({"energy": -1.0, "forces": np.zeros((len(molecule_object), 3))})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml(current_training_id):
        return [True, True], current_training_id

    return dict(builder=builder, sampler=sampler, qm=qm, ml=ml, bootstrap_set=2, parallel_samplers=2,
                target_queued_QM=2, save_h5_threshold=3)


def test_coordinator_restarts_after_a_crash_during_the_first_store(tmp_path, monkeypatch):
    with monkeypatch.context() as crash:
        crash_inside_store_data(crash)
        with pytest.raises(OSError, match="disk quota"):
            run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["current_h5_id"] == 0 and status["current_model_id"] == -1
    assert os.listdir(tmp_path / "h5") == []

    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    assert run.status["current_model_id"] >= 1
    assert stored_ids(tmp_path / "h5")
    assert not list((tmp_path / "h5").glob("*.partial"))


def test_coordinator_starts_with_a_partial_shard_left_by_a_crash(tmp_path, monkeypatch):
    (tmp_path / "h5").mkdir()
    (tmp_path / "h5" / "data-0000.h5.partial").write_bytes(b"half a shard")
    run = run_alf_driver(tmp_path, monkeypatch, **coordinator_tasks())
    assert run.status["current_model_id"] >= 1
    assert sorted(os.listdir(tmp_path / "h5"))[0] == "data-0000.h5"
    assert not list((tmp_path / "h5").glob("*.partial"))
