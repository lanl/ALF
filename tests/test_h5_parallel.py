"""Reading a label file's groups in several processes gives exactly what the serial read gives."""

import numpy as np
import pytest

from alframework.tools import h5_parallel


def _write_shard(directory):
    """Three compositions of different sizes, with byte-string IDs and species, as stored labels are."""
    import h5py
    rng = np.random.default_rng(7)
    directory.mkdir()
    with h5py.File(directory / "data.h5", "w") as handle:
        for name, species, frames in (("H2O", ["O", "H", "H"], 5), ("CH4", ["C", "H", "H", "H", "H"], 7),
                                      ("NH3", ["N", "H", "H", "H"], 3)):
            group = handle.create_group(name)
            atoms = len(species)
            group.create_dataset("species", data=np.array(species, dtype="S"), compression="gzip")
            group.create_dataset("coordinates", data=rng.normal(size=(frames, atoms, 3)), compression="gzip")
            group.create_dataset("forces", data=rng.normal(size=(frames, atoms, 3)), compression="gzip")
            group.create_dataset("energy", data=rng.normal(size=frames), compression="gzip")
            group.create_dataset("_id", data=np.array([f"{name}-{i}" for i in range(frames)], dtype="S"))


def test_each_group_reads_as_hippynns_loader_reads_it(tmp_path):
    pytest.importorskip("hippynn")
    from hippynn.databases._ani_reader import AniDataLoader
    _write_shard(tmp_path / "data")
    file = str(tmp_path / "data" / "data.h5")
    loader = AniDataLoader(file)
    expected = list(loader)
    loader.store.close()
    found = list(h5_parallel.read_groups(file, h5_parallel.group_paths(file), 2))
    assert [group["path"] for group in found] == [group["path"] for group in expected]
    for got, want in zip(found, expected):
        assert got.keys() == want.keys()
        for key in want:
            assert np.array_equal(np.asarray(got[key]), np.asarray(want[key])), key


def test_a_database_read_in_parallel_matches_the_serial_read(tmp_path, monkeypatch):
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn_interface import ALFPyAniDirectoryDB
    _write_shard(tmp_path / "data")

    def load(workers):
        monkeypatch.setenv(h5_parallel.READ_WORKERS, str(workers))
        return ALFPyAniDirectoryDB(directory=str(tmp_path / "data"), allow_unfound=True, quiet=True, seed=1,
                                   inputs=None, targets=None, species_key="species",
                                   coordinates_key="coordinates", atom_keys=("coordinates", "forces"))

    serial = load(1).arr_dict
    used = []
    read_groups = h5_parallel.read_groups
    monkeypatch.setattr(h5_parallel, "read_groups", lambda *args: used.append(args[2]) or read_groups(*args))
    parallel = load(3).arr_dict
    assert used == [3], "the second load reads in parallel"
    assert serial.keys() == parallel.keys()
    for key in serial:
        assert serial[key].dtype == parallel[key].dtype and np.array_equal(serial[key], parallel[key]), key


def test_small_files_read_serially_unless_asked(tmp_path, monkeypatch):
    _write_shard(tmp_path / "data")
    file = str(tmp_path / "data" / "data.h5")
    monkeypatch.delenv(h5_parallel.READ_WORKERS, raising=False)
    assert h5_parallel.workers_for(file, 3) == 1
    monkeypatch.setenv(h5_parallel.READ_WORKERS, "8")
    assert h5_parallel.workers_for(file, 3) == 3
