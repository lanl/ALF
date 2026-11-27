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
    found = list(h5_parallel.read_shard(file, workers=2))
    assert [group["path"] for group in found] == [group["path"] for group in expected]
    for got, want in zip(found, expected):
        assert got.keys() == want.keys()
        for key in want:
            assert np.array_equal(np.asarray(got[key]), np.asarray(want[key])), key


def test_a_database_read_in_parallel_matches_the_serial_read(tmp_path, monkeypatch):
    pytest.importorskip("hippynn")
    from alframework.ml_interfaces.hippynn.interface import ALFPyAniDirectoryDB
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


@pytest.mark.parametrize("workers", [None, 1])
def test_serial_shard_reuses_one_file_handle(tmp_path, monkeypatch, workers):
    import h5py

    _write_shard(tmp_path / "data")
    path = tmp_path / "data" / "data.h5"
    monkeypatch.delenv(h5_parallel.READ_WORKERS, raising=False)
    open_file = h5py.File
    handles = []

    def track_open(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", track_open)
    groups = list(h5_parallel.read_shard(path, keys={"species", "energy"}, workers=workers))
    assert len(handles) == 1
    assert not handles[0].id.valid
    assert [group["path"] for group in groups] == ["/CH4", "/H2O", "/NH3"]
    assert all(set(group) == {"path", "species", "energy"} for group in groups)


@pytest.mark.parametrize("workers", [1, 2])
def test_nested_groups_and_single_values_read_like_any_group(tmp_path, workers):
    import h5py

    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        water = handle.create_group("water")
        water.create_dataset("species", data=np.array([b"O", b"H", b"H"]))
        water.create_dataset("energy", data=np.array([-76.0, -75.9]))
        ammonia = handle.create_group("outer").create_group("ammonia")
        ammonia.create_dataset("species", data=np.array([b"N", b"H", b"H", b"H"]))
        ammonia.create_dataset("charge", data=0)

    groups = list(h5_parallel.read_shard(path, workers=workers))
    assert [group["path"] for group in groups] == ["/outer/ammonia", "/water"]
    assert groups[0]["species"] == ["N", "H", "H", "H"] and groups[0]["charge"] == 0
    assert groups[1]["species"] == ["O", "H", "H"]
    np.testing.assert_array_equal(groups[1]["energy"], [-76.0, -75.9])


def _alf_label_files(tmp_path):
    """A shard and a snapshot as ALF writes them, with formula groups stored out of name order."""
    from ase import Atoms
    from alframework.tools.data_shards import store_current_data
    from alframework.tools.label_parts import HDF5HeldLabels
    from alframework.tools.molecules import MoleculesObject
    properties = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}
    labels = []
    for index, formula in enumerate(("NH3", "H2O", "CH4", "H2O", "CO2")):
        atoms = Atoms(formula, positions=np.arange(3 * len(Atoms(formula))).reshape(-1, 3), cell=[9, 9, 9],
                      pbc=True)
        molecule = MoleculesObject(atoms, f"mol-{index}")
        molecule.store_results({"energy": -index, "forces": np.zeros((len(atoms), 3))})
        molecule.set_converged_flag(True)
        labels.append(molecule)
    shard = tmp_path / "shard" / "data.h5"
    shard.parent.mkdir()
    store_current_data(str(shard), labels, properties)
    snapshot = tmp_path / "snapshot" / "data.h5"
    held = HDF5HeldLabels(snapshot, properties, batch_size=2)
    held.hold(labels)
    held.commit(snapshot, properties, {})
    return shard, snapshot


def _walked(path, group_keys):
    """The groups the general walk selects, from a copy of ``path`` without ALF's schema attribute."""
    import shutil
    import h5py
    from alframework.tools.label_hdf5 import SCHEMA_VERSION
    copy = path.with_name("unmarked-" + path.name)  # beside the original, so a snapshot's parts still resolve
    shutil.copy(path, copy)
    with h5py.File(copy, "r+") as handle:
        del handle.attrs[SCHEMA_VERSION]
    with h5py.File(copy, "r") as handle:
        return [name for name, _ in h5_parallel._structure_groups(handle, group_keys=group_keys)]


@pytest.mark.parametrize("group_keys", [None, {"coordinates", "species"}, {"missing"}])
def test_alf_label_files_list_the_same_groups_without_opening_their_datasets(tmp_path, monkeypatch, group_keys):
    import h5py
    shard, snapshot = _alf_label_files(tmp_path)
    for path in (shard, snapshot):
        expected = _walked(path, group_keys)
        opened = []
        dataset_init = h5py.Dataset.__init__
        monkeypatch.setattr(h5py.Dataset, "__init__", lambda self, *a, **k: opened.append(1) or dataset_init(
            self, *a, **k))
        with h5py.File(path, "r") as handle:
            assert [name for name, _ in h5_parallel._structure_groups(handle, group_keys=group_keys)] == expected
        monkeypatch.setattr(h5py.Dataset, "__init__", dataset_init)
        assert not opened, f"{path.parent.name}: listing opened {len(opened)} datasets"
        assert len(expected) == (0 if group_keys == {"missing"} else 4), "four formulas, two labels of water"


def test_alf_label_files_read_the_same_through_the_fast_listing(tmp_path):
    shard, snapshot = _alf_label_files(tmp_path)
    for path in (shard, snapshot):
        found = list(h5_parallel.read_shard(path, workers=1, group_keys={"coordinates", "species"}))
        assert [group["path"] for group in found] == _walked(path, {"coordinates", "species"})
        assert sum(len(group["energy"]) for group in found) == 5


def test_a_missing_file_raises_when_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(h5_parallel.read_shard(tmp_path / "missing.h5"))


def test_parallel_budget_releases_in_order_and_runs_large_groups_alone(tmp_path):
    from concurrent.futures import Future

    jobs = iter([("a", 4), ("b", 6), ("large", 20), ("c", 4), ("d", 5)])
    submitted = []

    class Pool:
        def submit(self, function, file, path, keys):
            submitted.append(path)
            future = Future()
            future.set_result(path)
            return future

    groups = h5_parallel._read_in_order(Pool(), tmp_path / "data.h5", jobs, 4, None, 10)
    assert next(groups) == "a" and submitted == ["a", "b"]
    assert next(groups) == "b" and submitted == ["a", "b"]
    assert next(groups) == "large" and submitted == ["a", "b", "large"]
    assert list(groups) == ["c", "d"] and submitted == ["a", "b", "large", "c", "d"]


@pytest.mark.parametrize("species", [[b"O", b"H", b"H"], np.array([b"O", b"H", b"H"]), [8, 1, 1]])
def test_hippynn_decodes_root_level_species_from_an_explicit_shard(tmp_path, species):
    pytest.importorskip("hippynn")
    import h5py
    from alframework.ml_interfaces.hippynn.interface import ALFPyAniDirectoryDB

    file = tmp_path / "data.hdf5"
    with h5py.File(file, "w") as handle:
        handle["Z"] = species
        handle["positions"] = np.zeros((2, 3, 3))
        handle["energy"] = [-1., -2.]
    database = ALFPyAniDirectoryDB(
        file, inputs=["Z", "positions"], targets=["energy"], quiet=True, seed=1, read_workers=1,
        species_key="Z", coordinates_key="positions")
    restored = database.restarter.attempt_restart()
    for reader in (database, restored):
        np.testing.assert_array_equal(reader.arr_dict["Z"].numpy(), [[8, 1, 1], [8, 1, 1]])
