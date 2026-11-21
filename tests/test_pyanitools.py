"""pyanitools: saving and reading ANI-style HDF5 datasets."""

import numpy as np
import pytest

from alframework.tools.pyanitools import anidataloader, datapacker


def test_anidataloader_reads_back_saved_data(tmp_path):
    h5_path = tmp_path / "dataset.h5"
    packer = datapacker(str(h5_path))
    packer.store_data(
        "water",
        species=["O", "H", "H"],
        coordinates=np.array([[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]]]),
        energy=np.array([-76.0]),
    )
    packer.cleanup()

    loader = anidataloader(str(h5_path))
    try:
        assert loader.group_size() == 1
        assert loader.size() == 3

        loaded = loader.get_data("water")
        assert loaded["path"] == "/water"
        assert loaded["species"] == ["O", "H", "H"]
        np.testing.assert_allclose(loaded["energy"], [-76.0])

        iterated = list(loader)
        assert len(iterated) == 1
        np.testing.assert_allclose(iterated[0]["coordinates"], [[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]]])
    finally:
        loader.cleanup()


def test_anidataloader_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        anidataloader(str(tmp_path / "missing.h5"))


def test_anidataloader_reads_what_read_shard_reads(tmp_path):
    """The loader reads through read_shard's group reader: nested groups and single values included."""
    import h5py

    from alframework.tools.h5_parallel import read_shard

    h5_path = tmp_path / "dataset.h5"
    packer = datapacker(str(h5_path))
    packer.store_data("water", species=["O", "H", "H"], energy=np.array([-76.0, -75.9]))
    packer.cleanup()
    with h5py.File(h5_path, "a") as handle:
        group = handle.create_group("outer").create_group("ammonia")
        group.create_dataset("species", data=np.array([b"N", b"H", b"H", b"H"]))
        group.create_dataset("charge", data=0)

    expected = list(read_shard(h5_path, workers=1))
    loader = anidataloader(str(h5_path))
    try:
        found = list(loader)
        assert [group["path"] for group in found] == ["/outer/ammonia", "/water"]
        assert [group["path"] for group in found] == [group["path"] for group in expected]
        for got, want in zip(found, expected):
            assert got.keys() == want.keys()
            for key in got:
                assert np.array_equal(got[key], want[key]), key
        assert found[0]["species"] == ["N", "H", "H", "H"] and found[0]["charge"] == 0
        assert [group["path"] for group in loader.iter_group(loader.store["outer"])] == ["/ammonia"]
    finally:
        loader.cleanup()
