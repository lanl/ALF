import numpy as np

from alframework.tools.pyanitools import anidataloader, datapacker


def test_datapacker_and_anidataloader_round_trip_arrays_and_strings(tmp_path):
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


def test_anidataloader_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.h5"

    try:
        anidataloader(str(missing_path))
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
