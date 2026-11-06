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
