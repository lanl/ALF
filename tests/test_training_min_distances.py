"""The low-distance filter before training: ALF's batched search gives hippynn's minimum distances."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")
pytest.importorskip("torch_sim")

from hippynn.pretraining import calculate_min_dists

from alframework.ml_interfaces.hippynn.interface import training_minimum_distances
from alframework.tools import geometry

ATOMS = 24


def periodic_arrays(seed=0, systems=40):
    """Padded periodic systems of 12-24 atoms in cubic boxes, some with close pairs, as the database holds them."""
    rng = np.random.default_rng(seed)
    species = np.zeros((systems, ATOMS), dtype=np.int64)
    positions = np.zeros((systems, ATOMS, 3), dtype=np.float32)
    cells = np.zeros((systems, 3, 3), dtype=np.float32)
    for i in range(systems):
        n = int(rng.integers(12, ATOMS + 1))
        box = float(rng.uniform(7.0, 10.0))
        species[i, :n] = rng.choice([3, 9, 17], size=n)
        positions[i, :n] = rng.uniform(0, box, size=(n, 3))
        if i % 4 == 0:  # a close pair, across the boundary every other time
            positions[i, 1] = positions[i, 0] + np.array([0.5, 0.2, 0.1])
            if i % 8 == 0:
                positions[i, 0, 0], positions[i, 1, 0] = 0.1, box - 0.3
        cells[i] = np.eye(3) * box
    return {"Z": torch.as_tensor(species), "R": torch.as_tensor(positions), "cell": torch.as_tensor(cells)}


def hippynn_reference(arrays, cell="cell"):
    return calculate_min_dists(arrays, species_name="Z", positions_name="R", dist_hard_max=6.0, cell_name=cell,
                               batch_size=16, device="cpu")


def test_periodic_minimum_distances_match_hippynn_and_keep_the_same_structures():
    arrays = periodic_arrays()
    ours = training_minimum_distances(arrays, "Z", "R", "cell", "cpu", batch_size=16)
    theirs = hippynn_reference(arrays)
    assert ours.dtype == arrays["R"].dtype and ours.device.type == "cpu"
    assert torch.allclose(ours, theirs, rtol=1e-5, atol=1e-5)
    assert torch.equal(ours > 0.8, theirs > 0.8)


def test_structures_without_a_close_pair_get_their_exact_minimum_and_are_kept_as_by_hippynn():
    """No pair within ALF's 3 A search radius: the exact minimum-image minimum. hippynn gives a structure with
    no neighbor within 6 A the largest distance in its batch instead; both keep it."""
    arrays = periodic_arrays(systems=3)
    arrays["Z"][1] = 0
    arrays["Z"][1, :2] = torch.tensor([3, 9])
    arrays["R"][1, :2] = torch.tensor([[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]])  # 4 A apart, box 20 A
    arrays["cell"][1] = torch.eye(3) * 20.0
    arrays["Z"][2] = 0
    arrays["Z"][2, :2] = torch.tensor([3, 9])
    arrays["R"][2, :2] = torch.tensor([[1.0, 1.0, 1.0], [11.0, 1.0, 1.0]])  # 10 A apart, box 30 A: no neighbor
    arrays["cell"][2] = torch.eye(3) * 30.0
    ours = training_minimum_distances(arrays, "Z", "R", "cell", "cpu", batch_size=16)
    theirs = hippynn_reference(arrays)
    assert torch.allclose(ours[:2], theirs[:2], rtol=1e-5, atol=1e-5)
    assert ours[1] == pytest.approx(4.0, abs=1e-4) and ours[2] == pytest.approx(10.0, abs=1e-4)
    assert theirs[2] < 6.0, "hippynn's placeholder: the largest distance found in the batch"
    assert torch.equal(ours > 0.8, theirs > 0.8)


def test_nonperiodic_data_and_no_batched_search_use_hippynn(monkeypatch):
    arrays = periodic_arrays(systems=8)
    open_arrays = {k: v for k, v in arrays.items() if k != "cell"}
    assert torch.equal(training_minimum_distances(open_arrays, "Z", "R", None, "cpu", batch_size=16),
                       hippynn_reference(open_arrays, cell=None))
    monkeypatch.setattr(geometry, "_batched_neighbor_search", lambda: None)
    assert torch.equal(training_minimum_distances(arrays, "Z", "R", "cell", "cpu", batch_size=16),
                       hippynn_reference(arrays))


def test_screening_shared_among_devices_gives_the_same_distances():
    """Batches shared among several devices, as a member's GPUs share them, give exactly the same values."""
    arrays = periodic_arrays(seed=3, systems=53)
    one = training_minimum_distances(arrays, "Z", "R", "cell", "cpu", batch_size=8)
    three = training_minimum_distances(arrays, "Z", "R", "cell", "cpu", batch_size=8, devices=["cpu"] * 3)
    assert torch.equal(one, three)
