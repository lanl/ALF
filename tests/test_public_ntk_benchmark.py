"""Public data layouts, split reproducibility, and fixed-budget NTK replay metrics."""

import h5py
import numpy as np
import pytest
from ase import units

from examples.atom_resolved_uncertainty import benchmark_ntk as benchmark


@pytest.mark.parametrize("dataset", benchmark.DATASETS)
def test_prepare_preserves_units_atom_order_and_disjoint_trajectory_blocks(tmp_path, dataset):
    numbers = np.array([8, 1, 6])
    positions = np.arange(900, dtype=float).reshape(-1, 3, 3) / 100
    energies = np.arange(len(positions)).reshape(-1, 1)
    forces = -positions
    chronology = np.random.default_rng(8).permutation(len(positions))
    if dataset == "rmd17-aspirin":
        arrays = dict(nuclear_charges=numbers, coords=positions, energies=energies.ravel(), forces=forces,
                      old_indices=chronology)
    else:
        arrays = dict(z=numbers, R=positions, E=energies, F=forces)
        chronology = np.arange(len(positions))
    source = tmp_path / "source.npz"
    np.savez(source, **arrays)
    options = dict(train_count=12, calibration_count=8, test_count=8, seed=375, gap=3)
    out = tmp_path / "data"
    manifest = benchmark.prepare(dataset, source, out, **options)
    repeat = benchmark.prepare(dataset, source, tmp_path / "repeat", **options)
    assert manifest["input_sha256"] == repeat["input_sha256"]
    assert manifest["atomic_numbers"] == numbers.tolist()
    groups = []
    for name, spec in manifest["splits"].items():
        rows = spec["source_rows"]
        assert rows == repeat["splits"][name]["source_rows"]
        assert spec["trajectory_indices"] == sorted(chronology[rows])
        groups.append(set(rows))
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert min(manifest["splits"]["calibration"]["trajectory_indices"]) >= 63
    assert min(manifest["splits"]["test"]["trajectory_indices"]) >= 83

    rows = manifest["splits"]["training"]["source_rows"]
    with h5py.File(out / benchmark.TRAIN_SHARD) as handle:
        group = handle[benchmark.TRAIN_GROUP]
        assert group["species"][:].tolist() == [b"O", b"H", b"C"]
        np.testing.assert_array_equal(group["coordinates"], positions[rows])
        np.testing.assert_allclose(group["energy"], energies[rows, 0] * units.kcal / units.mol)
        np.testing.assert_allclose(group["forces"], forces[rows] * units.kcal / units.mol)
    pool = benchmark._read_pool(out, manifest, "test")
    np.testing.assert_array_equal(pool["numbers"], numbers)
    np.testing.assert_allclose(pool["forces"], forces[pool["source_rows"]] * units.kcal / units.mol)
    with (out / "test.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="does not match"):
        benchmark._read_pool(out, manifest, "test")


def _frame(row, selected, energy, force_error=0, energy_error=0):
    return dict(source_row=row, ensemble_selected=selected, ntk_selected=selected,
                ntk_projected=selected, ntk_joint_force=selected, ntk_total_energy=energy,
                selected_force_error_eV_per_A=force_error, energy_residual_eV=energy_error)


def test_calibration_and_energy_trigger_change_selection_at_equal_budget():
    calibration = benchmark._calibrate(
        [_frame(1000, 1.0, 2.0, 0.5, 3.0), _frame(1001, 1.0, 2.0, 0.5, 3.0)], None, None)
    frames = [_frame(row, 1.0, 1.0, 0, 3.0) for row in range(100, 0, -1)]
    frames.extend([_frame(20 + 100, 5.0, 1.0, 0, 3.0), _frame(10 + 100, 5.0, 1.0, 0, 3.0),
                   _frame(130, 0.1, 100.0, 2.0, 4.0), _frame(140, 4.0, 1.0, 2.0, 3.0)])
    # Use exactly 100 test frames so the 1% budget selects one frame.
    frames = frames[4:]
    metrics = benchmark._budget_metrics(frames, calibration)
    assert calibration["energy_residual_offset_eV"] == 3.0
    force_only = metrics["methods"]["ntk_selected"][0]
    joint = metrics["methods"]["ntk_selected_energy"][0]
    assert force_only["selected_source_rows"] == [110]
    assert joint["selected_source_rows"] == [130]
    assert force_only["large_force_error_missed"] == 2
    assert joint["large_force_error_missed"] == 1
    assert joint["large_energy_error_missed"] == 0
    assert benchmark._budget_metrics(list(reversed(frames)), calibration) == metrics
