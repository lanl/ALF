"""Geometry builders: condensed-phase and salt builders, external geometry import, and RDKit seed
preparation.

Tests that call RDKit skip without it.
"""

import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers
from ase.io import write

from alframework.builders import rdkit_molecule as seed
from alframework.builders.builders import (
    atomic_system_builder,
    construct_simulation_box,
    create_atomic_system,
    simple_condensed_phase_builder_task,
    simple_multi_condensed_phase_builder_task,
    to_mic,
)
from alframework.builders.external_geometry import (
    build_external_geometries,
    external_geometry_builder_task,
    geometry_manifest,
)
from alframework.tools.molecules_class import MoleculesObject


@pytest.mark.parametrize(
    "atom_charges,target_num_atoms",
    [
        ({"Na": 1, "Cl": -1}, 8),
        ({"Mg": 2, "Cl": -1}, 9),
        ({"Ca": 2, "F": -1, "O": -2}, 10),
    ],
    ids=["NaCl", "MgCl2", "CaFO"],
)
def test_atomic_system_is_charge_neutral(atom_charges, target_num_atoms):
    random.seed(1)

    system = create_atomic_system(atom_charges, target_num_atoms)

    total_charge = sum(atom_charges[atom] * count for atom, count in system.items())
    assert total_charge == 0
    assert sum(system.values()) >= target_num_atoms - 1
    assert set(system).issubset(atom_charges)


def test_distances_wrap_to_minimum_image():
    distances = np.array([[5.5, -5.5, 0.0], [1.0, 2.0, -6.0]])

    np.testing.assert_allclose(to_mic(distances, box_length=10.0), [[-4.5, 4.5, 0.0], [1.0, 2.0, 4.0]])


def test_box_respects_minimum_distance():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 2, "Cl": 2},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=False,
        max_iter=20,
        max_tries=5,
    )

    assert coords.shape == (4, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 8.0)
    for first, second in itertools.combinations(coords, 2):
        assert np.linalg.norm(to_mic(first - second, 8.0)) > 1.0


def test_box_can_return_scaled_coordinates():
    np.random.seed(5)
    coords = construct_simulation_box(
        {"Na": 1, "Cl": 1},
        min_distance=1.0,
        box_length=8.0,
        scale_coords=True,
    )

    assert coords.shape == (2, 3)
    assert np.all(coords >= 0.0)
    assert np.all(coords <= 1.0)


def test_overfull_box_is_rejected():
    with pytest.raises(ValueError, match="Density is too high"):
        construct_simulation_box({"Na": 10}, min_distance=2.0, box_length=3.0)


def test_salt_builder_returns_periodic_neutral_atoms():
    random.seed(2)
    np.random.seed(2)

    atom_charges = {"Na": 1, "Cl": -1}
    target_num_atoms = 4
    atoms = atomic_system_builder(atom_charges, target_num_atoms=target_num_atoms, min_distance=0.8, box_length=8.0)
    molecule = MoleculesObject(atoms, "salt")

    assert atoms.pbc.all()
    np.testing.assert_allclose(atoms.cell.lengths(), [8.0, 8.0, 8.0])
    assert len(atoms) >= target_num_atoms - 1
    assert len(molecule) == len(atoms)
    assert set(atoms.get_chemical_symbols()).issubset(atom_charges)
    assert sum(atom_charges[symbol] for symbol in atoms.get_chemical_symbols()) == 0


def test_condensed_phase_builders_return_periodic_systems():

    library = (
        str(Path(__file__).resolve().parents[1] / "examples" / "simple_water_multi_builder" / "fragment_library") + "/"
    )
    settings = {
        "molecule_library_dir": library,
        "cell_range": [[6.0, 6.0]] * 3,
        "solute_molecule_options": [["water.xyz"]],
        "Rrange": [0.5, 0.5],
    }
    builder_config = {"solvent_molecules": {"water.xyz": 1.0}, "min_dist": 1.25, "max_patience": 50}
    random.seed(3)
    np.random.seed(3)

    single = simple_condensed_phase_builder_task.func("single-0", builder_config, **settings)
    multi = simple_multi_condensed_phase_builder_task.func(["multi-0", "multi-1"], builder_config, **settings)

    assert single.get_atoms().pbc.all()
    assert [system.get_moleculeid() for system in multi] == ["multi-0", "multi-1"]
    for system in multi:
        assert system.get_atoms().pbc.all()


def test_condensed_phase_builders_can_build_clusters(tmp_path):
    """``"periodic": false`` in the builder configuration gives a cluster that is stored, and read back, as one."""
    from alframework.ml_interfaces.fine_tuning import shard_atoms
    from alframework.tools.tools import build_input_dict, store_current_data

    library = str(Path(__file__).resolve().parents[1] / "examples" / "simple_water" / "fragment_library") + "/"
    builder_config = {"molecule_library_dir": library, "cell_range": [[6.0, 6.0]] * 3,
                      "solute_molecule_options": [["water.xyz"]], "Rrange": [0.9, 0.9],
                      "solvent_molecules": {"water.xyz": 1.0}, "min_dist": 1.25, "max_patience": 50,
                      "periodic": False}
    random.seed(4)
    np.random.seed(4)
    systems = []
    for task, ids in ((simple_condensed_phase_builder_task, {"moleculeid": "mol-0000-0000000000"}),
                      (simple_multi_condensed_phase_builder_task, {"moleculeids": ["mol-0000-0000000001"]})):
        built = task.func(**build_input_dict(task.func, [dict(ids, builder_config=builder_config), builder_config],
                                             raise_on_fail=True))
        systems.extend(built if isinstance(built, list) else [built])
    for system in systems:
        atoms = system.get_atoms()
        assert not atoms.pbc.any() and np.allclose(atoms.cell.lengths(), 6.0) and len(atoms) > 3
        distances = atoms.get_all_distances()
        molecule = np.arange(len(atoms)) // 3  # Waters, three atoms each, in order.
        assert distances[molecule[:, None] != molecule[None, :]].min() >= 1.25
        system.store_results({"energy": -1.0, "forces": np.zeros((len(atoms), 3))})
        system.set_converged_flag(True)

    shard = tmp_path / "data-0000.h5"
    store_current_data(str(shard), systems, {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]})
    with h5py.File(shard) as h:
        assert all("cell" not in h[name] for name in h)
    assert [atoms.pbc.any() for atoms in shard_atoms([shard])] == [False, False]


@pytest.fixture
def geometry_sources(tmp_path):
    atoms = Atoms("OHH", positions=[[0, 0, 0], [0.95, 0, 0], [0, 0.95, 0]], cell=np.eye(3) * 8, pbc=[True, False, True])
    atoms.set_array("topology_atom_ids", np.arange(3))
    atoms.calc = SinglePointCalculator(atoms, energy=-42.0, forces=np.ones((3, 3)))
    write(tmp_path / "source.xyz", atoms, format="extxyz")
    permutation = [0, 2, 1]
    with h5py.File(tmp_path / "source.h5", "w") as h:
        g = h.create_group("water")
        g["coordinates"] = [atoms.positions[permutation]]
        g["species"] = np.array(["O", "H", "H"], dtype="S")
        g["topology_atom_ids"] = [permutation]
        g["_id"] = np.array(["source-water"], dtype="S")
        g["energy"] = [-123.0]
        g["forces"] = np.ones((1, 3, 3))
    return tmp_path


def test_external_geometry_drops_old_labels(geometry_sources):
    config = dict(sources=["source.xyz", "source.h5"], selection_mode="sequential")
    items = external_geometry_builder_task.func(
        moleculeid="ignored",
        moleculeids=["new-0", "new-1"],
        builder_config=config,
        master_directory=str(geometry_sources),
    )
    assert [m.get_moleculeid() for m in items] == ["new-0", "new-1"]
    for item in items:
        assert item.get_results() == {} and item.converged is None
        assert item.atoms.calc is None and item.atoms.info == {}
        assert len(item.get_metadata()["geometry_source_sha256"]) == 64
    np.testing.assert_array_equal(items[0].atoms.pbc, [True, False, True])
    np.testing.assert_allclose(items[0].atoms.cell, np.eye(3) * 8)
    np.testing.assert_array_equal(items[1].atoms.arrays["topology_atom_ids"], [0, 2, 1])
    assert items[1].get_metadata()["geometry_source_id"] == "source-water"
    single = external_geometry_builder_task.func("new-1", builder_config=config, master_directory=str(geometry_sources))
    np.testing.assert_allclose(single.atoms.positions, items[1].atoms.positions)


def test_external_geometry_hash_selection_depends_only_on_id(geometry_sources):
    config = dict(sources=["source.xyz", "source.h5"], selection_seed=19)
    a = build_external_geometries(["a", "b", "c"], config, str(geometry_sources))
    b = build_external_geometries(["c", "a"], config, str(geometry_sources))
    for x, y in [(a[2], b[0]), (a[0], b[1])]:
        assert x.get_metadata() == y.get_metadata()
        np.testing.assert_array_equal(x.atoms.positions, y.atoms.positions)


@pytest.mark.parametrize(
    "ids,mode,error,reason",
    [
        (["new-2"], "sequential", IndexError, "exhausted"),
        (["new"], "sequential", ValueError, "trailing integer"),
        (["a", "a"], "deterministic_hash", ValueError, "Duplicate"),
        (["a"], "bad", ValueError, "selection_mode"),
    ],
    ids=["past_the_end", "id_without_number", "duplicate_ids", "unknown_mode"],
)
def test_external_geometry_rejects_invalid_selection(geometry_sources, ids, mode, error, reason):
    with pytest.raises(error, match=reason):
        build_external_geometries(
            ids, dict(sources=["source.xyz", "source.h5"], selection_mode=mode), str(geometry_sources)
        )


@pytest.mark.parametrize("bad_ids", [[0, 0, 2], [0.0, 1.0, 2.0]], ids=["duplicate_id", "float_ids"])
def test_external_geometry_rejects_invalid_topology(geometry_sources, bad_ids):
    with h5py.File(geometry_sources / "source.h5", "a") as h:
        del h["water/topology_atom_ids"]
        h["water/topology_atom_ids"] = [bad_ids]
    with pytest.raises(ValueError, match="integer permutation"):
        build_external_geometries(["a"], dict(sources=["source.h5"]), str(geometry_sources))


def test_plain_xyz_gets_no_invented_topology(tmp_path):
    (tmp_path / "plain.xyz").write_text("2\nplain molecule\nH 0 0 0\nH 0 0 .74\n")
    m = build_external_geometries(["a"], dict(sources=["plain.xyz"]), str(tmp_path))[0]
    assert "topology_atom_ids" not in m.atoms.arrays
    assert not m.atoms.pbc.any()


def test_external_geometry_rejects_empty_or_malformed_sources(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        geometry_manifest({})
    with h5py.File(tmp_path / "bad.h5", "w") as h:
        h["species"] = [1, 1]
        h["coordinates"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="frames, atoms, 3"):
        geometry_manifest(dict(sources=["bad.h5"]), str(tmp_path))


# ---- Reading replay sources once per worker ----


def write_frames(path, counts, offset=0.0):
    """Write a shard with a root group and nested groups holding ``counts`` frames of H2."""
    with h5py.File(path, "w") as handle:
        for name, frames in counts.items():
            group = handle if name == "/" else handle.require_group(name)
            group["coordinates"] = np.full((frames, 2, 3), offset) + [[0, 0, 0], [0.74, 0, 0]]
            group["species"] = np.array(["H", "H"], dtype="S")


@pytest.fixture
def fresh_caches(monkeypatch):
    from alframework.builders import external_geometry
    from alframework.tools import files

    monkeypatch.setattr(files, "_HASHES", {})
    monkeypatch.setattr(external_geometry, "_SOURCE_INDEX", {})


@pytest.fixture
def counted(monkeypatch, fresh_caches):
    """Count hashes, HDF5 opens and XYZ reads made by the replay code."""
    from alframework.builders import external_geometry
    from alframework.tools import files

    counts = SimpleNamespace(hashes=0, opens=0, reads=0)
    real_sha256, real_file, real_read = hashlib.sha256, h5py.File, external_geometry.read

    def sha256(*args):
        counts.hashes += 1
        return real_sha256(*args)

    def open_file(*args, **kwargs):
        counts.opens += 1
        return real_file(*args, **kwargs)

    def read(*args, **kwargs):
        counts.reads += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(files, "hashlib", SimpleNamespace(sha256=sha256))
    monkeypatch.setattr(external_geometry.h5py, "File", open_file)
    monkeypatch.setattr(external_geometry, "read", read)
    return counts


def test_geometry_manifest_lists_every_frame_in_order(tmp_path, fresh_caches):
    write(tmp_path / "frames.xyz", [Atoms("H2", positions=[[0, 0, 0], [0, 0, d]]) for d in (0.7, 0.8)])
    write_frames(tmp_path / "data.h5", {"/": 1, "b": 2, "a": 1, "a/c": 3, "a/empty": 0, "z": 0})
    manifest = geometry_manifest(dict(sources=["frames.xyz", "data.h5"]), str(tmp_path))
    xyz, h5 = str((tmp_path / "frames.xyz").resolve()), str((tmp_path / "data.h5").resolve())
    xyz_hash, h5_hash = (hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (xyz, h5))
    expected = ([(xyz, None, i, xyz_hash) for i in range(2)] + [(h5, "/", 0, h5_hash)]
                + [(h5, "a", 0, h5_hash)] + [(h5, "a/c", i, h5_hash) for i in range(3)]
                + [(h5, "b", i, h5_hash) for i in range(2)])
    assert len(manifest) == len(expected) and list(manifest) == expected
    assert manifest[-1] == expected[-1] and manifest[2:4] == expected[2:4]
    with pytest.raises(IndexError):
        manifest[len(expected)]


def test_cached_sha256_hashes_each_version_of_a_file_once(tmp_path, counted):
    from alframework.tools.files import cached_sha256

    path = tmp_path / "seed.h5"
    path.write_bytes(b"first")
    assert cached_sha256(path) == cached_sha256(path) == hashlib.sha256(b"first").hexdigest()
    assert counted.hashes == 1
    path.write_bytes(b"second, longer")
    assert cached_sha256(path) == hashlib.sha256(b"second, longer").hexdigest()
    assert counted.hashes == 2


def test_geometry_manifest_reads_each_source_once_per_worker(tmp_path, counted):
    write(tmp_path / "frames.xyz", Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]))
    write_frames(tmp_path / "data.h5", {"a": 2})
    config = dict(sources=["frames.xyz", "data.h5"])
    counted.hashes = counted.opens = counted.reads = 0
    first = list(geometry_manifest(config, str(tmp_path)))
    assert (counted.hashes, counted.opens, counted.reads) == (2, 1, 1)
    assert list(geometry_manifest(config, str(tmp_path))) == first
    assert (counted.hashes, counted.opens, counted.reads) == (2, 1, 1)


def test_geometry_manifest_notices_a_changed_source(tmp_path, fresh_caches):
    path = tmp_path / "data-0000.h5"
    write_frames(path, {"a": 2})
    config = dict(sources=[path.name])
    before = list(geometry_manifest(config, str(tmp_path)))

    write_frames(tmp_path / "new.h5", {"a": 3})
    os.replace(tmp_path / "new.h5", path)
    replaced = list(geometry_manifest(config, str(tmp_path)))
    assert len(replaced) == 3 and replaced[0][3] != before[0][3]

    # An edit in place that keeps the size still changes the modified time.
    data = bytearray(path.read_bytes())
    stat = path.stat()
    with h5py.File(path, "r+") as handle:
        handle["a/coordinates"][0, 0, 0] = 0.25
    assert path.stat().st_size == stat.st_size and path.read_bytes() != bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert list(geometry_manifest(config, str(tmp_path)))[0][3] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_geometry_opens_each_source_once_per_task(tmp_path, counted):
    write_frames(tmp_path / "data.h5", {"a": 3})
    config = dict(sources=["data.h5"], selection_mode="sequential")
    build_external_geometries(["m-0"], config, str(tmp_path))  # Indexes the source.
    counted.opens = 0
    molecules = build_external_geometries(["m-0", "m-1", "m-2"], config, str(tmp_path))
    assert counted.opens == 1
    assert [m.get_metadata()["geometry_frame"] for m in molecules] == [0, 1, 2]


def test_rdkit_builder_rejects_a_changed_seed_after_caching_it(prepared_seed, tmp_path, fresh_caches):
    copy = tmp_path / "prepared"
    shutil.copytree(prepared_seed.output, copy)
    builder = dict(json.loads((copy / "builder_config.json").read_text()),
                   seed_manifest_path=str(copy / "provenance.json"))
    seed.build_rdkit_molecules(["mol-boot-0000000000"], builder)
    with h5py.File(copy / "seed.h5", "a") as handle:
        handle.attrs["edited"] = 1
    with pytest.raises(ValueError, match="Prepared RDKit input changed: seed.h5"):
        seed.build_rdkit_molecules(["mol-boot-0000000001"], builder)


def seed_opens(monkeypatch, seed_path):
    """Record each time any code opens ``seed_path`` with h5py."""
    real_file, opened, target = h5py.File, [], Path(seed_path).resolve()

    def open_file(*args, **kwargs):
        opened.append(Path(args[0]).resolve() == target)
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", open_file)
    return opened


def test_rdkit_builder_opens_the_seed_twice_per_task(prepared_seed, monkeypatch, fresh_caches):
    builder = json.loads((prepared_seed.output / "builder_config.json").read_text())
    seed.build_rdkit_molecules(["mol-boot-0000000000"], builder)  # Indexes the seed.
    opened = seed_opens(monkeypatch, prepared_seed.output / "seed.h5")
    hashed = []
    monkeypatch.setattr(seed, "_sha256", lambda path: hashed.append(Path(path).name) or "digest")
    molecules = seed.build_rdkit_molecules([f"mol-boot-{i:010d}" for i in range(4)], builder)
    assert sum(opened) == 2  # Once for the frames and once for their metadata, however many molecules.
    assert hashed == ["provenance.json"]  # The seed and reference hashes are cached.
    assert len({m.get_metadata()["rdkit_manifest_sha256"] for m in molecules}) == 1


@pytest.fixture
def ring_pool(tmp_path):
    """A hand-made CPP ring preparation, as the CPP example's prepare.py writes it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cpp_custom_modules", ROOT / "examples/CPP_excited_gpu4pyscf/custom_modules.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_frames(prepared / "seed.h5", {"ring": 3})
    with h5py.File(prepared / "seed.h5", "a") as handle:
        group = handle["ring"]
        group["tier_index"] = [0, 1, 2]
        group["rotation_angles_degrees"] = np.zeros((3, 2))
        group["rattle_rms_angstrom"] = [0.0, 0.01, 0.02]
    write(prepared / "reference.xyz", Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    manifest = dict(schema=module.SCHEMA, status="completed", ring_size=6, artifacts_sha256={
        name: hashlib.sha256((prepared / name).read_bytes()).hexdigest() for name in ("seed.h5", "reference.xyz")})
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    return SimpleNamespace(module=module, prepared=prepared,
                           builder={"seed_manifest_path": str(prepared / "manifest.json")})


def test_cpp_ring_builder_replays_its_seed_and_opens_it_twice_per_task(ring_pool, monkeypatch, fresh_caches):
    ring_pool.module.build_ring_molecules(["mol-boot-0000000000"], ring_pool.builder)  # Indexes the seed.
    opened = seed_opens(monkeypatch, ring_pool.prepared / "seed.h5")
    hashed = []
    monkeypatch.setattr(ring_pool.module, "_sha256", lambda path: hashed.append(Path(path).name) or "digest")
    molecules = ring_pool.module.build_ring_molecules([f"mol-boot-{i:010d}" for i in range(3)], ring_pool.builder)
    assert [m.get_metadata()["ring_tier"] for m in molecules] == [0, 1, 2]
    assert sum(opened) == 2  # Once for the frames and once for their metadata, however many molecules.
    assert hashed == ["manifest.json"]  # The seed and reference hashes are cached.


def test_cpp_ring_builder_rejects_a_changed_reference_after_caching_it(ring_pool, fresh_caches):
    ring_pool.module.build_ring_molecules(["mol-boot-0000000000"], ring_pool.builder)
    write(ring_pool.prepared / "reference.xyz", Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]]))
    with pytest.raises(ValueError, match="Prepared ring input changed: reference.xyz"):
        ring_pool.module.build_ring_molecules(["mol-boot-0000000001"], ring_pool.builder)


# ---- Preparing molecular seeds with RDKit (an optional builder capability) ----

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/AcAc_excited_pyseqm"


def example_config(name):
    return json.loads((EXAMPLES / f"seed_{name}.json").read_text())


def test_rdkit_fps_selection_is_deterministic():
    assert seed.ladder_quotas(50000, 6) == [8334, 8334, 8333, 8333, 8333, 8333]
    assert seed.ladder_quotas(5000, 6) == [834, 834, 833, 833, 833, 833]
    assert [seed.stratum_seed(42, level, 0) for level in range(6)] == [
        438055385,
        1262664333,
        1985621771,
        1596702514,
        702239142,
        649246379,
    ]
    features = np.arange(4, dtype=float)[:, None]
    np.testing.assert_array_equal(seed.farthest_point_order(features, 4), [1, 3, 0, 2])
    np.testing.assert_array_equal(seed.farthest_point_order(np.zeros((4, 2)), 4), np.arange(4))
    coordinates = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    numbers = np.array([8, 1, 1])
    descriptor = seed.pair_descriptor(coordinates, numbers)
    np.testing.assert_allclose(descriptor, [[np.sqrt(5), 1.0, 2.0]])
    order = [2, 0, 1]
    np.testing.assert_array_equal(seed.pair_descriptor(coordinates[:, order] + 10.0, numbers[order]), descriptor)
    with pytest.raises(ValueError, match="FPS"):
        seed.farthest_point_order(features, 5)


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"seed_size": 50001}, "seed_size"),
        ({"threads": True}, "threads"),
        ({"bond_scales": [0.0, float("nan")]}, "bond_scales"),
        ({"angle_scales": [0.0]}, "same length"),
        ({"geometry_screen": {"typo": 1.0}}, "geometry_screen"),
    ],
    ids=["seed_larger_than_pool", "boolean_threads", "nan_scale", "mismatched_ladders", "unknown_screen_setting"],
)
def test_rdkit_rejects_invalid_seed_settings(override, reason):
    with pytest.raises(ValueError, match=reason):
        seed.resolve_config({**example_config("keto"), **override}, base_directory=EXAMPLES)


def test_rdkit_geometry_screen_reports_the_right_failure():
    context = {
        "numbers": np.array([1, 1]),
        "bonds": np.array([[0, 1]]),
        "bond_centers": np.array([1.0]),
        "pairs": np.array([[0, 1]]),
        "nonbonded": np.empty((0, 2), dtype=int),
    }
    screen = seed.DEFAULT_SETTINGS["geometry_screen"]
    for distance in (0.85, 1.0, 1.50):
        assert seed.geometry_failure([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], context, screen) is None
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [0.849, 0.0, 0.0]], context, screen) == "bond_window"
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [1.501, 0.0, 0.0]], context, screen) == "bond_window"
    assert seed.geometry_failure(np.full((2, 3), np.nan), context, screen) == "non_finite"
    loose = {**screen, "bond_min_scale": 0.1}
    assert seed.geometry_failure([[0.0, 0.0, 0.0], [0.69, 0.0, 0.0]], context, loose) == "min_distance"
    context.update(
        numbers=np.array([1, 1, 1]), pairs=np.array([[0, 1], [0, 2], [1, 2]]), nonbonded=np.array([[0, 2], [1, 2]])
    )
    assert (
        seed.geometry_failure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.1, 0.0]], context, screen)
        == "inferred_extra_bond"
    )
    assert (
        seed.geometry_failure([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.82, 0.0]], context, screen)
        == "nonbonded_close"
    )


def test_rdkit_polar_hydrogen_bounds_are_wider(tmp_path):
    pytest.importorskip("rdkit")
    reference = tmp_path / "water.xyz"
    reference.write_text("3\nwater\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\n")
    config = seed.resolve_config({"smiles": "O", "reference_geometry": str(reference)})
    context = seed._reference_context(config)
    polar = seed.widened_bounds_matrix(context["molecule"], context["centers"], config, 5)
    ordinary = seed.widened_bounds_matrix(context["molecule"], context["centers"], {**config, "polar_h": False}, 5)
    center = context["centers"][0, 1]
    assert polar[0, 1] == pytest.approx(center * 1.1)
    assert polar[1, 0] == pytest.approx(center * 0.9)
    assert polar[0, 1] > ordinary[0, 1]
    assert polar[1, 0] < ordinary[1, 0]


def test_rdkit_rejects_same_elements_with_different_topology():
    pytest.importorskip("rdkit")
    config = {**example_config("keto"), "reference_geometry": "enol.xyz"}
    with pytest.raises(ValueError, match="connectivity|bond set"):
        seed._reference_context(seed.resolve_config(config, base_directory=EXAMPLES))


@pytest.fixture(scope="module", params=["enol", "keto"])
def prepared_seed(request, tmp_path_factory):
    """A small prepared seed, generated twice as two separate runs would."""
    pytest.importorskip("rdkit")
    config = {**example_config(request.param), "pool_size": 24, "seed_size": 12}
    directory = tmp_path_factory.mktemp(request.param)
    output, repeated = directory / "first", directory / "second"
    report = seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    seed.prepare_rdkit_seed(config, repeated, config_directory=EXAMPLES)
    return SimpleNamespace(report=report, output=output, repeated=repeated)


def test_rdkit_seed_preparation_records_provenance(prepared_seed):
    report = prepared_seed.report
    assert report["pool_count"] == 24 and report["selected_count"] == 12
    assert report["embedding_settings"]["useBasicKnowledge"] is True
    assert report["embedding_settings"]["useExpTorsionAnglePrefs"] is False
    assert report["embedding_settings"]["force_field_optimization"] is False
    for file, digest in report["artifacts_sha256"].items():
        assert hashlib.sha256((prepared_seed.output / file).read_bytes()).hexdigest() == digest


def test_rdkit_seed_preparation_is_reproducible(prepared_seed):
    with h5py.File(prepared_seed.output / "seed.h5") as first, h5py.File(prepared_seed.repeated / "seed.h5") as second:
        for key in first["molecule"]:
            np.testing.assert_array_equal(first["molecule"][key][()], second["molecule"][key][()])


def test_rdkit_seed_is_a_stratified_sample_of_the_pool(prepared_seed):
    with (
        h5py.File(prepared_seed.output / "pool.h5") as pool_file,
        h5py.File(prepared_seed.output / "seed.h5") as seed_file,
    ):
        pool, selected = pool_file["molecule"], seed_file["molecule"]
        indices = selected["pool_index"][()]
        assert len(np.unique(indices)) == 12
        species = selected["species"][()].astype("U2")
        assert species.ndim == 1
        assert np.all(np.diff([atomic_numbers[value] for value in species]) >= 0)
        np.testing.assert_array_equal(np.bincount(selected["strata"][()][:, 0]), [2] * 6)
        np.testing.assert_array_equal(selected["fps_rank"][()], [0, 1] * 6)
        np.testing.assert_array_equal(selected["coordinates"][()], pool["coordinates"][()][indices])
        np.testing.assert_array_equal(selected["strata"][()], pool["strata"][()][indices])
        np.testing.assert_array_equal(selected["fps_rank"][()], pool["fps_rank"][()][indices])
        assert not any(re.fullmatch(r"E\d+|F\d+|dE\d.*", key) for key in selected)


def test_rdkit_builder_replays_prepared_geometries(prepared_seed):
    with h5py.File(prepared_seed.output / "seed.h5") as handle:
        selected = handle["molecule"]
        canonical = selected["coordinates"][()][:, np.argsort(selected["topology_atom_ids"][()])]
    builder = json.loads((prepared_seed.output / "builder_config.json").read_text())
    molecules = seed.build_rdkit_molecules([f"mol-boot-{i:010d}" for i in range(12)], builder)
    assert len(molecules) == 12
    for index, molecule in enumerate(molecules):
        assert molecule.get_results() == {}
        order = np.argsort(molecule.atoms.arrays["topology_atom_ids"])
        np.testing.assert_array_equal(molecule.atoms.positions[order], canonical[index])
        assert molecule.get_metadata()["rdkit_fps_rank"] == index % 2
    with pytest.raises(IndexError, match="exhausted"):
        seed.build_rdkit_molecules(["mol-boot-0000000012"], builder)


def test_rdkit_runtime_replay_strips_labels(prepared_seed, tmp_path):
    builder = json.loads((prepared_seed.output / "builder_config.json").read_text())
    labels = tmp_path / "labels"
    labels.mkdir()
    shutil.copyfile(prepared_seed.output / "seed.h5", labels / "data-0000.h5")
    with h5py.File(labels / "data-0000.h5", "a") as handle:
        handle["molecule"]["energy"] = np.ones(12)
    replay = seed.build_rdkit_molecules(["runtime-1"], builder, str(labels / "data-{:04d}.h5"), 1)
    assert replay[0].get_results() == {} and not replay[0].get_metadata()["replay_external_bootstrap"]


def test_failed_rdkit_preparation_writes_nothing(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    from rdkit.Chem import AllChem

    config = {**example_config("keto"), "pool_size": 6, "seed_size": 6}
    output = tmp_path / "failed"
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="quota"):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert not output.exists()
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert marker.read_text() == "unchanged"


def test_rdkit_command_line_finds_reference_next_to_config(tmp_path):
    pytest.importorskip("rdkit")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {**example_config("enol"), "pool_size": 12, "seed_size": 6}
    (config_dir / "enol.xyz").write_bytes((EXAMPLES / "enol.xyz").read_bytes())
    path = config_dir / "seed.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alframework.builders.rdkit_molecule",
            "--config",
            str(path),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "provenance.json").read_text())
    assert report["selected_count"] == 6
    assert report["settings"]["reference_geometry"] == str(config_dir / "enol.xyz")
