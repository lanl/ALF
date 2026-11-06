import builtins
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.tools import molecular_topology as topology


def _reference_topology(**overrides):
    values = {
        "reference_path": "/tmp/reference.mol",
        "reference_sha256": "abc123",
        "reference_format": "mol",
        "reference_charge": 0,
        "atomic_numbers": (6, 6, 6),
        "bonds": ((0, 1), (1, 2)),
        "bond_lengths": (1.5, 1.5),
        "bond_min_scale": 0.70,
        "bond_max_scale": 1.35,
        "connectivity_scale": 1.25,
    }
    values.update(overrides)
    return topology.ReferenceTopology(**values)


def _chain_atoms(positions=None, symbols="CCC"):
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
    atoms = Atoms(symbols, positions=positions)
    atoms.set_array('topology_atom_ids', np.arange(len(atoms)))
    return atoms


def _topology_config(path, reference_format="auto", **overrides):
    config = {
        "enabled": True,
        "reference_conformer_path": str(path),
        "reference_format": reference_format,
        "reference_charge": 0,
        "bond_min_scale": 0.70,
        "bond_max_scale": 1.35,
        "connectivity_scale": 1.25,
    }
    config.update(overrides)
    return {"topology_check": config}


def _write_water_xyz(path: Path, first_bond=0.9572):
    path.write_text(
        "3\nwater\n"
        "O 0.0 0.0 0.0\n"
        f"H {first_bond} 0.0 0.0\n"
        "H -0.239 0.927 0.0\n",
        encoding="utf-8",
    )


def _write_rdkit_references(mol_path: Path, sdf_path: Path):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    molecule = Chem.RWMol()
    for atomic_number in (6, 6, 6):
        molecule.AddAtom(Chem.Atom(atomic_number))
    molecule.AddBond(0, 1, Chem.BondType.SINGLE)
    molecule.AddBond(1, 2, Chem.BondType.SINGLE)
    molecule = molecule.GetMol()
    conformer = Chem.Conformer(3)
    for index, point in enumerate(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0))
    ):
        conformer.SetAtomPosition(index, point)
    molecule.AddConformer(conformer)
    Chem.MolToMolFile(molecule, str(mol_path))
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(molecule)
    writer.close()


@pytest.mark.parametrize(
    ("filename", "reference_format", "atomic_numbers"),
    [
        ("reference.xyz", "xyz", (8, 1, 1)),
        ("reference.mol", "mol", (6, 6, 6)),
        ("reference.sdf", "sdf", (6, 6, 6)),
    ],
ids=['xyz', 'mol', 'sdf'])
def test_rdkit_loads_topology_reference_from_each_format(
    tmp_path,
    filename,
    reference_format,
    atomic_numbers,
):
    pytest.importorskip("rdkit")
    xyz_path = tmp_path / "reference.xyz"
    mol_path = tmp_path / "reference.mol"
    sdf_path = tmp_path / "reference.sdf"
    _write_water_xyz(xyz_path)
    _write_rdkit_references(mol_path, sdf_path)

    loaded = topology.load_reference_topology(
        _topology_config(tmp_path / filename, reference_format)
    )

    assert loaded.reference_format == reference_format
    assert loaded.reference_sha256
    assert loaded.bonds
    assert loaded.atomic_numbers == atomic_numbers


def test_changed_topology_reference_is_noticed(tmp_path):
    pytest.importorskip("rdkit")
    reference = tmp_path / "reference.xyz"
    _write_water_xyz(reference, first_bond=0.9572)
    config = _topology_config("reference.xyz")

    initial = topology.load_reference_topology(
        config,
        master_directory=str(tmp_path),
    )
    _write_water_xyz(reference, first_bond=1.01)
    refreshed = topology.load_reference_topology(
        config,
        master_directory=str(tmp_path),
    )

    assert initial.reference_path == str(reference.resolve())
    assert initial.reference_sha256 != refreshed.reference_sha256
    assert initial.bond_lengths != refreshed.bond_lengths


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bond_min_scale": 0.0}, "bond scales"),
        ({"bond_min_scale": 1.01}, "bond scales"),
        ({"bond_max_scale": 0.99}, "bond scales"),
        ({"connectivity_scale": 0.0}, "connectivity_scale"),
        ({"connectivity_scale": np.inf}, "finite"),
        ({"reference_charge": 0.5}, "integer"),
        ({"unexpected": 1}, "Unknown"),
    ],
ids=['zero_bond_min', 'bond_min_above_one', 'bond_max_below_one', 'zero_connectivity', 'infinite_connectivity', 'fractional_charge', 'unknown_key'])
def test_invalid_topology_settings_are_rejected(tmp_path, overrides, message):
    reference = tmp_path / "reference.xyz"
    _write_water_xyz(reference)

    with pytest.raises((ValueError, TypeError), match=message):
        topology.load_reference_topology(
            _topology_config(reference, **overrides)
        )


def test_missing_rdkit_gives_a_clear_error(tmp_path, monkeypatch):
    reference = tmp_path / "reference.xyz"
    _write_water_xyz(reference)
    original_import = builtins.__import__

    def missing_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("mocked missing RDKit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_rdkit)
    with pytest.raises(ImportError, match="topology extra"):
        topology.load_reference_topology(_topology_config(reference))


def test_topology_reference_with_extra_bonds_is_rejected(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    molecule = Chem.RWMol()
    for _ in range(3):
        molecule.AddAtom(Chem.Atom(6))
    molecule.AddBond(0, 1, Chem.BondType.SINGLE)
    molecule.AddBond(1, 2, Chem.BondType.SINGLE)
    molecule = molecule.GetMol()
    conformer = Chem.Conformer(3)
    for index, point in enumerate(
        ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (1.4, 0.0, 0.0))
    ):
        conformer.SetAtomPosition(index, point)
    molecule.AddConformer(conformer)
    path = tmp_path / "compressed.mol"
    Chem.MolToMolFile(molecule, str(path))

    with pytest.raises(ValueError, match="inconsistent with connectivity_scale"):
        topology.load_reference_topology(_topology_config(path))


def test_malformed_reference_is_rejected(tmp_path):
    pytest.importorskip("rdkit")
    path = tmp_path / "malformed.xyz"
    path.write_text("this is not an XYZ conformer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not parse"):
        topology.load_reference_topology(_topology_config(path))


def test_topology_accepts_bonds_at_the_limits():
    result = topology.validate_fixed_topology(
        _chain_atoms(
            [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [3.075, 0.0, 0.0]]
        ),
        _reference_topology(),
    )

    assert result.valid
    assert result.metrics["topology_min_bond_ratio"] == pytest.approx(0.70)
    assert result.metrics["topology_max_bond_ratio"] == pytest.approx(1.35)


@pytest.mark.parametrize(
    "positions",
    [
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.4, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.6, 0.0, 0.0]],
    ],
ids=['compressed', 'stretched'])
def test_topology_rejects_compressed_or_stretched_bonds(positions):
    result = topology.validate_fixed_topology(
        _chain_atoms(positions),
        _reference_topology(),
    )

    assert not result.valid
    assert result.reason == "topology_bond_window"


def test_topology_rejects_a_new_bond():
    height = np.sqrt(1.5**2 - 0.75**2)
    result = topology.validate_fixed_topology(
        _chain_atoms(
            [[0.0, 0.0, 0.0], [0.75, height, 0.0], [1.5, 0.0, 0.0]]
        ),
        _reference_topology(),
    )

    assert not result.valid
    assert result.reason == "topology_extra_bond"


def test_topology_rejects_wrong_element_or_nan_coordinates():
    identity = topology.validate_fixed_topology(
        _chain_atoms(symbols="COC"),
        _reference_topology(),
    )
    nonfinite = topology.validate_fixed_topology(
        _chain_atoms(
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        _reference_topology(),
    )

    assert identity.reason == "topology_identity"
    assert nonfinite.reason == "topology_nonfinite"


def test_topology_needs_ids_for_reordered_atoms():
    reference = _reference_topology()
    atoms = _chain_atoms()
    permuted = atoms[[2, 0, 1]]
    before = permuted.positions.copy()
    assert topology.validate_fixed_topology(permuted, reference).valid
    np.testing.assert_array_equal(permuted.positions, before)
    del permuted.arrays['topology_atom_ids']
    assert topology.validate_fixed_topology(permuted, reference).reason == 'topology_identity'
    atoms.pbc = True
    assert not topology.validate_fixed_topology(atoms, reference).valid


def test_opened_ring_is_rejected():
    reference = _reference_topology(atomic_numbers=(6,)*4,
        bonds=((0,1),(0,3),(1,2),(2,3)),bond_lengths=(1.5,)*4)
    atoms=Atoms('CCCC',positions=[[0,0,0],[1.5,0,0],[1.5,1.5,0],[0,1.5,0]])
    atoms.set_array('topology_atom_ids',np.arange(4))
    assert topology.validate_fixed_topology(atoms, reference).valid
    atoms.positions[3] = [-3,3,0]
    result=topology.validate_fixed_topology(atoms, reference)
    assert not result.valid and result.reason=='topology_bond_window'
