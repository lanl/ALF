"""Read surface inputs, validate molecular geometry, and check known net charges."""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from alframework.tools.files import file_key


_CHARGE_TOLERANCE = 1e-6
_BOND_FORMATS = {'.mol', '.mdl', '.sdf'}


def source_key(filename, master_directory):
    if not isinstance(filename, str) or not filename:
        raise ValueError('Expected a local geometry filename')
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(master_directory or '.') / path
    return file_key(path)


@dataclass(frozen=True)
class Geometry:
    atoms: Atoms
    charges: tuple = ()


@dataclass(frozen=True)
class AdsorbateInput:
    source_kind: str
    source: str
    atoms: Atoms
    anchor_count: int
    charge_sources: tuple
    force_field: str | None = None


def _validate_atoms(atoms, label):
    if not len(atoms) or not np.isfinite(atoms.positions).all() or (atoms.numbers <= 0).any():
        raise ValueError(f'{label}: expected nonempty, finite atomic geometry')


@lru_cache(maxsize=32)
def read_geometry(source):
    filename = source[0]
    try:
        images = read(filename, index=':', parallel=False)
    except Exception as exc:
        raise ValueError(f'{filename}: could not read geometry: {exc}') from exc
    if len(images) != 1:
        raise ValueError(f'{filename}: expected exactly one structure')
    atoms = images[0]
    _validate_atoms(atoms, filename)
    for occupancies in atoms.info.get('occupancy', {}).values():
        if len(occupancies) != 1 or not np.isclose(next(iter(occupancies.values())), 1.0):
            raise ValueError(f'{filename}: partial occupancies require an ordered input geometry')

    charges = [(f'file:{name}', atoms.info[name]) for name in ('charge', 'total_charge') if name in atoms.info]
    if atoms.has('initial_charges'):
        initial = atoms.get_initial_charges()
        if not np.isfinite(initial).all():
            raise ValueError(f'{filename}: initial_charges must be finite')
        charges.append(('file:initial_charges', initial.sum()))
    # Carry only geometry to the builder, with selected charge declarations alongside it.
    clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    return Geometry(clean, tuple(charges))


def check_neutral(charges, config, key, label):
    """Check all available charge declarations, returning their provenance."""
    if key in config:
        charges = (*charges, (f'config:{key}', config[key]))
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
           for _, value in charges):
        raise ValueError(f'{label}: charge declarations must be finite numbers')
    values = [value for _, value in charges]
    if values and max(values) - min(values) > _CHARGE_TOLERANCE:
        raise ValueError(f'{label}: conflicting charge declarations: {dict(charges)}')
    if any(abs(value) > _CHARGE_TOLERANCE for value in values):
        raise ValueError(f'{label}: nonzero net charge is not supported: {dict(charges)}')
    return tuple(name for name, _ in charges)


def _molecule_distances(atoms, label):
    _validate_atoms(atoms, label)
    if atoms.pbc.any():
        raise ValueError(f'{label}: adsorbate must be a nonperiodic molecule')
    radii = covalent_radii[atoms.numbers]
    contact = radii[:, None] + radii
    distances = atoms.get_all_distances()
    if np.triu(distances < 0.65 * contact, k=1).any():
        raise ValueError(f'{label}: adsorbate contains overlapping atoms')
    return distances, contact


@lru_cache(maxsize=32)
def _coordinate_molecule(source):
    geometry = read_geometry(source)
    distances, contact = _molecule_distances(geometry.atoms, source[0])
    bonds = csr_matrix(np.triu(distances <= 1.25 * contact, k=1))
    if connected_components(bonds, directed=False, return_labels=False) != 1:
        raise ValueError(f'{source[0]}: adsorbate has disconnected fragments')
    return geometry


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("SMILES and MOL/SDF adsorbates need RDKit. Install ALF with 'pip install .[topology]'.") from exc
    return Chem, AllChem


def _check_bond_graph(molecule, chem, label):
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f'{label}: could not parse a nonempty molecule')
    if len(chem.GetMolFrags(molecule)) != 1:
        raise ValueError(f'{label}: adsorbate has disconnected fragments')


def _rdkit_geometry(molecule, chem, label):
    if molecule.GetNumConformers() != 1:
        raise ValueError(f'{label}: expected exactly one molecular geometry')
    atoms = Atoms(numbers=[atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                  positions=molecule.GetConformer().GetPositions())
    distances, contact = _molecule_distances(atoms, label)
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if distances[i, j] > 1.5 * contact[i, j]:
            raise ValueError(f'{label}: bond {i}–{j} is too long for an intact adsorbate')
    return Geometry(atoms, (('rdkit:formal_charge', int(chem.GetFormalCharge(molecule))),))


@lru_cache(maxsize=32)
def _bond_file_molecule(source):
    filename = source[0]
    chem, _ = _rdkit()
    try:
        if Path(filename).suffix.lower() == '.sdf':
            supplier = chem.SDMolSupplier(filename, removeHs=False)
            if len(supplier) != 1:
                raise ValueError('expected exactly one structure')
            molecule = supplier[0]
        else:
            molecule = chem.MolFromMolFile(filename, removeHs=False)
    except Exception as exc:
        raise ValueError(f'{filename}: could not read molecular geometry: {exc}') from exc
    _check_bond_graph(molecule, chem, filename)
    geometry = _rdkit_geometry(molecule, chem, filename)
    try:
        declarations = tuple((f'file:{name}', float(molecule.GetProp(name)))
                             for name in ('charge', 'total_charge') if molecule.HasProp(name))
    except ValueError as exc:
        raise ValueError(f'{filename}: charge declarations must be finite numbers') from exc
    return Geometry(geometry.atoms, (*geometry.charges, *declarations))


@lru_cache(maxsize=32)
def _smiles_molecule(smiles, seed, relax):
    chem, all_chem = _rdkit()
    label = f'SMILES {smiles!r}'
    options = chem.SmilesParserParams()
    options.removeHs = False
    try:
        molecule = chem.MolFromSmiles(smiles, options)
    except Exception as exc:
        raise ValueError(f'{label}: could not parse molecule: {exc}') from exc
    _check_bond_graph(molecule, chem, label)
    # Reject charged species before embedding, including isolated ions.
    check_neutral((('rdkit:formal_charge', chem.GetFormalCharge(molecule)),), {}, 'charge', label)
    anchor_count = molecule.GetNumAtoms()
    try:
        molecule = chem.AddHs(molecule)
        parameters = all_chem.ETKDGv3()
        digest = hashlib.sha256(f'{seed}|{smiles}'.encode()).digest()
        parameters.randomSeed = int.from_bytes(digest[:4], 'big') % (2**31 - 1)
        parameters.numThreads = 1
        result = all_chem.EmbedMolecule(molecule, parameters)
    except Exception as exc:
        raise ValueError(f'{label}: RDKit could not generate a conformer: {exc}') from exc
    if result < 0:
        raise ValueError(f'{label}: RDKit could not generate a conformer')
    force_field = _relax_conformer(molecule, all_chem, label) if relax else None
    return _rdkit_geometry(molecule, chem, label), anchor_count, force_field


def _relax_conformer(molecule, all_chem, label):
    if all_chem.MMFFHasAllMoleculeParams(molecule):
        name, optimize = 'MMFF94', all_chem.MMFFOptimizeMolecule
    elif all_chem.UFFHasAllMoleculeParams(molecule):
        name, optimize = 'UFF', all_chem.UFFOptimizeMolecule
    else:
        raise ValueError(f'{label}: neither MMFF94 nor UFF has parameters for this conformer')
    try:
        status = optimize(molecule, maxIters=500)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'{label}: {name} optimization failed: {exc}') from exc
    if status != 0:
        raise ValueError(f'{label}: {name} optimization did not converge (status {status})')
    return name


def load_adsorbate(entry, master_directory, seed):
    if ('file' in entry) == ('smiles' in entry):
        raise ValueError('Each adsorbate must specify exactly one of file and smiles')
    relax = entry.get('relax_smiles', False)
    if type(relax) is not bool or (relax and 'smiles' not in entry):
        raise ValueError('relax_smiles must be Boolean and can only enable optimization for a SMILES adsorbate')
    force_field = None
    if 'smiles' in entry:
        smiles = entry['smiles']
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError('adsorbate smiles must be a nonempty string')
        geometry, anchor_count, force_field = _smiles_molecule(smiles, seed, relax)
        label = f'SMILES {smiles!r}'
        kind, name = 'smiles', smiles
    else:
        source = source_key(entry['file'], master_directory)
        label = source[0]
        reader = _bond_file_molecule if Path(label).suffix.lower() in _BOND_FORMATS else _coordinate_molecule
        geometry = reader(source)
        anchor_count = len(geometry.atoms)
        kind, name = 'file', label
    sources = check_neutral(geometry.charges, entry, 'charge', label)
    return AdsorbateInput(kind, name, geometry.atoms, anchor_count, sources, force_field)
