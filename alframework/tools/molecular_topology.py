"""Opt-in fixed-connectivity validation for molecular sampling.

RDKit is used only to read one required reference conformer and determine its
bond graph. Runtime validation uses NumPy distances against that cached graph.
Validation requires reference atom order. Topology-aware replay and sampler
extensions normalize explicit IDs using tools.atom_order before validation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import covalent_radii


_SUPPORTED_REFERENCE_FORMATS = frozenset({"xyz", "mol", "mdl", "sdf"})
_TOPOLOGY_KEYS = frozenset(
    {
        "enabled",
        "reference_conformer_path",
        "reference_format",
        "reference_charge",
        "bond_min_scale",
        "bond_max_scale",
        "connectivity_scale",
    }
)


@dataclass(frozen=True)
class ReferenceTopology:
    """One immutable reference connectivity in its required atom order."""

    reference_path: str
    reference_sha256: str
    reference_format: str
    reference_charge: int
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_lengths: tuple[float, ...]
    bond_min_scale: float
    bond_max_scale: float
    connectivity_scale: float

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def bond_set(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.bonds)


@dataclass(frozen=True)
class TopologyValidationResult:
    """Result and diagnostics from validating one geometry."""

    valid: bool
    reason: str | None
    metrics: dict[str, Any]
    violations: tuple[dict[str, Any], ...]


def topology_enabled(sampler_config: dict[str, Any] | None) -> bool:
    """Return whether fixed-topology checks are configured."""

    raw = dict(sampler_config or {}).get("topology_check")
    if raw is None:
        return False
    if not isinstance(raw, dict):
        raise TypeError("topology_check must be a dictionary when provided.")
    return bool(raw.get("enabled", False))


def _required_finite_float(
    settings: dict[str, Any],
    key: str,
) -> float:
    if key not in settings:
        raise ValueError(
            f"topology_check.{key} is required when topology checking is enabled."
        )
    try:
        value = float(settings[key])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"topology_check.{key} must be a finite number."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"topology_check.{key} must be a finite number.")
    return value


def _resolve_reference_path(
    raw_path: Any,
    *,
    master_directory: str | None,
) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError(
            "topology_check.reference_conformer_path is required when "
            "topology checking is enabled."
        )
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        base = (
            Path(master_directory).expanduser()
            if master_directory
            else Path.cwd()
        )
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Topology reference conformer does not exist: {path}"
        )
    return path


def _load_rdkit_reference(
    path: Path,
    reference_format: str,
    reference_charge: int,
):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Topology checking requires ALF's topology extra (RDKit)."
        ) from exc

    selected_format = str(reference_format).strip().lower()
    if selected_format == "auto":
        selected_format = path.suffix.lower().lstrip(".")
    if selected_format not in _SUPPORTED_REFERENCE_FORMATS:
        raise ValueError(
            "topology_check.reference_format must resolve to XYZ, MOL, MDL, "
            f"or SDF; received {selected_format!r}."
        )

    try:
        if selected_format == "xyz":
            molecule = Chem.MolFromXYZBlock(path.read_text(encoding="utf-8"))
            if molecule is not None:
                rdDetermineBonds.DetermineBonds(
                    molecule,
                    charge=int(reference_charge),
                )
        elif selected_format in {"mol", "mdl"}:
            molecule = Chem.MolFromMolFile(
                str(path),
                removeHs=False,
                sanitize=True,
            )
        else:
            supplier = Chem.SDMolSupplier(
                str(path),
                removeHs=False,
                sanitize=True,
            )
            molecule = next(
                (item for item in supplier if item is not None),
                None,
            )
    except Exception as exc:
        raise ValueError(
            f"RDKit could not determine topology from reference {path}: {exc}"
        ) from exc
    if molecule is None:
        raise ValueError(
            f"RDKit could not parse topology reference conformer: {path}"
        )
    if molecule.GetNumConformers() < 1:
        raise ValueError(
            f"Topology reference conformer has no coordinates: {path}"
        )
    return molecule, selected_format


@lru_cache(maxsize=32)
def _load_reference_cached(
    reference_path: str,
    reference_sha256: str,
    reference_format: str,
    reference_charge: int,
    bond_min_scale: float,
    bond_max_scale: float,
    connectivity_scale: float,
) -> ReferenceTopology:
    path = Path(reference_path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != reference_sha256:
        raise RuntimeError(
            "Topology reference changed while it was being loaded; retry "
            f"after the file is stable: {path}"
        )
    molecule, selected_format = _load_rdkit_reference(
        path,
        reference_format,
        reference_charge,
    )
    positions = np.asarray(
        molecule.GetConformer().GetPositions(),
        dtype=np.float64,
    )
    atomic_numbers = tuple(
        int(atom.GetAtomicNum()) for atom in molecule.GetAtoms()
    )
    if len(atomic_numbers) < 2:
        raise ValueError("Topology reference must contain at least two atoms.")
    if any(number <= 0 for number in atomic_numbers):
        raise ValueError(
            "Topology reference atoms must have positive atomic numbers."
        )
    if positions.shape != (len(atomic_numbers), 3):
        raise ValueError(
            "Topology reference coordinates must have shape (n_atoms, 3)."
        )
    if not np.isfinite(positions).all():
        raise ValueError("Topology reference coordinates must be finite.")

    bonds = tuple(
        sorted(
            (
                min(
                    int(bond.GetBeginAtomIdx()),
                    int(bond.GetEndAtomIdx()),
                ),
                max(
                    int(bond.GetBeginAtomIdx()),
                    int(bond.GetEndAtomIdx()),
                ),
            )
            for bond in molecule.GetBonds()
        )
    )
    if not bonds:
        raise ValueError(
            "RDKit did not identify any bonds in the topology reference."
        )
    bond_lengths = tuple(
        float(np.linalg.norm(positions[first] - positions[second]))
        for first, second in bonds
    )
    if not np.isfinite(bond_lengths).all() or min(bond_lengths) <= 0.0:
        raise ValueError(
            "Topology reference bond lengths must be finite and positive."
        )

    bond_set = frozenset(bonds)
    for first, second in combinations(range(len(atomic_numbers)), 2):
        if (first, second) in bond_set:
            continue
        radius_sum = float(
            covalent_radii[atomic_numbers[first]]
            + covalent_radii[atomic_numbers[second]]
        )
        distance = float(np.linalg.norm(positions[first] - positions[second]))
        if distance <= connectivity_scale * radius_sum:
            raise ValueError(
                "Topology reference is inconsistent with connectivity_scale: "
                f"non-reference atom pair ({first}, {second}) has distance "
                f"{distance:.6g} A, at or below the new-bond threshold "
                f"{connectivity_scale * radius_sum:.6g} A."
            )

    return ReferenceTopology(
        reference_path=str(path),
        reference_sha256=file_hash,
        reference_format=selected_format,
        reference_charge=int(reference_charge),
        atomic_numbers=atomic_numbers,
        bonds=bonds,
        bond_lengths=bond_lengths,
        bond_min_scale=float(bond_min_scale),
        bond_max_scale=float(bond_max_scale),
        connectivity_scale=float(connectivity_scale),
    )


def load_reference_topology(
    sampler_config: dict[str, Any] | None,
    *,
    master_directory: str | None = None,
) -> ReferenceTopology | None:
    """Load and validate the configured reference topology."""

    raw = dict(sampler_config or {}).get("topology_check")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("topology_check must be a dictionary when provided.")
    unknown = sorted(set(raw) - _TOPOLOGY_KEYS)
    if unknown:
        raise ValueError(
            "Unknown topology_check options: " + ", ".join(unknown)
        )
    if not bool(raw.get("enabled", False)):
        return None

    bond_min_scale = _required_finite_float(raw, "bond_min_scale")
    bond_max_scale = _required_finite_float(raw, "bond_max_scale")
    connectivity_scale = _required_finite_float(raw, "connectivity_scale")
    if not (0.0 < bond_min_scale <= 1.0 <= bond_max_scale):
        raise ValueError(
            "topology_check bond scales must satisfy "
            "0 < bond_min_scale <= 1 <= bond_max_scale."
        )
    if connectivity_scale <= 0.0:
        raise ValueError(
            "topology_check.connectivity_scale must be positive."
        )
    path = _resolve_reference_path(
        raw.get("reference_conformer_path"),
        master_directory=master_directory,
    )
    raw_charge = raw.get("reference_charge", 0)
    try:
        reference_charge = int(raw_charge)
        numeric_charge = float(raw_charge)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "topology_check.reference_charge must be an integer."
        ) from exc
    if (
        not math.isfinite(numeric_charge)
        or numeric_charge != float(reference_charge)
    ):
        raise ValueError(
            "topology_check.reference_charge must be an integer."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _load_reference_cached(
        str(path),
        digest,
        str(raw.get("reference_format", "auto")),
        reference_charge,
        bond_min_scale,
        bond_max_scale,
        connectivity_scale,
    )


@lru_cache(maxsize=16)
def _topology_arrays(topology: ReferenceTopology):
    """Index and reference arrays that depend only on the reference, for validate_fixed_topology:
    bonds in their stored order, then every other atom pair in ``combinations`` order with its covalent
    radius sum."""
    bonds = np.asarray(topology.bonds, dtype=np.int64).reshape(-1, 2)
    references = np.asarray(topology.bond_lengths, dtype=np.float64)
    pair_first, pair_second = np.triu_indices(topology.n_atoms, 1)
    bonded = np.zeros((topology.n_atoms, topology.n_atoms), dtype=bool)
    bonded[bonds[:, 0], bonds[:, 1]] = True  # a stored (first, second) pair, as bond_set holds it
    keep = ~bonded[pair_first, pair_second]
    pair_first, pair_second = pair_first[keep], pair_second[keep]
    radii = np.asarray([covalent_radii[number] for number in topology.atomic_numbers], dtype=np.float64)
    arrays = (bonds[:, 0], bonds[:, 1], references, pair_first, pair_second,
              radii[pair_first] + radii[pair_second])
    for array in arrays:
        array.flags.writeable = False
    return arrays


def validate_fixed_topology(
    atoms,
    topology: ReferenceTopology | None,
) -> TopologyValidationResult:
    """Validate exact atom identity, bond windows, and fixed connectivity."""

    if topology is None:
        return TopologyValidationResult(True, None, {}, ())
    from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, validated_atom_ids
    try:
        if np.any(atoms.pbc):
            raise ValueError('Chemical topology screening requires nonperiodic molecules')
        if TOPOLOGY_ATOM_IDS not in atoms.arrays:
            raise ValueError('Chemical topology screening requires explicit topology_atom_ids')
        ids = validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
        atoms = atoms[np.argsort(ids, kind='stable')]
    except ValueError as exc:
        return TopologyValidationResult(False, 'topology_identity',
            dict(topology_enabled=True, topology_reference_sha256=topology.reference_sha256),
            (dict(kind='topology_identity', detail=str(exc)),))
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    expected_numbers = np.asarray(topology.atomic_numbers, dtype=np.int64)
    base_metrics = {
        "topology_enabled": True,
        "topology_reference_path": topology.reference_path,
        "topology_reference_sha256": topology.reference_sha256,
        "topology_reference_format": topology.reference_format,
        "topology_bond_min_scale": topology.bond_min_scale,
        "topology_bond_max_scale": topology.bond_max_scale,
        "topology_connectivity_scale": topology.connectivity_scale,
    }
    if not np.array_equal(numbers, expected_numbers):
        violation = {
            "kind": "topology_identity",
            "expected_atomic_numbers": expected_numbers.tolist(),
            "actual_atomic_numbers": numbers.tolist(),
        }
        return TopologyValidationResult(
            False,
            "topology_identity",
            base_metrics,
            (violation,),
        )

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if (
        positions.shape != (topology.n_atoms, 3)
        or not np.isfinite(positions).all()
    ):
        violation = {
            "kind": "topology_nonfinite",
            "detail": (
                "Coordinates are non-finite or do not have shape "
                f"({topology.n_atoms}, 3)."
            ),
        }
        return TopologyValidationResult(
            False,
            "topology_nonfinite",
            base_metrics,
            (violation,),
        )

    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )
    violations: list[dict[str, Any]] = []
    first, second, references, pair_first, pair_second, radius_sums = _topology_arrays(topology)
    # Elementwise, the same float64 operations, comparisons, and order as one pair at a time.
    bond_distances = distances[first, second]
    ratios = bond_distances / references
    below_minimum = (ratios < topology.bond_min_scale) & ~np.isclose(
        ratios, topology.bond_min_scale, rtol=1.0e-12, atol=1.0e-12)
    above_maximum = (ratios > topology.bond_max_scale) & ~np.isclose(
        ratios, topology.bond_max_scale, rtol=1.0e-12, atol=1.0e-12)
    for bond in np.flatnonzero(below_minimum | above_maximum):
        violations.append(
            {
                "kind": "topology_bond_window",
                "atom_indices": [int(first[bond]), int(second[bond])],
                "distance_A": float(bond_distances[bond]),
                "reference_distance_A": float(references[bond]),
                "ratio": float(ratios[bond]),
                "allowed_ratio": [
                    topology.bond_min_scale,
                    topology.bond_max_scale,
                ],
            }
        )
    bond_ratios = ratios.tolist()
    pair_distances = distances[pair_first, pair_second]
    nonbonded = pair_distances / radius_sums
    for pair in np.flatnonzero(nonbonded <= topology.connectivity_scale):
        violations.append(
            {
                "kind": "topology_extra_bond",
                "atom_indices": [int(pair_first[pair]), int(pair_second[pair])],
                "distance_A": float(pair_distances[pair]),
                "covalent_radius_ratio": float(nonbonded[pair]),
                "connectivity_scale": topology.connectivity_scale,
            }
        )
    nonbonded_ratios = nonbonded.tolist()

    metrics = {
        **base_metrics,
        "topology_min_bond_ratio": float(min(bond_ratios)),
        "topology_max_bond_ratio": float(max(bond_ratios)),
        "topology_min_nonbonded_covalent_ratio": (
            float(min(nonbonded_ratios))
            if nonbonded_ratios
            else None
        ),
        "topology_violation_count": len(violations),
    }
    reason = str(violations[0]["kind"]) if violations else None
    return TopologyValidationResult(
        not violations,
        reason,
        metrics,
        tuple(violations),
    )


def topology_metadata(
    result: TopologyValidationResult,
) -> dict[str, Any]:
    """Return JSON-safe candidate provenance for a topology result."""

    return {
        **dict(result.metrics),
        "topology_valid": bool(result.valid),
        "topology_reject_reason": result.reason,
        "topology_violations": [
            dict(violation) for violation in result.violations
        ],
    }
