"""Unlabelled local CPPn pools from an explicit, minimized S0 reference.

Phenyl rotations and projected Cartesian rattles follow the original CPP
initial-sampling study. No QM, optimization, or GPU import occurs here.
This method supports para-linked cycloparaphenylenes of any size, not arbitrary
ring chemistry; use rdkit_molecule for reference-centred general molecules.
Run ``python -m alframework.builders.ring_molecule --help`` for CPU preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any

import h5py
import numpy as np
from ase.data import chemical_symbols, covalent_radii
from ase.io import read
from parsl import python_app

SCHEMA = "alf.cpp_ring_pool.v1"
DEFAULT_SETTINGS = {
    "count": 1000, "seed": 42, "shuffle_samples": True,
    "tiers": [
        {"name": "narrow", "fraction": .5, "rotation_range_degrees": [-15., 15.],
         "rattle_rms_range_angstrom": [0., .015]},
        {"name": "broad", "fraction": .5, "rotation_range_degrees": [-30., 30.],
         "rattle_rms_range_angstrom": [0., .03]},
    ],
    "screening": {"bond_min_scale": .9, "bond_max_scale": 1.1,
        "connectivity_scale": 1.2, "duplicate_precision_angstrom": 1.e-6,
        "max_attempts_per_structure": 50,
        "nonbonded_min_angstrom": {"1-1": 1.2, "1-6": 1.4, "6-6": 1.8}},
}


def validate_config(config):
    """Validate preparation settings without importing RDKit or GPU libraries."""
    unknown = set(config) - set(DEFAULT_SETTINGS) - {'reference_geometry'}
    if unknown:
        raise ValueError(f'Unknown ring preparation settings: {sorted(unknown)}')
    for name, minimum in [('count', 1), ('seed', 0)]:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(config['shuffle_samples'], bool):
        raise ValueError('shuffle_samples must be Boolean')
    tiers = config['tiers']
    if not isinstance(tiers, list) or not tiers:
        raise ValueError('At least one sampling tier is required')
    names = []
    for tier in tiers:
        if set(tier) - {'name', 'fraction', 'rotation_range_degrees', 'rotation_max_degrees',
                        'rattle_rms_range_angstrom', 'rattle_rms_angstrom'}:
            raise ValueError('Unknown ring sampling tier setting')
        names.append(tier['name'])
        if not isinstance(tier['name'], str) or not tier['name']:
            raise ValueError('Sampling tier names must be nonempty strings')
        fraction = tier['fraction']
        if isinstance(fraction, bool) or not np.isfinite(fraction) or fraction <= 0:
            raise ValueError('Tier fractions must be finite and positive')
        component_ranges(tier)
    if len(set(names)) != len(names) or not np.isclose(sum(t['fraction'] for t in tiers), 1.):
        raise ValueError('Tier names must be unique and fractions must sum to one')
    screen = config['screening']
    if set(screen) != set(DEFAULT_SETTINGS['screening']):
        raise ValueError('Missing or unknown ring screening setting')
    for key in ('bond_min_scale', 'bond_max_scale', 'connectivity_scale', 'duplicate_precision_angstrom'):
        value = screen[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not screen['bond_min_scale'] < 1 < screen['bond_max_scale']:
        raise ValueError('Bond-length bounds must bracket one')
    attempts = screen['max_attempts_per_structure']
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError('max_attempts_per_structure must be a positive integer')
    limits = screen['nonbonded_min_angstrom']
    if set(limits) != {'1-1', '1-6', '6-6'} or any(
        isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in limits.values()
    ):
        raise ValueError('Invalid nonbonded distance limits')
    return config


def _components(adjacency):
    remaining = set(range(len(adjacency)))
    while remaining:
        reached, pending = set(), [min(remaining)]
        while pending:
            atom = pending.pop()
            if atom in reached:
                continue
            reached.add(atom)
            pending.extend(set(np.flatnonzero(adjacency[atom])) - reached)
        remaining -= reached
        yield sorted(reached)


def infer_topology(atoms, screening):
    """Infer a single para-linked CPP macrocycle without atom reordering."""
    positions, numbers = np.asarray(atoms.positions), np.asarray(atoms.numbers)
    if atoms.pbc.any() or set(numbers.tolist()) != {1, 6} or not np.isfinite(positions).all():
        raise ValueError('Expected a finite nonperiodic explicit-hydrogen CPP reference')
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    thresholds = screening['connectivity_scale'] * (covalent_radii[numbers, None] + covalent_radii[numbers][None, :])
    adjacency = (distances < thresholds) & ~np.eye(len(atoms), dtype=bool)
    carbons = np.flatnonzero(numbers == 6)
    nring, remainder = divmod(len(carbons), 6)
    if remainder or nring < 3 or len(atoms) != nring * 10 or len(list(_components(adjacency))) != 1:
        raise ValueError('Reference must be a connected C(6n)H(4n) CPP macrocycle, n >= 3')
    carbon_graph = adjacency & (numbers[:, None] == 6) & (numbers[None, :] == 6)
    degree = carbon_graph.sum(axis=1)
    links = np.argwhere(np.triu(carbon_graph & (degree[:, None] == 3) & (degree[None, :] == 3), 1))
    phenyl_graph = carbon_graph.copy()
    if len(links):
        phenyl_graph[links[:, 0], links[:, 1]] = phenyl_graph[links[:, 1], links[:, 0]] = False
    groups, anchors, rings = [], [], []
    for ring in _components(phenyl_graph):
        if numbers[ring[0]] != 6:
            continue
        if len(ring) != 6 or any(phenyl_graph.sum(axis=1)[ring] != 2):
            raise ValueError('Expected six-membered carbon cycles')
        ends = [atom for atom in ring if degree[atom] == 3]
        if len(ends) != 2 or phenyl_graph[ends[0], ends[1]] or (phenyl_graph[ends[0]] & phenyl_graph[ends[1]]).any():
            raise ValueError('Expected two para connection carbons per phenyl')
        hydrogens = sorted(set(np.flatnonzero(adjacency[ring].any(axis=0))) & set(np.flatnonzero(numbers == 1)))
        if len(hydrogens) != 4 or any(adjacency.sum(axis=1)[hydrogens] != 1):
            raise ValueError('Expected four terminal hydrogens per phenyl')
        groups.append(ring + hydrogens)
        anchors.append(ends)
        rings.append(ring)
    membership = {atom: i for i, ring in enumerate(rings) for atom in ring}
    if len(rings) != nring:
        raise ValueError('Reference does not contain the expected phenyl groups')
    macrocycle = np.zeros((nring, nring), dtype=bool)
    for a, b in links:
        macrocycle[membership[a], membership[b]] = macrocycle[membership[b], membership[a]] = True
    if (len(groups) != nring or len(links) != nring or adjacency.sum() != 22 * nring
            or sorted(atom for group in groups for atom in group) != list(range(len(atoms)))
            or len(list(_components(macrocycle))) != 1 or any(macrocycle.sum(axis=1) != 2)):
        raise ValueError('Reference is not a single para-linked CPP macrocycle')
    bonds = np.argwhere(np.triu(adjacency, 1))
    # The historical nonbonded clearance excludes graph distances one and two.
    within_two = adjacency | ((adjacency.astype(int) @ adjacency.astype(int)) > 0)
    pairs = np.argwhere(np.triu(~within_two, 1))
    limits = [screening['nonbonded_min_angstrom']['-'.join(map(str, sorted(numbers[[a, b]])))] for a, b in pairs]
    return dict(nring=nring, groups=groups, anchors=anchors, rings=rings,
                links=links.tolist(), adjacency=adjacency, bonds=bonds,
                bond_lengths=distances[bonds[:, 0], bonds[:, 1]],
                connectivity_thresholds=thresholds, nonbonded_pairs=pairs,
                nonbonded_limits=np.asarray(limits))

def component_ranges(component: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve range settings while retaining the original fixed-rattle schema."""
    if "rotation_range_degrees" in component and "rotation_max_degrees" in component:
        raise ValueError("Specify a rotation interval OR a legacy maximum, not both.")
    if "rattle_rms_range_angstrom" in component and "rattle_rms_angstrom" in component:
        raise ValueError("Specify a rattle interval OR a fixed RMS, not both.")
    rotation = component.get("rotation_range_degrees")
    if rotation is None:
        maximum = float(component["rotation_max_degrees"])
        rotation = [-maximum, maximum]
    rattle = component.get("rattle_rms_range_angstrom")
    if rattle is None:
        rattle = [component["rattle_rms_angstrom"]] * 2
    intervals = []
    for values, low, high, name in ((rotation, -180, 180, "rotation"), (rattle, 0, np.inf, "rattle")):
        bounds = np.asarray(values, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]
                or bounds[0] < low or bounds[1] > high):
            raise ValueError(f"Invalid {name} interval: {values}.")
        intervals.append(tuple(float(x) for x in bounds))
    return intervals[0], intervals[1]

def rotate_phenyls(positions: np.ndarray, topology: dict[str, Any], angles: np.ndarray) -> np.ndarray:
    """Rodrigues rotations about fixed para endpoints, including attached H."""
    if len(angles) != topology["nring"]:
        raise ValueError("Supply one angle per phenyl group.")
    result = positions.copy()
    for group, (a, b), degrees in zip(topology["groups"], topology["anchors"], angles):
        origin = positions[a]
        axis = positions[b] - origin
        axis = axis / np.linalg.norm(axis)
        displacement = positions[group] - origin
        theta = np.deg2rad(degrees)
        result[group] = (origin + displacement * np.cos(theta)
                         + np.cross(axis, displacement) * np.sin(theta)
                         + np.outer(displacement @ axis, axis) * (1 - np.cos(theta)))
        result[[a, b]] = positions[[a, b]]  # Avoid roundoff at the fixed endpoints.
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_ring_pool(config, output_dir, *, config_directory=None):
    """Atomically publish geometry-only seeds and their reference/provenance.

    The caller supplies a ground-state minimized reference in the intended
    electronic-structure environment. Preparation does not optimize it or
    establish its QM convergence. Explicit reference atom order is retained.
    """
    if not isinstance(config, dict) or not isinstance(config.get('reference_geometry'), str) or not config['reference_geometry']:
        raise ValueError('reference_geometry must identify a minimized S0 reference')
    settings = deepcopy(DEFAULT_SETTINGS)
    settings.update(deepcopy(config))
    validate_config(settings)
    reference = Path(settings['reference_geometry']).expanduser()
    if not reference.is_absolute():
        reference = Path(config_directory or '.') / reference
    reference = reference.resolve(strict=True)
    if reference.suffix.lower() not in {'.xyz', '.extxyz'}:
        raise ValueError('Ring preparation requires an XYZ/extxyz minimized reference')
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing ring preparation output: {output}')
    fingerprint = _sha256(reference)
    atoms = read(reference, parallel=False)
    samples = generate_samples(atoms, settings, settings['count'], settings['seed'])
    report = dict(schema=SCHEMA, status='completed', settings=settings,
                  reference_sha256=fingerprint, reference_optimization_performed=False,
                  implementation_sha256=_sha256(__file__),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, h5py=h5py.__version__),
                  ring_size=samples['topology']['nring'], selected_count=settings['count'],
                  tier_counts=np.bincount(samples['tier_index'], minlength=len(settings['tiers'])).tolist(),
                  rejections=samples['rejections'], sampling_seconds=samples['sampling_seconds'],
                  storage_to_reference=list(range(len(atoms))),
                  limitation='Local coverage-oriented distortions; not a Boltzmann ensemble or conformational search')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        shutil.copyfile(reference, temporary/'reference.xyz')
        if _sha256(reference) != fingerprint or _sha256(temporary/'reference.xyz') != fingerprint:
            raise RuntimeError('Reference changed during ring preparation')
        with h5py.File(temporary/'seed.h5', 'w') as handle:
            handle.attrs.update(schema=SCHEMA, energy_force_labels_present=False)
            group = handle.create_group(f'cpp{report["ring_size"]}')
            group['species'] = np.asarray([chemical_symbols[z] for z in atoms.numbers], dtype='S2')
            group.create_dataset('coordinates', data=samples['coordinates'], compression='gzip', shuffle=True)
            group['topology_atom_ids'] = np.arange(len(atoms), dtype=np.int64)
            group['_id'] = np.asarray([f'cpp{report["ring_size"]}-seed-{i:010d}' for i in range(settings['count'])], dtype='S40')
            for key in ('rotation_angles_degrees', 'rattle_rms_angstrom', 'requested_rattle_rms_angstrom', 'tier_index', 'attempt_count'):
                group[key] = samples[key]
        (temporary/'builder_config.json').write_text(json.dumps(dict(seed_manifest_path='seeds/provenance.json', selection_seed=settings['seed']), indent=2)+'\n')
        report['artifacts_sha256'] = {name:_sha256(temporary/name) for name in ('seed.h5', 'reference.xyz', 'builder_config.json')}
        (temporary/'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def build_ring_molecules(moleculeids, builder_config, h5_path=None, current_h5_id=0, master_directory=None):
    """Sequential finite bootstrap, then deterministic closed-shard replay."""
    from alframework.builders.external_geometry import build_external_geometries
    root = Path(master_directory or '.').expanduser().resolve()
    path = Path(builder_config['seed_manifest_path']).expanduser()
    path = path if path.is_absolute() else root/path
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'completed':
        raise ValueError('Ring builder requires a completed preparation manifest')
    for name in ('seed.h5', 'reference.xyz'):
        if _sha256(path.parent/name) != manifest['artifacts_sha256'][name]:
            raise ValueError('Prepared ring input changed: '+name)
    if int(current_h5_id) == 0:
        replay = dict(sources=[str(path.parent/'seed.h5')], selection_mode='sequential')
    else:
        if h5_path is None:
            raise ValueError('Runtime ring replay requires h5_path')
        sources = []
        for index in range(int(current_h5_id)):
            shard = Path(str(h5_path).format(index))
            shard = shard if shard.is_absolute() else root/shard
            if not shard.is_file():
                raise FileNotFoundError(f'Closed label shard missing: {shard}')
            sources.append(str(shard))
        replay = dict(sources=sources, selection_mode='deterministic_hash',
                      selection_seed=builder_config.get('selection_seed', 42))
    molecules = build_external_geometries(moleculeids, replay, root)
    for molecule in molecules:
        molecule.update_metadata(dict(ring_manifest_sha256=_sha256(path), ring_size=manifest['ring_size'],
                                      replay_external_bootstrap=int(current_h5_id)==0))
        if int(current_h5_id) == 0:
            with h5py.File(path.parent/'seed.h5') as handle:
                metadata = molecule.get_metadata()
                group, frame = handle[metadata['geometry_group']], metadata['geometry_frame']
                molecule.update_metadata(dict(ring_tier=int(group['tier_index'][frame]),
                                              ring_rotation_angles_degrees=group['rotation_angles_degrees'][frame].tolist(),
                                              ring_rattle_rms_angstrom=float(group['rattle_rms_angstrom'][frame])))
    return molecules


@python_app(executors=['alf_sampler_executor'])
def ring_molecule_builder_task(moleculeid=None, moleculeids=None, builder_config=None,
                               h5_path=None, current_h5_id=0, master_directory=None):
    """Ordinary scalar/plural ALF builder; no generation or QM inside workers."""
    plural = moleculeids is not None
    molecules = build_ring_molecules(moleculeids if plural else [moleculeid], builder_config or {},
                                    h5_path, current_h5_id, master_directory)
    return molecules if plural else molecules[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.config.expanduser().resolve(strict=True)
    report = prepare_ring_pool(json.loads(source.read_text()), args.output_dir, config_directory=source.parent)
    print(f'Prepared {report["selected_count"]} unlabeled CPP{report["ring_size"]} geometries in {args.output_dir}')
    return 0

def random_rattle(positions: np.ndarray, rms: float, rng: Any) -> np.ndarray:
    """Remove six rigid directions; normalize sqrt(mean_i |dR_i|^2)."""
    displacement = rng.normal(size=positions.shape)
    displacement -= displacement.mean(axis=0)
    centered = positions - positions.mean(axis=0)
    rotations = np.stack([np.cross(axis, centered).reshape(-1) for axis in np.eye(3)], axis=1)
    coefficients = np.linalg.lstsq(rotations, displacement.reshape(-1), rcond=None)[0]
    displacement -= (rotations @ coefficients).reshape(positions.shape)
    displacement -= displacement.mean(axis=0)
    actual = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
    if actual <= 0:
        raise ValueError("Degenerate random displacement.")
    return displacement * (rms / actual)

def align_to_anchor(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove overall rigid motion with fixed atom correspondence, no reflections."""
    p = positions - positions.mean(axis=0)
    q = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    signs = np.eye(3)
    signs[-1, -1] = np.linalg.det(u @ vt)
    return p @ u @ signs @ vt + reference.mean(axis=0)

def screen_geometry(positions: np.ndarray, topology: dict[str, Any], settings: dict[str, Any]) -> str | None:
    if positions.shape != topology["adjacency"].shape[:1] + (3,) or not np.isfinite(positions).all():
        return "nonfinite_or_shape"
    d = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    bonds = topology["bonds"]
    ratio = d[bonds[:, 0], bonds[:, 1]] / topology["bond_lengths"]
    if np.any(ratio < settings["bond_min_scale"]) or np.any(ratio > settings["bond_max_scale"]):
        return "bond_length"
    adjacency = (d < topology["connectivity_thresholds"]) & ~np.eye(len(positions), dtype=bool)
    if not np.array_equal(adjacency, topology["adjacency"]):
        return "connectivity"
    pairs = topology["nonbonded_pairs"]
    if np.any(d[pairs[:, 0], pairs[:, 1]] < topology["nonbonded_limits"]):
        return "nonbonded_clash"
    return None

def tier_counts(count: int, tiers: list[dict[str, Any]]) -> list[int]:
    if count < 1:
        raise ValueError("Count must be positive.")
    if all("accepted_count" in t for t in tiers):
        quotas = [int(t["accepted_count"]) for t in tiers]
        if sum(quotas) != count:
            raise ValueError("Requested count differs from explicit component counts.")
        return quotas
    quotas = [int(np.floor(count * t["fraction"])) for t in tiers]
    quotas[-1] += count - sum(quotas)
    return quotas

def generate_samples(atoms: Any, config: dict[str, Any], count: int, seed: int) -> dict[str, Any]:
    validate_config(config)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    topology = infer_topology(atoms, config["screening"])
    anchor = np.asarray(atoms.positions, dtype=np.float64)
    invalid = screen_geometry(anchor, topology, config["screening"])
    if invalid:
        raise ValueError(f"Optimized anchor fails screening: {invalid}.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, topology["nring"]]))
    reference_key = np.rint(anchor / config["screening"]["duplicate_precision_angstrom"]).astype(np.int64).tobytes()
    seen = {reference_key}
    coordinates, rotations, rattles, requested_rattles, labels, attempts = [], [], [], [], [], []
    rejected: Counter[str] = Counter()
    started = time.perf_counter()
    for index, (tier, quota) in enumerate(zip(config["tiers"], tier_counts(count, config["tiers"]))):
        rotation_bounds, rattle_bounds = component_ranges(tier)
        for _ in range(quota):
            for attempt in range(1, config["screening"]["max_attempts_per_structure"] + 1):
                angles = rng.uniform(*rotation_bounds, topology["nring"])
                rotated = rotate_phenyls(anchor, topology, angles)
                rms = rattle_bounds[0] if rattle_bounds[0] == rattle_bounds[1] else float(rng.uniform(*rattle_bounds))
                displacement = random_rattle(rotated, rms, rng)
                candidate = align_to_anchor(rotated + displacement, anchor)
                reason = screen_geometry(candidate, topology, config["screening"])
                key = np.rint(candidate / config["screening"]["duplicate_precision_angstrom"]).astype(np.int64).tobytes()
                if reason is None and key in seen:
                    reason = "duplicate"
                if reason:
                    rejected[reason] += 1
                    continue
                seen.add(key)
                coordinates.append(candidate)
                rotations.append(angles)
                rattles.append(float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))))
                requested_rattles.append(rms)
                labels.append(index)
                attempts.append(attempt)
                break
            else:
                raise RuntimeError(f"Cannot fill {tier['name']} tier after {attempt} attempts for structure {len(coordinates)}; rejections={dict(rejected)}.")
    result = {
        "coordinates": np.asarray(coordinates, dtype=np.float64),
        "species": np.asarray(atoms.numbers, dtype=np.int64),
        "rotation_angles_degrees": np.asarray(rotations, dtype=np.float64),
        "rattle_rms_angstrom": np.asarray(rattles, dtype=np.float64),
        "requested_rattle_rms_angstrom": np.asarray(requested_rattles, dtype=np.float64),
        "tier_index": np.asarray(labels, dtype=np.int64),
        "attempt_count": np.asarray(attempts, dtype=np.int64),
        "rejections": dict(rejected), "topology": topology,
        "sampling_seconds": time.perf_counter() - started,
    }
    if result["coordinates"].shape != (count, len(atoms), 3):
        raise RuntimeError("Sampling output shape mismatch.")
    if config.get("shuffle_samples", False):
        order = rng.permutation(count)
        for key in ("coordinates", "rotation_angles_degrees", "rattle_rms_angstrom",
                    "requested_rattle_rms_angstrom", "tier_index", "attempt_count"):
            result[key] = result[key][order]
        result["generation_index"] = order
    return result


if __name__ == '__main__':
    raise SystemExit(main())
