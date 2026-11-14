"""Prepare unlabelled, reference-centred KDG-ladder seeds with stratified FPS.

Run ``python -m alframework.builders.rdkit_molecule --help``. RDKit is an
optional dependency; no dynamics, force-field optimization, or QM is performed.
The numerical method follows the original ALF initial-sampling benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import shutil
import tempfile

import h5py
import numpy as np
from ase.data import chemical_symbols, covalent_radii

from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS
from alframework.tools.files import cached_sha256


SCHEMA = "alf.rdkit_kdg_fps.v1"
DEFAULT_SETTINGS = {
    "pool_size": 50000, "seed_size": 5000, "seed": 42, "threads": 1,
    "box_size_mult": 4.0, "max_embedding_rounds": 5, "polar_h": True,
    "bond_scales": [0., .02, .04, .06, .08, .10],
    "angle_scales": [0., .02, .04, .06, .08, .10],
    "xh_bond_scales": [0., .01, .02, .03, .03, .03],
    "xh_angle_scales": [0., .01, .02, .03, .03, .03],
    "torsion_scales": [0., .06, .12, .18, .24, .30],
    "geometry_screen": {
        "bond_min_scale": .85, "bond_max_scale": 1.50,
        "nonbonded_min_scale": 1.40, "connectivity_scale": 1.25,
        "min_distance_A": .7,
    },
}
EMBEDDING_SETTINGS = {
    "useRandomCoords": True, "useExpTorsionAnglePrefs": False,
    "useBasicKnowledge": True, "pruneRmsThresh": -1.,
    "enforceChirality": True, "trackFailures": True,
    "enableSequentialRandomSeeds": True,
}


def _rdkit():
    try:
        from rdkit import Chem, DistanceGeometry, __version__
        from rdkit.Chem import AllChem, rdDistGeom
    except ImportError as exc:
        raise ImportError("RDKit seed preparation requires ALF's topology extra: pip install '.[topology]'.") from exc
    return Chem, DistanceGeometry, AllChem, rdDistGeom, __version__


def resolve_config(config, *, base_directory=None):
    """Validate JSON settings; reference paths are relative to the config file."""
    if not isinstance(config, dict):
        raise TypeError("Seed configuration must be a JSON object.")
    unknown = set(config) - set(DEFAULT_SETTINGS) - {"smiles", "reference_geometry", "reference_charge"}
    if unknown:
        raise ValueError(f"Unknown seed configuration keys: {sorted(unknown)}")
    settings = {**DEFAULT_SETTINGS, **config}
    for key in ("smiles", "reference_geometry"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    path = Path(settings['reference_geometry']).expanduser()
    if not path.is_absolute():
        path = Path(base_directory or Path.cwd()) / path
    settings['reference_geometry'] = str(path.resolve(strict=True))
    for key in ('pool_size', 'seed_size', 'threads', 'max_embedding_rounds', 'seed'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if key == 'seed' else 'positive'} integer.")
    if settings['seed_size'] > settings['pool_size']:
        raise ValueError("seed_size cannot exceed pool_size.")
    if 'reference_charge' in settings:
        charge = settings['reference_charge']
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise ValueError('reference_charge must be an integer.')
    if not isinstance(settings['polar_h'], bool):
        raise ValueError('polar_h must be Boolean.')
    scale_keys = ('bond_scales', 'angle_scales', 'xh_bond_scales', 'xh_angle_scales', 'torsion_scales')
    for key in scale_keys:
        values = settings[key]
        if not isinstance(values, list) or not values or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
            for v in values
        ):
            raise ValueError(f'{key} must be a nonempty list of finite nonnegative scales.')
        settings[key] = list(map(float, values))
    if len({len(settings[key]) for key in scale_keys}) != 1:
        raise ValueError('All ladder scale lists must have the same length.')
    screen = config.get('geometry_screen', {})
    if not isinstance(screen, dict) or set(screen) - set(DEFAULT_SETTINGS['geometry_screen']):
        raise ValueError('geometry_screen contains unknown settings or is not an object.')
    settings['geometry_screen'] = {**DEFAULT_SETTINGS['geometry_screen'], **screen}
    numeric = {'box_size_mult': settings['box_size_mult'], **settings['geometry_screen']}
    for key, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if settings['geometry_screen']['bond_min_scale'] > settings['geometry_screen']['bond_max_scale']:
        raise ValueError('bond_min_scale cannot exceed bond_max_scale.')
    return settings


def ladder_quotas(total, n_strata):
    """Even quotas, assigning remainder to the least-distorted levels first."""
    if n_strata < 1 or total < 0:
        raise ValueError('Quotas require n_strata >= 1 and total >= 0.')
    quotient, remainder = divmod(total, n_strata)
    return [quotient + (index < remainder) for index in range(n_strata)]


def stratum_seed(seed, level, round_index):
    """Keep the benchmark's method ID 4 and KDG mode ID 2 unchanged."""
    entropy = [int(seed), 4, int(level), 2, int(round_index)]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0] % np.iinfo(np.int32).max)


def _connectivity_graph(molecule):
    Chem, *_ = _rdkit()
    graph = Chem.RWMol()
    for atom in molecule.GetAtoms():
        graph.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in molecule.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    graph = graph.GetMol()
    Chem.SanitizeMol(graph, Chem.SanitizeFlags.SANITIZE_NONE)
    return graph


def _reference_context(settings):
    Chem, *_ = _rdkit()
    from alframework.tools.molecular_topology import _load_rdkit_reference

    molecule = Chem.MolFromSmiles(settings['smiles'])
    if molecule is None:
        raise ValueError('Invalid SMILES.')
    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() < 2:
        raise ValueError('Pair-distance seed selection requires at least two atoms.')
    charge = int(Chem.GetFormalCharge(molecule))
    if settings.get('reference_charge', charge) != charge:
        raise ValueError('reference_charge does not match the SMILES formal charge.')
    reference, _ = _load_rdkit_reference(Path(settings['reference_geometry']), 'auto', charge)
    if reference.GetNumAtoms() != molecule.GetNumAtoms():
        raise ValueError('Reference and hydrogen-complete SMILES have different atom counts.')
    match = _connectivity_graph(reference).GetSubstructMatch(_connectivity_graph(molecule))
    if len(match) != molecule.GetNumAtoms() or len(set(match)) != len(match):
        raise ValueError('Reference connectivity does not match the SMILES graph.')
    # match[native atom] is its reference ID, not a species-sorting permutation.
    match = np.asarray(match, dtype=np.int64)
    native_bonds = np.asarray(sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                                      for b in molecule.GetBonds()), dtype=np.int64).reshape(-1, 2)
    reference_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in reference.GetBonds()}
    if {tuple(sorted(match[pair])) for pair in native_bonds} != reference_bonds:
        raise ValueError('Atom mapping does not reproduce the complete reference bond set.')
    positions = np.asarray(reference.GetConformer().GetPositions(), dtype=float)[match]
    if not np.isfinite(positions).all():
        raise ValueError('Reference coordinates must be finite.')
    numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64)
    pairs = np.asarray(list(combinations(range(len(numbers)), 2)), dtype=np.int64)
    bond_set = {tuple(pair) for pair in native_bonds}
    nonbonded = np.asarray([pair for pair in pairs if tuple(pair) not in bond_set], dtype=np.int64).reshape(-1, 2)
    centers = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    return {'molecule': molecule, 'numbers': numbers, 'native_to_reference': match,
            'centers': centers, 'bonds': native_bonds, 'pairs': pairs,
            'nonbonded': nonbonded, 'reference_charge': charge,
            'bond_centers': centers[native_bonds[:, 0], native_bonds[:, 1]]}


def widened_bounds_matrix(molecule, centers, settings, level):
    """Reference-centred 1–2/1–3 intervals and widened heavy 1–4 intervals."""
    Chem, DistanceGeometry, _, rdDistGeom, _ = _rdkit()
    bounds = np.asarray(rdDistGeom.GetMoleculeBoundsMatrix(
        molecule, set15bounds=True, scaleVDW=False, doTriangleSmoothing=True,
        useMacrocycle14config=True), dtype=float)
    separations = np.asarray(Chem.GetDistanceMatrix(molecule))
    heavy = [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()]
    heavy_or_polar = [heavy[i] or (settings['polar_h'] and any(
        neighbor.GetAtomicNum() in {7, 8, 16} for neighbor in atom.GetNeighbors()))
        for i, atom in enumerate(molecule.GetAtoms())]
    for first, second in combinations(range(molecule.GetNumAtoms()), 2):
        separation = int(separations[first, second])
        lower, upper = bounds[second, first], bounds[first, second]
        if separation >= 3:
            if separation == 3 and heavy[first] and heavy[second]:
                scale = settings['torsion_scales'][level]
                midpoint = .5 * (lower + upper)
                bounds[second, first] = min(lower, midpoint * (1. - scale))
                bounds[first, second] = max(upper, midpoint * (1. + scale))
            continue
        key = 'bond_scales' if separation == 1 else 'angle_scales'
        if not (heavy_or_polar[first] and heavy_or_polar[second]):
            key = 'xh_' + key
        center = centers[first, second]
        half_width = max(.5 * (upper - lower), settings[key][level] * center)
        bounds[second, first], bounds[first, second] = center - half_width, center + half_width
    if not DistanceGeometry.DoTriangleSmoothing(bounds):
        raise RuntimeError(f'Triangle smoothing failed for ladder level {level}.')
    return bounds


def geometry_failure(positions, context, screen):
    """First seed-screen failure, in the original benchmark's priority order."""
    numbers = context['numbers']
    positions = np.asarray(positions)
    if positions.shape != (len(numbers), 3) or not np.isfinite(positions).all():
        return 'non_finite'
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    bonds = context['bonds']
    ratios = distances[bonds[:, 0], bonds[:, 1]] / context['bond_centers']
    if np.any((ratios < screen['bond_min_scale']) | (ratios > screen['bond_max_scale'])):
        return 'bond_window'
    pairs = context['nonbonded']
    radii = covalent_radii[numbers[pairs[:, 0]]] + covalent_radii[numbers[pairs[:, 1]]]
    nonbonded = distances[pairs[:, 0], pairs[:, 1]]
    if np.any(nonbonded <= screen['connectivity_scale'] * radii):
        return 'inferred_extra_bond'
    if np.any(nonbonded < screen['nonbonded_min_scale'] * radii):
        return 'nonbonded_close'
    pairs = context['pairs']
    if np.min(distances[pairs[:, 0], pairs[:, 1]]) < screen['min_distance_A']:
        return 'min_distance'
    return None


def generate_pool(settings, context):
    """Generate exact accepted quotas; insufficient strata fail explicitly."""
    Chem, _, AllChem, _, _ = _rdkit()
    n_levels = len(settings['bond_scales'])
    quotas = ladder_quotas(settings['pool_size'], n_levels)
    frames, strata, diagnostics = [], [], []
    for level, quota in enumerate(quotas):
        bounds = widened_bounds_matrix(context['molecule'], context['centers'], settings, level)
        accepted, rounds, rejections = [], [], Counter()
        for round_index in range(settings['max_embedding_rounds']):
            missing = quota - len(accepted)
            if missing == 0:
                break
            seed = stratum_seed(settings['seed'], level, round_index)
            parameters = AllChem.ETKDGv3()
            for key, value in EMBEDDING_SETTINGS.items():
                setattr(parameters, key, value)
            parameters.randomSeed = seed
            parameters.numThreads = settings['threads']
            parameters.boxSizeMult = settings['box_size_mult']
            parameters.SetBoundsMat(bounds)
            work = Chem.Mol(context['molecule'])
            ids = list(AllChem.EmbedMultipleConfs(work, numConfs=missing, params=parameters))
            rejected = Counter()
            for conformer_id in ids:
                positions = np.asarray(work.GetConformer(conformer_id).GetPositions(), dtype=float)
                failure = geometry_failure(positions, context, settings['geometry_screen'])
                if failure is None:
                    accepted.append(positions.copy())
                else:
                    rejected[failure] += 1
            rejections.update(rejected)
            rounds.append({'round': round_index, 'seed': seed, 'requested': missing,
                           'embedded': len(ids), 'rejection_counts': dict(rejected),
                           'embedding_failure_counts': list(parameters.GetFailureCounts())})
            if not ids:
                break
        detail = {'level': level, 'torsion_mode': 'kdg', 'mode_index': 2,
                  'quota': quota, 'accepted': len(accepted), 'rounds': rounds,
                  'rejection_counts': dict(rejections)}
        if len(accepted) != quota:
            raise RuntimeError(f'KDG ladder stratum could not meet its quota: {json.dumps(detail)}')
        diagnostics.append(detail)
        frames.extend(accepted)
        strata.extend([(level, 2)] * quota)
    return np.asarray(frames), np.asarray(strata, dtype=np.int64), diagnostics


def pair_descriptor(coordinates, numbers):
    """Unscaled pair distances sorted within each unordered element-pair type."""
    numbers = np.asarray(numbers)
    first, second = np.triu_indices(len(numbers), 1)
    low, high = np.minimum(numbers[first], numbers[second]), np.maximum(numbers[first], numbers[second])
    distances = np.linalg.norm(coordinates[:, first] - coordinates[:, second], axis=-1)
    return np.concatenate([np.sort(distances[:, (low == a) & (high == b)], axis=1)
                           for a, b in sorted(set(zip(low.tolist(), high.tolist())))], axis=1)


def farthest_point_order(features, count):
    """Centroid-nearest start, max–min traversal, lowest-row tie breaking."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all() or not 0 < count <= len(features):
        raise ValueError('FPS needs finite 2D features and 0 < count <= population size.')
    order = np.empty(count, dtype=np.int64)
    order[0] = np.argmin(((features - features.mean(axis=0)) ** 2).sum(axis=1))
    nearest, chosen = np.full(len(features), np.inf), np.zeros(len(features), dtype=bool)
    chosen[order[0]] = True
    for position in range(1, count):
        nearest = np.minimum(nearest, np.sqrt(((features - features[order[position-1]]) ** 2).sum(axis=1)))
        nearest[chosen] = -1.
        order[position] = np.argmax(nearest)
        chosen[order[position]] = True
    return order


def select_seeds(coordinates, numbers, strata, count, n_levels):
    features = pair_descriptor(coordinates, numbers)
    selected, ranks, details = [], [], []
    for level, quota in enumerate(ladder_quotas(count, n_levels)):
        members = np.flatnonzero(strata[:, 0] == level)
        if quota > len(members):
            raise ValueError(f'Stratum {level} has {len(members)} frames; {quota} required.')
        if quota:
            selected.extend(members[farthest_point_order(features[members], quota)].tolist())
            ranks.extend(range(quota))
        details.append({'level': level, 'pool': len(members), 'selected': quota})
    return np.asarray(selected, dtype=np.int64), np.asarray(ranks, dtype=np.int64), details


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_geometry_h5(path, coordinates, numbers, ids, pool_indices, strata, ranks):
    with h5py.File(path, 'w') as handle:
        handle.attrs['schema'] = SCHEMA
        handle.attrs['energy_force_labels_present'] = False
        group = handle.create_group('molecule')
        group['species'] = np.asarray([chemical_symbols[z] for z in numbers], dtype='S2')
        group.create_dataset('coordinates', data=coordinates, compression='gzip', shuffle=True)
        group[TOPOLOGY_ATOM_IDS] = ids
        group['_id'] = np.asarray([f'rdkit-kdg-{index:010d}' for index in pool_indices], dtype='S32')
        group['pool_index'], group['strata'], group['fps_rank'] = pool_indices, strata, ranks


def prepare_rdkit_seed(config, output_dir, *, config_directory=None):
    """Prepare and atomically publish a geometry pool, FPS seeds and provenance."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite seed preparation output: {output}')
    settings = resolve_config(config, base_directory=config_directory)
    context = _reference_context(settings)
    reference_hash = _sha256(settings['reference_geometry'])
    coordinates, strata, embedding = generate_pool(settings, context)
    selected, ranks, selection = select_seeds(coordinates, context['numbers'], strata,
                                              settings['seed_size'], len(settings['bond_scales']))
    # Preserve reference identities while following upstream's atomic-number sort.
    order = np.argsort(context['numbers'], kind='stable')
    coordinates, numbers = coordinates[:, order], context['numbers'][order]
    ids = context['native_to_reference'][order]
    pool_ranks = np.full(len(coordinates), -1, dtype=np.int64)
    pool_ranks[selected] = ranks
    if _sha256(settings['reference_geometry']) != reference_hash:
        raise RuntimeError('Reference geometry changed during preparation.')
    builder = {'seed_manifest_path': str(output / 'provenance.json'), 'selection_seed': settings['seed']}
    screen = settings['geometry_screen']
    reference_name = 'reference' + Path(settings['reference_geometry']).suffix.lower()
    topology_config = {'enabled': True, 'reference_conformer_path': str(output / reference_name),
                       'reference_format': 'auto', 'reference_charge': context['reference_charge'],
                       **{key: screen[key] for key in ('bond_min_scale', 'bond_max_scale', 'connectivity_scale')}}
    report = {'schema': SCHEMA, 'status': 'completed', 'settings': settings,
              'configuration_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'reference_sha256': reference_hash, 'implementation_sha256': _sha256(__file__),
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'h5py': h5py.__version__, 'rdkit': _rdkit()[-1]},
              'embedding_settings': {**EMBEDDING_SETTINGS, 'force_field_optimization': False},
              'native_to_reference': context['native_to_reference'].tolist(),
              'storage_to_native': order.tolist(), 'storage_to_reference': ids.tolist(),
              'pool_count': len(coordinates), 'selected_count': len(selected),
              'embedding': embedding, 'selection': selection,
              'selection_method': 'per-stratum greedy max-min FPS, centroid-nearest start',
              'descriptor': 'unscaled all-atom pair distances sorted within element-pair type',
              'alf_configuration': {
                  'master': {'builder_task': 'alframework.builders.rdkit_molecule.rdkit_molecule_builder_task',
                             'builder_config_path': str(output / 'builder_config.json'),
                             'bootstrap_set': len(selected), 'finite_bootstrap': True},
                  'sampler': {'topology_check': topology_config, 'distcut': screen['min_distance_A']},
              }}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        shutil.copyfile(settings['reference_geometry'], temporary / reference_name)
        if _sha256(temporary / reference_name) != reference_hash:
            raise RuntimeError('Reference geometry changed before its snapshot was written.')
        _write_geometry_h5(temporary / 'pool.h5', coordinates, numbers, ids, np.arange(len(coordinates)), strata, pool_ranks)
        _write_geometry_h5(temporary / 'seed.h5', coordinates[selected], numbers, ids, selected, strata[selected], ranks)
        (temporary / 'builder_config.json').write_text(json.dumps(builder, indent=2) + '\n')
        report['artifacts_sha256'] = {name: _sha256(temporary / name) for name in ('pool.h5', 'seed.h5', 'builder_config.json', reference_name)}
        (temporary / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


prepare_rdkit_pool = prepare_rdkit_seed


def build_rdkit_molecules(moleculeids, builder_config, h5_path=None, current_h5_id=0, master_directory=None):
    """Replay FPS bootstrap seeds, then geometry from immutable closed label shards."""
    from alframework.builders.external_geometry import build_external_geometries
    root = Path(master_directory or '.').expanduser().resolve()
    manifest_path = Path(builder_config['seed_manifest_path']).expanduser()
    if not manifest_path.is_absolute():manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'completed':
        raise ValueError('RDKit builder requires a completed preparation manifest')
    seed = manifest_path.parent / 'seed.h5'
    # Hashed once per worker per version of each file, not on every task.
    for name in ('seed.h5', 'reference'+Path(manifest['settings']['reference_geometry']).suffix.lower()):
        if cached_sha256(manifest_path.parent/name) != manifest['artifacts_sha256'][name]:
            raise ValueError('Prepared RDKit input changed: '+name)
    if int(current_h5_id) == 0:
        config = dict(sources=[str(seed)], selection_mode='sequential')
    else:
        if h5_path is None:raise ValueError('Runtime replay requires h5_path')
        paths = []
        for index in range(int(current_h5_id)):
            path = Path(str(h5_path).format(index)).expanduser()
            path = path if path.is_absolute() else root / path
            if not path.is_file():raise FileNotFoundError('Closed label shard missing: '+str(path))
            paths.append(str(path))
        config = dict(sources=paths, selection_mode='deterministic_hash',
                      selection_seed=int(builder_config.get('selection_seed', 42)))
    molecules = build_external_geometries(moleculeids, config, master_directory)
    manifest_sha256 = _sha256(manifest_path)
    for molecule in molecules:
        molecule.update_metadata(dict(rdkit_manifest_sha256=manifest_sha256,
                                      replay_external_bootstrap=int(current_h5_id)==0))
    if int(current_h5_id) == 0:
        with h5py.File(seed, 'r') as handle:
            for molecule in molecules:
                md = molecule.get_metadata()
                group = handle[md['geometry_group']]
                index = md['geometry_frame']
                molecule.update_metadata(dict(rdkit_pool_index=int(group['pool_index'][index]),
                                              rdkit_stratum=group['strata'][index].tolist(),
                                              rdkit_fps_rank=int(group['fps_rank'][index])))
    return molecules


from parsl import python_app
from alframework.tools.task_timeline import timed


@python_app(executors=['alf_sampler_executor'])
@timed('builder')
def rdkit_molecule_builder_task(moleculeid=None, moleculeids=None, builder_config=None,
                                h5_path=None, current_h5_id=0, master_directory=None):
    """Ordinary ALF builder; CPU geometry preparation is explicit and separate."""
    plural = moleculeids is not None
    result = build_rdkit_molecules(moleculeids if plural else [moleculeid], builder_config or {},
                                   h5_path, current_h5_id, master_directory)
    return result if plural else result[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path, help='JSON seed configuration; reference paths are relative to this file.')
    parser.add_argument('--output-dir', required=True, type=Path, help='New output directory; existing paths are never overwritten.')
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text())
    report = prepare_rdkit_seed(config, args.output_dir, config_directory=config_path.parent)
    print(f"Prepared {report['selected_count']} KDG/FPS seeds from {report['pool_count']} accepted pool frames in {args.output_dir}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
