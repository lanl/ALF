"""Generate bare slabs and adsorbed molecules from a pool of bulk materials."""

import hashlib
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil, floor, gcd

import numpy as np
from ase.build import surface
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic, get_layers, minkowski_reduce
from ase.neighborlist import neighbor_list
from parsl import python_app
from scipy.spatial import Delaunay, QhullError

from alframework.builders.surface_inputs import AdsorbateInput, check_neutral, load_adsorbate, read_geometry, source_key
from alframework.builders.surface_symmetry import bulk_operations, slab_has_equivalent_faces
from alframework.tools.molecules import MoleculesObject
from alframework.tools.task_timeline import timed


_IN_PLANE = (True, True, False)
_BULK_INDEX = 'surface_bulk_index'
_SITE_TYPES = ('ontop', 'bridge', 'hollow')
_LENGTH_TOLERANCE = 1e-8


def _length(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError(f'{name} must be a finite {"nonnegative" if allow_zero else "positive"} number')
    return float(value)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _in_plane_bounds(config):
    shared_min = _length(config.get('min_in_plane_A', 8), 'min_in_plane_A')
    bounds = []
    for axis in ('a', 'b'):
        lower_key, upper_key = f'min_{axis}_A', f'max_{axis}_A'
        lower = _length(config.get(lower_key, shared_min), lower_key)
        upper = _length(config[upper_key], upper_key) if upper_key in config else float('inf')
        target_key = f'target_{axis}_A'
        target = _length(config[target_key], target_key) if target_key in config else None
        if lower > upper:
            raise ValueError(f'{lower_key} must not exceed {upper_key}')
        bounds.append((lower, upper, target))
    return tuple(bounds)


def _bulk_geometry(source):
    bulk = read_geometry(source).atoms
    if (not bulk.pbc.all() or not np.isfinite(bulk.cell).all()
            or np.linalg.det(bulk.cell) <= 1e-8):
        raise ValueError(f'{source[0]}: bulk requires a right-handed periodic conventional cell')
    return bulk


def _miller_index(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 3
            or any(type(index) is not int for index in values) or not any(values)):
        raise ValueError('Each miller_indices entry must contain three integers, including a nonzero index')
    divisor = gcd(*values)
    if next(index for index in values if index) < 0:
        divisor = -divisor
    return tuple(index // divisor for index in values)


def _miller_order(miller):
    return max(map(abs, miller)), sum(map(abs, miller)), tuple(-index for index in miller)


@lru_cache(maxsize=32)
def _miller_families(source, maximum, tolerance):
    operations = bulk_operations(_bulk_geometry(source), tolerance)
    candidates = {_miller_index(miller) for miller in product(range(-maximum, maximum + 1), repeat=3)
                  if gcd(*miller) == 1}
    pending = candidates.copy()
    families = []
    for miller in sorted(candidates, key=_miller_order):
        if miller in pending:
            # With fractional row coordinates, x' = x @ operation + t, so
            # operation @ h pulls the plane coefficients back to the original coordinates.
            orbit = {_miller_index(tuple(normal)) for normal in (operations @ miller).tolist()}
            family = tuple(sorted(orbit & pending, key=_miller_order))
            pending.difference_update(family)
            families.append(family)
    return tuple(families)


def _bond_cutoffs(bonds):
    pairs = {}
    for bond in bonds:
        if not isinstance(bond, (tuple, list)) or len(bond) != 3:
            raise ValueError('Each protected_bonds entry must contain two elements and a cutoff')
        first, second, distance = bond
        if any(not isinstance(element, str) or element not in atomic_numbers or element == 'X'
               for element in (first, second)):
            raise ValueError('protected_bonds must name two chemical elements and a cutoff in Angstrom')
        pair = tuple(sorted((first, second)))
        if pair in pairs:
            raise ValueError(f'Duplicate protected bond {pair}')
        pairs[pair] = _length(distance, 'protected bond cutoff')
    return tuple((first, second, distance) for (first, second), distance in sorted(pairs.items()))


def _neighbor_signatures(atoms, first, second):
    """Count protected neighbors by original bulk index, including periodic images."""
    signatures = [Counter() for _ in atoms]
    origin = atoms.arrays[_BULK_INDEX]
    for i, j in zip(first, second):
        signatures[i][origin[j]] += 1
    return signatures


def _termination_shifts(phases, bond_starts, bond_ends):
    """Return midpoints of gaps between planes that do not cross protected bonds."""
    levels = np.sort(phases % 1)
    levels = levels[np.r_[True, np.diff(levels) > 1e-8]]
    gaps = np.diff(np.r_[levels, levels[0] + 1])
    for shift in ((levels + gaps / 2) % 1)[gaps > 1e-8]:
        # A bond crosses the cut if its unwrapped endpoints lie in different repeats.
        if np.all(np.floor(bond_starts - shift) == np.floor(bond_ends - shift)):
            yield float(shift)


def _nearest_repeat(length, target, lower, upper):
    ideal = target / length
    candidates = (min(upper, max(lower, repeat)) for repeat in (floor(ideal), ceil(ideal)))
    return min(candidates, key=lambda repeat: (round(abs(length * repeat - target), 8), repeat))


def _in_plane_repeats(lengths, bounds, substrate_atoms, atom_limit):
    ranges, targets = [], []
    for axis, length, (lower, upper, target) in zip(('a', 'b'), lengths, bounds):
        repeat = max(1, ceil((lower - _LENGTH_TOLERANCE) / length))
        final_length = repeat * length
        if final_length > upper + _LENGTH_TOLERANCE:
            raise ValueError(f'|{axis}| cannot fit [{lower:g}, {upper:g}] Å: '
                             f'the smallest repeat meeting the minimum is {final_length:g} Å')
        if target is None:
            ranges.append((repeat, repeat))
            targets.append(final_length)
        else:
            maximum = max(repeat, ceil(target / length))
            if np.isfinite(upper):
                maximum = min(maximum, floor((upper + _LENGTH_TOLERANCE) / length))
            ranges.append((repeat, maximum))
            targets.append(target)

    nearest = tuple(_nearest_repeat(length, target, *limits)
                    for length, target, limits in zip(lengths, targets, ranges))
    max_atoms, reserved = atom_limit
    if max_atoms is None or nearest[0] * nearest[1] * substrate_atoms + reserved <= max_atoms:
        return nearest
    (min_a, max_a), (min_b, max_b) = ranges
    minimum_atoms = min_a * min_b * substrate_atoms + reserved
    if minimum_atoms > max_atoms:
        raise ValueError(f'{minimum_atoms} atoms including {reserved} reserved for the adsorbate '
                         f'exceed max_atoms={max_atoms}')
    budget = (max_atoms - reserved) // substrate_atoms
    length_a, length_b = lengths
    target_a, target_b = targets
    # For each feasible a repeat, b has the closest feasible value for its remaining atom budget.
    candidates = ((a, _nearest_repeat(length_b, target_b, min_b, min(max_b, budget // a)))
                  for a in range(min_a, min(max_a, budget // min_b) + 1))

    def rank(repeats):
        a, b = repeats
        error = abs(length_a * a - target_a) + abs(length_b * b - target_b)
        return round(error, 8), a * b, a, b

    return min(candidates, key=rank)


def _facet_slabs(bulk, miller, thickness, bounds, atom_limit, bonds, symmetry, independent):
    scaled, bond_starts, bond_ends, cutoffs, expected = bonds
    phases = scaled @ miller
    spacing = 1 / np.linalg.norm(bulk.cell.reciprocal().T @ miller)
    shifts = tuple(_termination_shifts(phases, bond_starts @ miller, bond_ends @ miller))
    symmetry_mode, tolerance = symmetry
    slabs = []
    rejected = set()
    bulk_counts = Counter(bulk.numbers)
    best_thickness = float('inf')
    for shift in shifts:
        translated = bulk.copy()
        translated.set_scaled_positions(scaled - shift * np.asarray(miller) / np.dot(miller, miller))
        unit = surface(translated, miller, 1, periodic=True)
        stacks = {1: unit}
        upper_shifts = shifts if independent else (shift,)
        for upper in upper_shifts:
            fraction = (upper - shift) % 1 if upper != shift else 1
            prefix = unit.positions[:, 2] < fraction * spacing if upper != shift else slice(None)
            prefix_atoms = unit.numbers[prefix]
            count = len(prefix_atoms)
            if not count:
                continue
            if upper != shift:
                counts = Counter(prefix_atoms)
                # Adding full repeats cannot restore stoichiometry if the partial repeat changes it.
                if any(counts[number] * len(bulk) != count * n for number, n in bulk_counts.items()):
                    rejected.add('the slab does not preserve bulk stoichiometry')
                    continue
            occupied = np.ptp(unit.positions[prefix, 2])
            repeats = max(0, ceil((thickness - occupied - _LENGTH_TOLERANCE) / spacing))
            occupied += repeats * spacing
            if independent and occupied > best_thickness + _LENGTH_TOLERANCE:
                continue
            try:
                a, b = _in_plane_repeats(unit.cell.lengths()[:2], bounds, repeats * len(unit) + count, atom_limit)
            except ValueError as exc:
                rejected.add(str(exc))
                continue
            # ASE builds the full stack so lateral offsets between repeats are preserved.
            layers = repeats + 1
            if layers not in stacks:
                stacks[layers] = surface(translated, miller, layers, periodic=True)
            slab = stacks[layers]
            if upper != shift:
                slab = slab[slab.positions[:, 2] < (repeats + fraction) * spacing]
            slab = slab.repeat((a, b, 1))
            slab.pbc = _IN_PLANE
            if cutoffs:
                left, right = neighbor_list('ij', slab, cutoffs)
                actual = _neighbor_signatures(slab, left, right)
                if any(actual[i] != expected[j] for i, j in enumerate(slab.arrays[_BULK_INDEX])):
                    rejected.add('the slab loses protected neighbors')
                    continue
            del slab.arrays[_BULK_INDEX]
            symmetric = slab_has_equivalent_faces(slab, tolerance) if symmetry_mode != 'off' else None
            if symmetry_mode == 'require' and not symmetric:
                rejected.add("top and bottom faces are inequivalent (slab_symmetry='require')")
                continue
            if independent and occupied < best_thickness - _LENGTH_TOLERANCE:
                slabs.clear()
                best_thickness = occupied
            upper_cut = shift + repeats + fraction if independent else None
            slabs.append((miller, shift, slab, symmetric, upper_cut))
    return tuple(slabs), tuple(sorted(rejected or {'check protected_bonds'}))


@lru_cache(maxsize=32)
def _slabs(source, families, thickness, bounds, bonds, atom_limit, automatic, symmetry, independent):
    bulk = _bulk_geometry(source).copy()
    bulk.new_array(_BULK_INDEX, np.arange(len(bulk)))
    scaled = bulk.get_scaled_positions()
    bulk.set_scaled_positions(scaled)
    cutoffs = {(first, second): distance for first, second, distance in bonds}
    first, second, offsets = (neighbor_list('ijS', bulk, cutoffs) if cutoffs else
                              (np.empty(0, int), np.empty(0, int), np.empty((0, 3), int)))
    if cutoffs:
        symbols = bulk.get_chemical_symbols()
        matched = {tuple(sorted((symbols[i], symbols[j]))) for i, j in zip(first, second)}
        if missing := cutoffs.keys() - matched:
            raise ValueError(f'{source[0]}: no neighbors matched protected_bonds {sorted(missing)}')
    expected = _neighbor_signatures(bulk, first, second) if cutoffs else None
    bond_data = scaled, scaled[first], scaled[second] + offsets, cutoffs, expected
    result, skipped = [], []
    for family in families:
        rejected = set()
        for miller in family:
            slabs, reasons = _facet_slabs(bulk, miller, thickness, bounds, atom_limit, bond_data, symmetry, independent)
            if slabs:
                result.append(slabs)
                break
            rejected.update(reasons)
        else:
            detail = '; '.join(sorted(rejected))
            if not automatic:
                raise ValueError(f'{source[0]}: no intact slab for Miller {family[0]}; {detail}')
            skipped.append(f'{family[0]}: {detail}')
    if not result:
        raise ValueError(f'{source[0]}: no facet families survived max_miller. ' + ' | '.join(skipped))
    if skipped:
        warnings.warn(f'{source[0]}: skipped facet families: ' + ' | '.join(skipped), stacklevel=2)
    return tuple(result) if automatic else tuple(slab for group in result for slab in group)


def _flip_slab(slab):
    """Expose the other face using a proper rotation and a right-handed cell."""
    slab.positions[:, 1:] *= -1
    cell = np.asarray(slab.cell).copy()
    cell[:, 1:] *= -1
    cell[1:] *= -1
    slab.set_cell(cell)


class _SurfaceSites:
    """A periodic 2D mesh, with Cartesian heights interpolated from the exposed atoms."""

    def __init__(self, slab, top):
        xy_cell = minkowski_reduce(slab.cell, pbc=_IN_PLANE)[0][:2, :2]
        self.cell = xy_cell
        self.inverse = np.linalg.inv(xy_cell)
        # Keep the highest atom if several exposed atoms share a projected position.
        uv = top[:, :2] @ self.inverse
        uv = np.round(uv % 1, decimals=8) % 1
        by_height = np.argsort(-top[:, 2], kind='stable')
        _, unique = np.unique(uv[by_height], axis=0, return_index=True)
        highest = by_height[unique]
        self.ontop = np.column_stack((uv[highest] @ xy_cell, top[highest, 2]))
        offsets = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]) @ xy_cell
        offsets = np.column_stack((offsets, np.zeros(len(offsets))))
        self.points = (self.ontop[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        try:
            self.mesh = Delaunay(self.points[:, :2])
        except QhullError as exc:
            raise ValueError('Could not triangulate the exposed surface atoms') from exc

    def _within_cell(self, sites):
        uv = np.round(sites[:, :2] @ self.inverse, decimals=8)
        sites = sites[((uv >= 0) & (uv < 1)).all(axis=1)]
        _, indices = np.unique(np.round(sites, decimals=8), axis=0, return_index=True)
        return sites[indices]

    def named(self, site_types):
        triangles = self.points[self.mesh.simplices]
        sites = {'ontop': self.ontop}
        if 'bridge' in site_types:
            sites['bridge'] = self._within_cell(((triangles + np.roll(triangles, 1, axis=1)) / 2).reshape(-1, 3))
        if 'hollow' in site_types:
            sites['hollow'] = self._within_cell(triangles.mean(axis=1))
        return [(kind, point) for kind in site_types for point in sites[kind]]

    def random(self, rng):
        # Uniform in the in-plane cell is area-weighted over its projected triangles.
        xy = rng.random(2) @ self.cell
        index = int(self.mesh.find_simplex(xy))
        if index < 0:
            raise ValueError('Surface triangulation does not cover the in-plane cell')
        transform = self.mesh.transform[index]
        weights = transform[:2] @ (xy - transform[2])
        z = np.r_[weights, 1 - weights.sum()] @ self.points[self.mesh.simplices[index], 2]
        return 'random', np.r_[xy, z]


def _oriented(geometry, anchor, max_tilt, rng):
    adsorbate = geometry.copy()
    adsorbate.translate(-adsorbate.positions[anchor].copy())
    if len(adsorbate) > 1:
        vector = adsorbate.positions.mean(axis=0)
        if np.linalg.norm(vector) < 1e-8:
            vector = adsorbate.positions[np.argmax(np.linalg.norm(adsorbate.positions, axis=1))]
        if np.linalg.norm(vector) < 1e-8:
            raise ValueError('Adsorbate atoms coincide at the binding anchor')
        adsorbate.rotate(vector, 'z')
        adsorbate.rotate(rng.uniform(0, 360), 'z')
        tilt = np.degrees(np.arccos(rng.uniform(np.cos(np.radians(max_tilt)), 1)))
        azimuth = rng.uniform(0, 2 * np.pi)
        adsorbate.rotate(tilt, (np.cos(azimuth), np.sin(azimuth), 0))
    return adsorbate


def _place(slab, adsorbate, site, gap):
    adsorbate.translate(site)
    differences = adsorbate.positions[:, None, :] - slab.positions
    lateral = differences.copy()
    lateral[:, :, 2] = 0
    distances = find_mic(lateral.reshape(-1, 3), slab.cell, pbc=_IN_PLANE)[1].reshape(lateral.shape[:2])
    radii = covalent_radii[adsorbate.numbers, None] + covalent_radii[slab.numbers]
    needed_z2 = (radii + gap) ** 2 - distances ** 2
    close = needed_z2 > 0
    # Lift the whole molecule along the surface normal without distorting internal bonds.
    lift = np.max(np.sqrt(needed_z2[close]) - differences[:, :, 2][close], initial=gap)
    adsorbate.translate((0, 0, lift))
    adsorbate.set_tags(np.full(len(adsorbate), 2))
    return slab + adsorbate


def _has_image_clash(atoms):
    radii = covalent_radii[atoms.numbers]
    adsorbate = atoms.get_tags() == 2
    if atoms.cell[2, 2] - np.ptp(atoms.positions[:, 2]) >= 2 * radii.max():
        # With sufficient vertical vacuum, only the adsorbate's lateral images need checking.
        if not adsorbate.any():
            return False
        atoms = atoms[adsorbate]
        atoms.pbc = _IN_PLANE
        radii = radii[adsorbate]
        adsorbate = np.ones(len(atoms), dtype=bool)
    i, j, distance, shift = neighbor_list('ijdS', atoms, 2 * radii.max())
    images = (shift[:, 2] != 0) | (shift.any(axis=1) & adsorbate[i] & adsorbate[j])
    return bool(np.any(images & (distance < radii[i] + radii[j] - 1e-8)))


def _finish_cell(atoms, vacuum):
    shift = vacuum / 2 - atoms.positions[:, 2].min()
    atoms.center(vacuum=vacuum / 2, axis=2)
    atoms.pbc = True
    return shift


@dataclass(frozen=True)
class _Placement:
    geometry: AdsorbateInput
    anchors: tuple
    mode: str
    site_types: tuple
    max_tilt: float
    gap: float
    surface_elements: tuple


@dataclass(frozen=True)
class _Material:
    name: str | None
    weight: float
    source: str
    charge_sources: tuple
    slabs: tuple
    automatic: bool
    face: str
    depth: float
    vacuum: float
    fixed_bottom_layers: int
    adsorbates: tuple


def _prepare_placement(entry, master_directory, seed):
    geometry = load_adsorbate(entry, master_directory, seed)
    label = geometry.source
    anchors = entry.get('anchor_indices', [])
    if (not isinstance(anchors, (list, tuple)) or not anchors
            or any(type(index) is not int or not 0 <= index < geometry.anchor_count for index in anchors)):
        raise ValueError(f'{label}: anchor_indices must contain valid zero-based input atom indices')
    mode = entry.get('mode', 'random')
    if mode not in ('random', 'binding_sites'):
        raise ValueError('Adsorbate mode must be random or binding_sites')
    kinds = tuple(entry.get('site_types', _SITE_TYPES))
    if mode == 'binding_sites' and (not kinds or any(kind not in _SITE_TYPES for kind in kinds)):
        raise ValueError('site_types must select from ontop, bridge, and hollow')
    max_tilt = _length(entry.get('max_tilt_deg', 15), 'max_tilt_deg', allow_zero=True)
    if max_tilt > 90:
        raise ValueError('max_tilt_deg must not exceed 90 degrees')
    gap = _length(entry.get('gap_A', 0.1), 'gap_A', allow_zero=True)
    elements = entry.get('surface_elements', [])
    if (not isinstance(elements, (list, tuple))
            or any(not isinstance(symbol, str) or symbol not in atomic_numbers or symbol == 'X'
                   for symbol in elements)):
        raise ValueError(f'{label}: surface_elements must be a list of chemical element symbols')
    return _Placement(geometry, tuple(anchors), mode, kinds, max_tilt, gap, tuple(elements))


def _prepare_material(config, master_directory, seed, name=None, weight=1.0):
    source = source_key(config.get('bulk_file'), master_directory)
    charge = check_neutral(read_geometry(source).charges, config, 'bulk_charge', source[0])
    automatic = 'max_miller' in config
    if automatic == ('miller_indices' in config):
        raise ValueError('Specify exactly one of max_miller and miller_indices')
    symmetry = config.get('slab_symmetry', 'off')
    if symmetry not in ('off', 'report', 'require'):
        raise ValueError('slab_symmetry must be off, report, or require')
    tolerance = _length(config.get('symmetry_tolerance_A', 0.001), 'symmetry_tolerance_A')
    if automatic:
        maximum = _positive_integer(config['max_miller'], 'max_miller')
        families = _miller_families(source, maximum, tolerance)
    else:
        millers = tuple(dict.fromkeys(_miller_index(values) for values in config['miller_indices']))
        if not millers:
            raise ValueError('At least one entry in miller_indices is required')
        families = tuple((miller,) for miller in millers)

    entries = config.get('adsorbates', [])
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('adsorbates must be a list of objects')
    adsorbates = tuple(_prepare_placement(entry, master_directory, seed) for entry in entries)
    max_atoms, reserved = None, 0
    if 'max_atoms' in config:
        max_atoms = _positive_integer(config['max_atoms'], 'max_atoms')
        reserved = max((len(placement.geometry.atoms) for placement in adsorbates), default=0)
    face = config.get('face', 'top')
    if face not in ('top', 'bottom', 'both'):
        raise ValueError('face must be top, bottom, or both')
    depth = _length(config.get('surface_depth_A', 2), 'surface_depth_A')
    vacuum = _length(config.get('vacuum_A', 15), 'vacuum_A')
    fixed_bottom_layers = config.get('fixed_bottom_layers', 0)
    if type(fixed_bottom_layers) is not int or fixed_bottom_layers < 0:
        raise ValueError('fixed_bottom_layers must be a nonnegative integer')
    independent = config.get('allow_independent_terminations', False)
    if type(independent) is not bool:
        raise ValueError('allow_independent_terminations must be Boolean')
    slabs = _slabs(source, families, _length(config.get('min_slab_thickness_A', 10), 'min_slab_thickness_A'),
                   _in_plane_bounds(config), _bond_cutoffs(config.get('protected_bonds', [])),
                   (max_atoms, reserved), automatic, (symmetry, tolerance), independent)
    return _Material(name, weight, source[0], charge, slabs, automatic, face, depth, vacuum,
                     fixed_bottom_layers, adsorbates)


def _material_pools(config, master_directory, seed):
    if 'materials' not in config:
        return (_prepare_material(config, master_directory, seed),)
    entries = config['materials']
    if 'bulk_file' in config or not isinstance(entries, list) or not entries:
        raise ValueError('Provide a nonempty materials list or one top-level bulk_file')
    defaults = {key: value for key, value in config.items() if key != 'materials'}
    seen, pools = set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('materials entries must be objects')
        name = entry.get('id')
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError('materials entries require unique nonempty string IDs')
        seen.add(name)
        weight = _length(entry.get('weight', 1), f'material {name!r} weight')
        settings = defaults.copy()
        if 'miller_indices' in entry or 'max_miller' in entry:
            for key in ('miller_indices', 'max_miller'):
                settings.pop(key, None)
        settings.update(entry)
        try:
            pools.append(_prepare_material(settings, master_directory, seed, name, weight))
        except (ValueError, OSError, ImportError) as exc:
            raise type(exc)(f'Material {name!r}: {exc}') from exc
    return tuple(pools)


def _charge_metadata(sources):
    return dict(net=0, assumed=not sources, sources=list(sources))


def build_surface(moleculeid, builder_config, master_directory=None):
    """Build one slab or adsorbate geometry deterministically from its ALF ID."""
    if not isinstance(moleculeid, str) or not moleculeid:
        raise ValueError('A nonempty string moleculeid is required')
    seed = builder_config.get('seed', 0)
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(f'{seed}|{moleculeid}'.encode()).digest()[:16], 'big'))
    materials = _material_pools(builder_config, master_directory, seed)
    material = materials[0]
    if len(materials) > 1:
        weights = np.array([item.weight for item in materials])
        weights /= weights.max()
        material = materials[rng.choice(len(materials), p=weights / weights.sum())]
    slabs = material.slabs
    if material.automatic:
        slabs = slabs[rng.integers(len(slabs))]
    miller, cut, template, symmetric, upper_cut = slabs[rng.integers(len(slabs))]
    slab = template.copy()
    face = material.face
    if face == 'both':
        face = ('top', 'bottom')[rng.integers(2)]
    if face == 'bottom':
        _flip_slab(slab)
    fixed_indices = []
    if material.fixed_bottom_layers:
        layers, heights = get_layers(slab, (0, 0, 1))
        if material.fixed_bottom_layers >= len(heights):
            raise ValueError(f'{material.source}: fixed_bottom_layers must leave a mobile substrate layer '
                             f'for Miller {miller} ({len(heights)} layers)')
        fixed_indices = np.flatnonzero(layers < material.fixed_bottom_layers).tolist()
    exposed = slab.positions[:, 2] >= slab.positions[:, 2].max() - material.depth
    slab.set_tags(exposed.astype(int))
    metadata = dict(bulk_file=material.source, miller_index=list(miller), termination_shift=cut, face=face,
                    bulk_charge=_charge_metadata(material.charge_sources))
    if upper_cut is not None:
        metadata['upper_termination_shift'] = upper_cut
    if material.name is not None:
        metadata['material_id'] = material.name
    if symmetric is not None:
        metadata['slab_symmetric'] = symmetric
    if material.adsorbates:
        entry = material.adsorbates[rng.integers(len(material.adsorbates))]
        slab, placement = _add_adsorbate(slab, exposed, entry, material.vacuum, rng, metadata)
        metadata.update(placement)
    else:
        _finish_cell(slab, material.vacuum)
        if _has_image_clash(slab):
            raise ValueError(f'{material.source}: increase vacuum_A for Miller {miller} to avoid periodic overlap')
    molecule = MoleculesObject(slab, moleculeid)
    molecule.update_metadata({'surface_builder': metadata, 'vacuum_axes': ['c']})
    if fixed_indices:
        slab.set_constraint(FixAtoms(indices=fixed_indices))
        molecule.update_metadata({'fixed_substrate_indices': fixed_indices})
    return molecule


def _add_adsorbate(slab, exposed, entry, vacuum, rng, surface_metadata):
    geometry = entry.geometry
    eligible = exposed
    if entry.surface_elements:
        eligible = eligible & np.isin(slab.get_chemical_symbols(), entry.surface_elements)
    if not eligible.any():
        raise ValueError(f'No exposed atoms match surface_elements for {surface_metadata}')
    mesh = _SurfaceSites(slab, slab.positions[eligible])
    named = mesh.named(entry.site_types) if entry.mode == 'binding_sites' else None
    if named == []:
        raise ValueError(f'No {entry.site_types} binding sites found for {surface_metadata}')
    for _ in range(24):
        kind, site = mesh.random(rng) if named is None else named[rng.integers(len(named))]
        anchor = int(rng.choice(entry.anchors))
        combined = _place(slab, _oriented(geometry.atoms, anchor, entry.max_tilt, rng), site, entry.gap)
        shift = _finish_cell(combined, vacuum)
        if not _has_image_clash(combined):
            metadata = {f'adsorbate_{geometry.source_kind}': geometry.source,
                        'adsorbate_charge': _charge_metadata(geometry.charge_sources),
                        'mode': entry.mode, 'site_type': kind, 'site': (site + [0, 0, shift]).tolist(),
                        'anchor_index': anchor, 'anchor_atom_index': len(slab) + anchor}
            if geometry.force_field is not None:
                metadata['adsorbate_force_field'] = geometry.force_field
            return combined, metadata
    raise ValueError(f'{geometry.source}: no placement clears its periodic images for {surface_metadata}; '
                     'increase the in-plane dimensions or vacuum_A')


@python_app(executors=['alf_sampler_executor'])
@timed('builder')
def surface_builder_task(moleculeid, builder_config, master_directory=None):
    """Return one unlabeled slab candidate using ALF's standard builder interface."""
    return build_surface(moleculeid, builder_config, master_directory)
