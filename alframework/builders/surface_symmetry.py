"""Geometric symmetry operations used for facet and slab selection."""

from itertools import chain, product
from math import ceil, floor, sqrt

import numpy as np
from ase.geometry import minkowski_reduce
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


def _vectors_of_length(triangular, length, tolerance):
    """Enumerate lattice vectors in a thin shell using the triangular basis."""
    dimension = len(triangular)
    vector = np.zeros(dimension, dtype=int)
    lower = max(0.0, length - tolerance) ** 2
    upper = (length + tolerance) ** 2

    def visit(axis, used):
        remaining = upper - used
        if remaining < 0:
            return
        diagonal = triangular[axis, axis]
        offset = triangular[axis, axis + 1:] @ vector[axis + 1:]
        center = -offset / diagonal
        radius = sqrt(remaining) / abs(diagonal)
        first, last = ceil(center - radius), floor(center + radius)
        choices = range(first, last + 1)
        if axis == 0 and used < lower:
            # Skip the interior of the shell instead of enumerating it.
            inner = sqrt(lower - used) / abs(diagonal)
            choices = chain(range(first, min(last, floor(center - inner)) + 1),
                            range(max(first, ceil(center + inner)), last + 1))
        for index in choices:
            vector[axis] = index
            if axis == 0:
                yield vector.copy()
            else:
                yield from visit(axis - 1, used + (diagonal * index + offset) ** 2)

    yield from visit(dimension - 1, 0.0)


def _lattice_operations(basis, tolerance):
    """Find unimodular operations preserving basis lengths and angles."""
    basis = np.asarray(basis)
    triangular = np.linalg.qr(basis.T)[1]
    lengths = np.linalg.norm(basis, axis=1)
    sums = np.linalg.norm(basis[:, None] + basis, axis=2)
    differences = np.linalg.norm(basis[:, None] - basis, axis=2)
    length_options = [[(row, row @ basis) for row in _vectors_of_length(triangular, length, tolerance)]
                      for length in lengths]

    def assemble(rows, cartesian):
        index = len(rows)
        if index == len(basis):
            matrix = np.array(rows)
            if abs(round(np.linalg.det(matrix))) == 1:
                yield matrix
            return
        for row, vector in length_options[index]:
            if all(abs(np.linalg.norm(vector + other) - sums[index, j]) <= tolerance
                   and abs(np.linalg.norm(vector - other) - differences[index, j]) <= tolerance
                   for j, other in enumerate(cartesian)):
                yield from assemble([*rows, row], [*cartesian, vector])

    yield from assemble([], [])


class _PeriodicPointMatcher:
    """Match species in a reduced cell; only wrap dimensions that are periodic."""

    def __init__(self, positions, numbers, cell, pbc, tolerance):
        self.cell = cell
        self.inverse = np.linalg.inv(cell)
        self.pbc = np.asarray(pbc)
        self.numbers = numbers
        self.tolerance = tolerance
        self.positions = self._wrap(positions)
        shifts = np.array(list(product(*[(-1, 0, 1) if flag else (0,) for flag in pbc]))) @ cell
        tiled = (self.positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        self.tree = cKDTree(tiled)
        _, inverse, counts = np.unique(numbers, return_inverse=True, return_counts=True)
        rare = np.flatnonzero(counts[inverse] == counts.min())
        self.anchor = rare[np.argmax(self.positions[rare, 2])] if not self.pbc[2] else rare[0]
        self.targets = self.positions[numbers == numbers[self.anchor]]

    def _wrap(self, positions):
        scaled = positions @ self.inverse
        scaled[:, self.pbc] %= 1.0
        return scaled @ self.cell

    def matches(self, transformed):
        for translation in self.targets - transformed[self.anchor]:
            if not self.pbc[2]:
                if abs(translation[2]) > self.tolerance:
                    continue
                translation[2] = 0  # The face swap must stay within one slab.
            if self._has_bijection(self._wrap(transformed + translation)):
                return True
        return False

    def _has_bijection(self, positions):
        nearby = self.tree.query_ball_point(positions, self.tolerance)
        size = len(self.numbers)
        options = []
        for number, indices in zip(self.numbers, nearby):
            matches = {index % size for index in indices if number == self.numbers[index % size]}
            if not matches:
                return False
            options.append(matches)
        if all(len(matches) == 1 for matches in options):
            return len(set.union(*options)) == size
        edges = [(i, j) for i, matches in enumerate(options) for j in matches]
        rows, columns = zip(*edges)
        graph = csr_matrix((np.ones(len(edges), dtype=bool), (rows, columns)), shape=(size, size))
        return bool(np.all(maximum_bipartite_matching(graph) != -1))


def bulk_operations(atoms, tolerance):
    """Return validated fractional row-vector operations in the input basis."""
    cell, transform = minkowski_reduce(atoms.cell)
    inverse = np.rint(np.linalg.inv(transform)).astype(int)
    scaled = np.linalg.solve(cell.T, atoms.positions.T).T % 1
    matcher = _PeriodicPointMatcher(atoms.positions, atoms.numbers, cell, (True, True, True), tolerance)
    identity = np.eye(3, dtype=int)
    operations = [inverse @ op @ transform for op in _lattice_operations(cell, tolerance)
                  if not np.array_equal(op, identity) and matcher.matches(scaled @ op @ cell)]
    return np.array([identity, *operations])


def slab_has_equivalent_faces(atoms, tolerance):
    """Check for a face swap of the bare ASE slab, without periodicity through vacuum."""
    pbc = (True, True, False)
    cell, _ = minkowski_reduce(atoms.cell, pbc=pbc)
    xy_basis = cell[:2, :2]
    positions = atoms.positions.copy()
    z = positions[:, 2]
    z -= (z.max() + z.min()) / 2
    scaled_xy = np.linalg.solve(xy_basis.T, positions[:, :2].T).T
    matcher = _PeriodicPointMatcher(positions, atoms.numbers, cell, pbc, tolerance)
    return any(matcher.matches(np.column_stack((scaled_xy @ op @ xy_basis, -z)))
               for op in _lattice_operations(xy_basis, tolerance))
