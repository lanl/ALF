"""Greedy selection in the whitened energy tangent space of one frozen model."""

from collections.abc import Mapping

import numpy as np


def energy_ntk_row(payload):
    """Return the validated basis identity, ridge, and whitened candidate row."""
    if not isinstance(payload, Mapping):
        raise ValueError("Adaptive NTK candidates must carry an energy_ntk row.")
    basis, ridge, row = (payload.get(key) for key in ("basis_id", "ridge", "row"))
    if not isinstance(basis, str) or not basis:
        raise ValueError("An energy_ntk row requires a basis ID.")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("An energy_ntk ridge must be finite and positive.")
    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1 or not row.size or not np.isfinite(row).all():
        raise ValueError("An energy_ntk row must be a finite, nonempty vector.")
    return basis, float(ridge), row


def initial_energy_score(candidate):
    _, ridge, row = energy_ntk_row(candidate.get("energy_ntk"))
    score = ridge * float(row @ row)
    if not np.isfinite(score):
        raise ValueError("Nonfinite initial energy NTK score.")
    return score


def select_ntk_candidates(candidates, k, *, adaptive=True):
    """Rank a fixed pool with scores from a common frozen energy-only basis.

    Let L factor the fitted regularized information matrix. With whitened
    rows z = L^-1 phi, the information matrix starts at identity. Adding a
    selected geometry updates it by z z.T, without needing its reference label.
    The score is ridge * z.T @ inverse @ z. Return copies with audit metadata.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("The NTK selection budget must be a positive integer.")
    ordered = sorted(candidates, key=_identity)
    if not ordered:
        return []
    if len({_identity(candidate) for candidate in ordered}) != len(ordered):
        raise ValueError("Adaptive NTK candidate IDs must be unique within a pool.")
    payloads = [energy_ntk_row(candidate.get("energy_ntk")) for candidate in ordered]
    basis, ridge, first = payloads[0]
    if any((b, r, len(z)) != (basis, ridge, len(first)) for b, r, z in payloads):
        raise ValueError("Adaptive NTK candidates must share one fitted basis, ridge, and sketch width.")
    rows = np.stack([z for _, _, z in payloads])
    initial = ridge * np.einsum("ij,ij->i", rows, rows)
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial energy NTK scores.")
    scores = initial.copy()
    selected = np.zeros(len(rows), dtype=bool)
    output = []
    information = np.eye(rows.shape[1], dtype=np.float64)
    inverse = information.copy()
    tolerance = 1e-10 * max(1.0, float(initial.max()))

    def refactor():
        factor = np.linalg.cholesky(information)
        whitener = np.linalg.solve(factor, np.eye(len(factor)))
        whitened = rows @ whitener.T
        return whitener.T @ whitener, ridge * np.einsum("ij,ij->i", whitened, whitened)

    budget = min(k, len(ordered))
    for rank in range(budget):
        index = int(np.argmax(np.where(selected, -np.inf, scores)))
        candidate = ordered[index]
        output.append({**candidate, "adaptive_ntk": {
            "basis_id": basis, "initial_score": float(initial[index]),
            "selection_score": float(scores[index]), "selection_order": rank,
        }})
        selected[index] = True
        if not adaptive or rank + 1 == budget:
            continue

        row = rows[index]
        information += np.outer(row, row)
        direction = inverse @ row
        denominator = 1.0 + row @ direction
        refactor_needed = (rank + 1) % 32 == 0 or not np.isfinite(denominator) or denominator <= 0
        if not refactor_needed:
            cross = rows @ direction
            scores -= (ridge / denominator) * cross**2
            inverse -= np.outer(direction, direction) / denominator
            active = scores[~selected]
            refactor_needed = (not np.isfinite(inverse).all() or not np.isfinite(active).all()
                               or np.any(active < -tolerance))
        if refactor_needed:
            inverse, scores = refactor()
            if not np.isfinite(inverse).all() or not np.isfinite(scores).all():
                raise FloatingPointError("Could not recover finite adaptive NTK scores.")
        np.maximum(scores, 0.0, out=scores)
    return output


def _identity(candidate):
    return str(candidate["trajectory_id"]), int(candidate["step"])
