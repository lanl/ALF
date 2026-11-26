"""The sketched information matrix of a fit, its ridge, and posterior variances."""

from collections import Counter
import hashlib
import json
import math

import numpy as np
import torch


class Posterior:
    """A = Σ rowsᵀ rows / scale² over the training rows, and the factor of A + λI.

    The variance of a quantity with sketched row r is λ rᵀ (A + λI)⁻¹ r. In the noiseless limit this is
    the squared distance of r from the span of the training rows. Keep A to change λ later with
    :meth:`with_ridge`, for example when calibrating on a validation set.
    """

    def __init__(self, width, *, device="cpu", dtype=torch.float64):
        self.width = width
        self.information = torch.zeros((width, width), device=device, dtype=dtype)
        self.counts = Counter()
        self.ridge = None
        self._factor = None

    @property
    def device(self):
        return self.information.device

    @property
    def dtype(self):
        return self.information.dtype

    def add(self, rows, scale, kind):
        """Add training rows ``[..., p]`` observed with standard deviation ``scale``."""
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"The {kind} scale must be finite and positive.")
        flat = rows.reshape(-1, self.width) / scale
        if not torch.isfinite(flat).all():
            raise FloatingPointError(f"Nonfinite {kind} NTK rows.")
        self.information.addmm_(flat.T, flat)
        self.counts[kind] += len(flat)
        self._factor = None

    def finalize(self, *, ridge=None, ridge_factor=1e-3):
        """Fix λ, either given or as ``ridge_factor`` times the mean diagonal of A, and factor A + λI."""
        if not sum(self.counts.values()):
            raise ValueError("Supply at least one training geometry.")
        if ridge is None:
            if not math.isfinite(ridge_factor) or ridge_factor <= 0:
                raise ValueError("ridge_factor must be finite and positive.")
            ridge = ridge_factor * float(self.information.diagonal().mean())
        ridge = float(ridge)
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("The NTK ridge is not finite and positive. Check model sensitivities and scales.")
        regularized = self.information.clone()
        regularized.diagonal().add_(ridge)
        factor, status = torch.linalg.cholesky_ex(regularized)
        if int(status) != 0 or not torch.isfinite(factor).all():
            raise ValueError("Could not factor the NTK information matrix. Check scales, ridge, and precision.")
        self.ridge, self._factor = ridge, factor
        return self

    def with_ridge(self, ridge):
        """The same information matrix with another ridge."""
        other = Posterior(self.width, device=self.device, dtype=self.dtype)
        other.information = self.information
        other.counts = Counter(self.counts)
        return other.finalize(ridge=ridge)

    def to(self, device=None, dtype=None):
        other = Posterior(self.width, device=device or self.device, dtype=dtype or self.dtype)
        other.information = self.information.to(device=other.device, dtype=other.dtype)
        other.counts = Counter(self.counts)
        return other.finalize(ridge=self.ridge)

    @property
    def factor(self):
        if self._factor is None:
            raise RuntimeError("Finalize the posterior before scoring.")
        return self._factor

    def whiten(self, rows):
        """Return L⁻¹ r for rows ``[..., p]``, where L Lᵀ = A + λI."""
        flat = rows.reshape(-1, self.width).to(self.factor.dtype)
        whitened = torch.linalg.solve_triangular(self.factor, flat.T, upper=False).T
        return whitened.reshape(rows.shape)

    def variance(self, rows):
        """Return λ rᵀ (A + λI)⁻¹ r for rows ``[..., p]``, with shape ``[...]``."""
        return self.ridge * self.whiten(rows).square().sum(-1)

    def basis_id(self, metadata):
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(self.information.detach().cpu().numpy()).tobytes())
        digest.update(repr(self.ridge).encode())
        return digest.hexdigest()
