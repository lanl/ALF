"""A seeded Gaussian sketch of a model's trainable parameters."""

import hashlib
import math

import numpy as np
import torch

SKETCH_VERSION = 1


class Sketch:
    """S ∈ ℝ^{p×P} with independent N(0, 1/p) entries, generated in chunks of rows.

    Chunk ``c`` of the sketch comes from its own CPU float64 generator, seeded from ``(seed, c)``. Any
    chunk can therefore be regenerated without the others, and the directions are identical on every
    device. ``store`` keeps the generated chunks on the device. Leave it at ``None`` to store when the
    whole sketch takes at most ``store_limit_bytes``; otherwise each use regenerates it.
    """

    def __init__(self, shapes, width, seed, *, device="cpu", dtype=torch.float64, chunk_size=64,
                 store=None, store_limit_bytes=1 << 30):
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("sketch_dim must be a positive integer.")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("The sketch chunk size must be a positive integer.")
        self.names = tuple(shapes)
        if not self.names:
            raise ValueError("The model has no trainable parameters.")
        self.shapes = tuple(torch.Size(shapes[name]) for name in self.names)
        self.sizes = tuple(math.prod(shape) for shape in self.shapes)
        self.total = sum(self.sizes)
        self.width, self.seed = width, int(seed)
        self.device, self.dtype = torch.device(device), dtype
        self.chunk_size = min(chunk_size, width)
        self.n_chunks = math.ceil(width / self.chunk_size)
        itemsize = torch.empty((), dtype=dtype).element_size()
        self.store = (width * self.total * itemsize <= store_limit_bytes) if store is None else bool(store)
        self._cache = {}
        self.digest = self._digest()

    def _rows(self, c):
        start = c * self.chunk_size
        count = min(self.chunk_size, self.width - start)
        state = np.random.SeedSequence([self.seed, c]).generate_state(1, dtype=np.uint64)[0]
        generator = torch.Generator(device="cpu").manual_seed(int(state) & ((1 << 63) - 1))
        rows = torch.randn(count, self.total, generator=generator, dtype=torch.float64)
        return rows.div_(math.sqrt(self.width))

    def flat_chunk(self, c):
        """Return rows ``[count, P]`` of chunk ``c`` on the sketch device and dtype."""
        if c in self._cache:
            return self._cache[c]
        rows = self._rows(c).to(device=self.device, dtype=self.dtype)
        if self.store:
            self._cache[c] = rows
        return rows

    def tangents(self, c):
        """Return chunk ``c`` as one ``[count, *shape]`` direction tensor per parameter name."""
        flat = self.flat_chunk(c)
        return {name: block.reshape(len(flat), *shape)
                for name, shape, block in zip(self.names, self.shapes, flat.split(self.sizes, dim=1))}

    def project(self, gradients):
        """Return ``gradients @ Sᵀ`` for flat parameter gradients ``[m, P]``, as ``[m, p]``."""
        return torch.cat([gradients @ self.flat_chunk(c).T for c in range(self.n_chunks)], dim=1)

    def to(self, device=None, dtype=None):
        """A sketch with the same directions on another device or dtype."""
        return Sketch(dict(zip(self.names, self.shapes)), self.width, self.seed,
                      device=device or self.device, dtype=dtype or self.dtype,
                      chunk_size=self.chunk_size, store=self.store)

    def _digest(self):
        digest = hashlib.sha256(repr((SKETCH_VERSION, self.seed, self.width, self.chunk_size,
                                      tuple(zip(self.names, map(tuple, self.shapes))))).encode())
        # Hash the first generated values too, so a change in the generator is detected.
        digest.update(self._rows(0)[:, :min(self.total, 4096)].numpy().tobytes())
        return digest.hexdigest()
