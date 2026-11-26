"""Per-atom novelty from the distance to reference environments of the same element."""

import math

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.neighborlist import neighbor_list

from alframework.tools.config import load_module_from_string
from alframework.tools.geometry import neighbor_search


class LocalEnvironmentNovelty:
    """Compare smooth, element-resolved radial descriptors in Euclidean distance.

    Reference structures should cover every element the potential will sample.
    The score is dimensionless and describes novelty in the radial descriptor.
    """

    def __init__(self, reference, *, cutoff=5.0, radial_bins=12, distance_block_size=1024, device='cpu'):
        reference = list(reference)
        if not reference or any(not len(atoms) for atoms in reference):
            raise ValueError('Local novelty needs nonempty reference structures.')
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('The local novelty cutoff must be positive and finite.')
        if type(radial_bins) is not int or radial_bins < 2:
            raise ValueError('Local novelty needs at least two radial bins.')
        if type(distance_block_size) is not int or distance_block_size < 1:
            raise ValueError('The local novelty distance block size must be a positive integer.')
        numbers = np.concatenate([atoms.numbers for atoms in reference])
        if np.any((numbers < 1) | (numbers > 118)):
            raise ValueError('Local novelty needs reference atomic numbers from 1 to 118.')
        species = np.unique(numbers)
        self.cutoff = cutoff
        self.block_size = distance_block_size
        self.device = torch.device(device)
        lookup = torch.full((119,), -1, dtype=torch.long)
        lookup[species] = torch.arange(len(species))
        self._lookup = lookup
        self._centers = torch.linspace(0., cutoff, radial_bins, dtype=torch.float32)
        self._width = cutoff / (radial_bins - 1)
        self._n_species = len(species)
        descriptors = torch.cat([self._ase_descriptor(atoms) for atoms in reference])
        self._references = tuple(descriptors[numbers == element].to(self.device) for element in species)
        self._lookup = lookup.to(self.device)
        self._centers = self._centers.to(self.device)

    def _channels(self, numbers):
        channels = self._lookup[numbers.clamp(0, 118)]
        if torch.any((numbers < 1) | (numbers > 118) | (channels < 0)):
            raise ValueError('A sampled element is missing from the local novelty reference.')
        return channels

    def _descriptor(self, channels, first, second, distances):
        distances = distances.to(self._centers.dtype).clamp(max=self.cutoff)
        radial = ((distances[:, None] - self._centers) / self._width).square().mul(-0.5).exp()
        envelope = 0.5 * (1 + (math.pi / self.cutoff * distances).cos())
        rows = first * self._n_species + channels[second]
        descriptor = radial.new_zeros((len(channels) * self._n_species, len(self._centers)))
        descriptor.index_add_(0, rows, radial * envelope[:, None])
        return descriptor.reshape(len(channels), -1)

    def _ase_descriptor(self, atoms):
        first, second, distances = neighbor_list('ijd', atoms, self.cutoff)
        channels = self._channels(torch.as_tensor(atoms.numbers, device=self._lookup.device))
        return self._descriptor(channels, *(torch.as_tensor(a, device=channels.device)
                                            for a in (first, second, distances)))

    def _score(self, channels, descriptors):
        scores = descriptors.new_empty(len(channels))
        for element, references in enumerate(self._references):
            indices = torch.where(channels == element)[0]
            for block in indices.split(self.block_size):
                nearest = descriptors.new_full((len(block),), float('inf'))
                for ref in references.split(self.block_size):
                    distances = torch.cdist(descriptors[block], ref, compute_mode='donot_use_mm_for_euclid_dist')
                    nearest = torch.minimum(nearest, distances.amin(1))
                scores[block] = nearest
        return scores

    @torch.no_grad()
    def score_atoms(self, atoms):
        """Score one ASE structure, using ASE's neighbor list and the shared descriptor."""
        if not len(atoms):
            raise ValueError('Local novelty needs a nonempty structure.')
        channels = self._channels(torch.as_tensor(atoms.numbers, device=self.device))
        return self._score(channels, self._ase_descriptor(atoms)).cpu().numpy()

    @torch.no_grad()
    def score_batch(self, numbers, positions, cells, pbc, counts, search):
        """Return packed atom scores and one maximum per structure on the input device."""
        channels = self._channels(numbers)
        n_atoms = torch.as_tensor(counts, device=positions.device)
        system_idx = torch.repeat_interleave(torch.arange(len(counts), device=positions.device), n_atoms,
                                               output_size=len(numbers))
        # TorchSim's search needs a complete lattice for periodic wrapping. A
        # partially periodic ASE structure may leave nonperiodic vectors unset.
        if bool(pbc.any()) and bool((torch.linalg.det(cells).abs() < 1e-12).any()):
            raise ValueError('Packed periodic local novelty requires full-rank cells.')
        mapping, systems, shifts = search(positions, cells, pbc.expand(len(counts), 3), self.cutoff, system_idx)
        first, second = mapping
        vectors = positions[second] - positions[first] + torch.einsum(
            'pi,pij->pj', shifts.to(positions.dtype), cells[systems])
        scores = self._score(channels, self._descriptor(channels, first, second, vectors.norm(dim=1)))
        maxima = scores.new_zeros(len(counts)).scatter_reduce_(0, system_idx, scores, reduce='amax')
        return scores, maxima


class LocalNoveltyEvaluator:
    """Wrap a single packed potential and score its geometry only at acquisition checks."""

    n_models = 1

    def __init__(self, evaluator, novelty, *, neighbor_backend='torch_linked_cell'):
        if evaluator.n_models != 1:
            raise ValueError('Local novelty expects a single packed potential.')
        if any(callable(getattr(evaluator, hook, None))
               for hook in ('observe_uncertainty', 'observe_uncertainty_batch')):
            raise ValueError('Wrap the base potential before adding an uncertainty observation hook.')
        self._novelty = novelty
        self._search = neighbor_search(neighbor_backend)
        if self._search is None or neighbor_backend in ('ase', 'vesin'):
            raise ValueError('Packed local novelty needs a TorchSim device neighbor backend.')
        self.evaluate = evaluator.evaluate
        self._evaluator = evaluator

    @property
    def last_uncertainty(self):
        return getattr(self._evaluator, 'last_uncertainty', None)

    @property
    def last_atomic_uncertainty(self):
        return getattr(self._evaluator, 'last_atomic_uncertainty', None)

    def observe_uncertainty_batch(self, numbers, positions, cells, pbc, counts, energies, forces):
        scores, maxima = self._novelty.score_batch(numbers, positions, cells, pbc, counts, self._search)
        atom_scores = scores.cpu().numpy()
        offsets = np.cumsum([0, *counts])
        return [_observation(atom_scores[start:stop], maximum)
                for start, stop, maximum in zip(offsets[:-1], offsets[1:], maxima.cpu().tolist())]


def _observation(scores, maximum):
    return {'uncertainty': {'local_novelty_max': float(maximum)}, 'atomic_uncertainty': {'local_novelty': scores}}


class LocalNoveltyCalculator(Calculator):
    """Wrap an ASE potential with the same local environment score."""

    def __init__(self, calculator, novelty):
        super().__init__()
        self.calculator = calculator
        self.novelty = novelty
        self.implemented_properties = calculator.implemented_properties

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        for name in properties:
            self.calculator.get_property(name, self.atoms)
        self.results = dict(self.calculator.results)
        scores = self.novelty.score_atoms(self.atoms)
        observation = _observation(scores, scores.max())
        for name, field in (('alf_uncertainty', 'uncertainty'), ('alf_atomic_uncertainty', 'atomic_uncertainty')):
            self.results[name] = {**self.results.get(name, {}), **observation[field]}


def load_local_novelty_batched(model_directory, device, *, base_loader, reference_path,
                              base_options=None, neighbor_backend='torch_linked_cell', **novelty_options):
    """ALF batched calculator loader for a potential and an ASE-readable reference file."""
    load = load_module_from_string(base_loader)
    evaluator = load(model_directory, device=device, **(base_options or {}))
    novelty = LocalEnvironmentNovelty(read(reference_path, index=':'), device=device, **novelty_options)
    return LocalNoveltyEvaluator(evaluator, novelty, neighbor_backend=neighbor_backend)
