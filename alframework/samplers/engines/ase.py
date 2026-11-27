"""ASE dynamics engine and its selected-state calculator adapters."""

import copy

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms, FixCom
from ase.md.langevin import Langevin

from alframework.ml_interfaces.ensemble_inference import SelectedStateEvaluator, add_observed_uncertainty
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.gap_seeking import gap_seeking_energy_forces
from alframework.tools.geometry import minimum_distance

OBSERVED_PROPERTIES = ['energy', 'forces', 'energy_stdev', 'forces_stdev_mean', 'forces_stdev_max']


def has_fixed_atoms(atoms):
    return any(isinstance(constraint, FixAtoms) and len(constraint.index) for constraint in atoms.constraints)


class _PredictedMember(Calculator):
    """One ensemble member's selected-state results, read from its owner's current prediction."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, owner, member):
        super().__init__()
        self.owner, self.member = owner, member

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        prediction = self.owner.prediction
        self.results = dict(energy=float(prediction['energies'][self.member, 0]),
                            forces=prediction['forces'][0][self.member])


class _StateMeanCalculator(Calculator):
    """ASE propagation view retaining unbiased member/state observations.

    Wells and UDD go through ``MLMD_calculator``, so that the formulas stay in one place.
    ``prediction`` stays unbiased either way, for screening and uncertainty. After ``set_gap_seeking``,
    the gap-seeking surface replaces the selected state's mean, and any well still adds on top.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, evaluator, calculator_options=None):
        super().__init__()
        self.evaluator = evaluator
        options = copy.deepcopy(calculator_options or {})
        self.bias = None
        self.gap_seeking = None
        if options.get('well_params') is not None or options.get('udd_bias_weight') is not None:
            members = [_PredictedMember(self, member) for member in range(evaluator.n_models)]
            self.bias = MLMD_calculator(members, **options)

    def set_gap_seeking(self, pair, sigma, alpha_eV):
        self.gap_seeking = {'pair': tuple(pair), 'sigma': float(sigma), 'alpha_eV': float(alpha_eV)}
        self.reset()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.prediction = self.evaluator.predict([atoms])
        mean_energy = float(self.prediction['energies'].mean())
        mean_forces = self.prediction['forces'][0].mean(0)
        if self.bias is None:
            energy, forces = mean_energy, mean_forces
        else:
            self.bias.calculate(atoms, ['energy', 'forces'])
            energy, forces = float(self.bias.results['energy']), self.bias.results['forces']
        if self.gap_seeking is not None:
            lower, upper = self.gap_seeking['pair']
            state_energies = self.prediction['state_energies'][:, :, 0].mean(0)
            state_forces = self.prediction['state_forces'][0].mean(0)
            surface_energy, surface_forces = gap_seeking_energy_forces(
                float(state_energies[lower]), float(state_energies[upper]), state_forces[lower], state_forces[upper],
                self.gap_seeking['sigma'], self.gap_seeking['alpha_eV'])
            energy, forces = energy + surface_energy - mean_energy, forces + surface_forces - mean_forces
        self.results = dict(energy=float(energy), forces=forces)


class ASEEngine:
    """One ASE Langevin integrator per trajectory.

    Engine contract used by :func:`~alframework.samplers.batched_mlmd.run_mlmd_trajectories`.
    ``step(indices, n)`` returns ``{index: exception}`` for members that failed; ``observe(i)``
    returns the unbiased per-member prediction and minimum distance at the
    current frame; ``atoms(i)``, ``set_temperature(i, T)``,
    ``scale_cell(i, scalar)``, ``snapshot(i)``, and ``finish(i)``. A batched engine implements
    the same methods over packed systems, freezing members on ``finish``.
    An optional ``synchronize()`` waits for device work before a timing checkpoint.
    """

    # Uses the ase_calculator ensemble (see engines.factory).
    accepts_batched_calculator = False

    def __init__(self, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, friction=0.02):
        self._states = isinstance(models, SelectedStateEvaluator)
        self.n_models = models.n_models if self._states else len(models)
        self._atoms = list(atoms_list)
        # FixAtoms already anchors the substrate. Adding FixCom would translate the fixed positions.
        self._constraints = [list(atoms.constraints) for atoms in self._atoms]
        for atoms, constraints in zip(self._atoms, self._constraints):
            if not has_fixed_atoms(atoms) and not any(isinstance(constraint, FixCom) for constraint in constraints):
                atoms.set_constraint(constraints + [FixCom()])
        self._dynamics = []
        self._writers = {}
        self.steps = [0] * len(self._atoms)
        for atoms, temperature, rng in zip(self._atoms, temperatures_K, rngs):
            # MLMD_calculator appends its well potential to the list and edits its options.
            atoms.calc = _StateMeanCalculator(models, calculator_options) if self._states else \
                MLMD_calculator(list(models), **copy.deepcopy(calculator_options or {}))
            self._dynamics.append(Langevin(atoms, dt * units.fs, friction=friction,
                                           temperature_K=temperature, rng=rng, fixcm=False))

    def set_gap_seeking(self, index, pair, sigma, alpha_eV):
        self._atoms[index].calc.set_gap_seeking(pair, sigma, alpha_eV)

    def attach_trajectory(self, index, path, interval):
        writer = Trajectory(path, mode='w', atoms=self._atoms[index])
        self._dynamics[index].attach(writer.write, interval=interval)
        self._writers[index] = writer

    def step(self, indices, n):
        failures = {}
        for index in indices:
            try:
                self._dynamics[index].run(n)
                self.steps[index] += n
            except Exception as exc:
                failures[index] = exc
        return failures

    def observe(self, index):
        return self._observe(index, score_uncertainty=True)

    def observe_physical(self, index):
        return self._observe(index, score_uncertainty=False) if self._states else self.observe(index)

    def _observe(self, index, *, score_uncertainty):
        atoms = self._atoms[index]
        if self._states:
            atoms.calc.calculate(atoms)
            prediction = atoms.calc.prediction
            if score_uncertainty:
                add_observed_uncertainty(prediction, atoms.calc.evaluator,
                                         [(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)])
            return dict(prediction=prediction, min_distance=minimum_distance(atoms))
        atoms.calc.calculate(atoms, properties=list(OBSERVED_PROPERTIES))
        results = atoms.calc.results
        prediction = {'energies': np.array(results['energy_contributions'][:self.n_models], dtype=float)[:, None],
                      'forces': [np.array(results['forces_contributions'][:self.n_models], dtype=float)]}
        if results.get('alf_uncertainty'):
            prediction['uncertainty'] = [results['alf_uncertainty']]
        if results.get('alf_atomic_uncertainty'):
            prediction['atomic_uncertainty'] = [results['alf_atomic_uncertainty']]
        return {'prediction': prediction, 'min_distance': minimum_distance(atoms)}

    def atoms(self, index):
        return self._atoms[index]

    def snapshot(self, index):
        """A detached copy of the current frame, with the constraints it came with, to label later."""
        atoms = self._atoms[index].copy()
        atoms.set_constraint(copy.deepcopy(self._constraints[index]))
        return atoms

    def set_temperature(self, index, temperature_K):
        self._dynamics[index].set_temperature(temperature_K=temperature_K)

    def scale_cell(self, index, scalar):
        atoms = self._atoms[index]
        if has_fixed_atoms(atoms):
            raise ValueError('Sampling with FixAtoms requires a fixed cell.')
        atoms.set_cell(scalar * atoms.get_cell(), scale_atoms=True)

    def finish(self, index):
        writer = self._writers.pop(index, None)
        if writer is not None:
            writer.close()
        self._atoms[index].calc = None
        # A FixCom left on would remove the net force from any labels read through these Atoms.
        self._atoms[index].set_constraint(self._constraints[index])
