"""Multi-trajectory MLMD sampling with first-trigger returns.

``mlmd_trajectories_task`` runs one or more independent Langevin trajectories
in a single sampler task. It follows ``mlmd_sampling`` step for step:

- one initial MD step, then a check every ``Ncheck`` steps;
- the temperature and optional density follow ``annealing_schedule``;
- a trajectory stops at its first eligible (uncertain) frame and returns it
  for labeling, or stops with ``atoms=None`` on a ``distcut`` violation;
- trajectories that never trigger return ``atoms=None`` after ``maxt``.

Each trajectory stops independently; stopped trajectories are not stepped
again while their batch-mates continue. The task returns once, when its last
trajectory finishes, with one ``MoleculesObject`` per input in input order.

Eligibility and ranking come from :mod:`alframework.samplers.acquisition` and
use unbiased physical ensemble predictions, never UDD or well-potential terms.

Sampler configuration additions (all other keys are those of
``simple_mlmd_sampling_task``)::

    "driver": {"strategy": "batched", "engine": "ase"},
    "batching": {"batch_size": 8, "partial_policy": "timeout", "max_wait_seconds": 5.0},
    "acquisition": {"strategy": "uncertainty"},
    "random_seed": 1234,
    "device": "cpu"

``driver.engine`` ``torchsim`` or ``alchemi`` packs a task's trajectories into
one engine state. Either may use a batched
ensemble loader, ``"batched_calculator":
"alframework.ml_interfaces.hippynn_interface.load_hippynn_batched_ensemble"``
(with optional ``batched_calculator_options``), in place of ``ase_calculator``.

With ``random_seed``, each trajectory's random stream (schedule draws,
trajectory-file selection, Langevin noise) is derived from the seed and its
molecule ID, so results do not depend on how trajectories are grouped. UDD is
selected with ``MLMD_calculator_options.udd_bias_weight``.
"""

import copy
import hashlib
import os
import pickle as pkl
import time

import numpy as np
from ase import units
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from parsl import python_app
from alframework.tools.task_timeline import timed

from alframework.ml_interfaces.ensemble_inference import ensemble_statistics, SelectedStateEvaluator
from alframework.samplers.acquisition import acquisition_settings, assess_frame, frame_id, top_k, worker_retention_limit
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.gap_seeking import adjacent_pairs, gap_seeking_settings, gap_seeking_energy_forces, select_trigger
from alframework.samplers.geometry import minimum_distance
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.gpu import sampling_device
from alframework.tools.sampler_coordinator import driver_settings, state_settings, resolve_selected_state
from alframework.tools.tools import annealing_schedule, load_module_from_config

OBSERVED_PROPERTIES = ['energy', 'forces', 'energy_stdev', 'forces_stdev_mean', 'forces_stdev_max']


def trajectory_seed(molecule_id, random_seed=None):
    """Return a hex seed for one trajectory.

    With ``random_seed`` the seed depends only on it and the molecule ID;
    otherwise it is drawn from OS entropy (and recorded in metadata).
    """
    material = os.urandom(32) if random_seed is None else repr((random_seed, str(molecule_id))).encode()
    return hashlib.sha256(material).hexdigest()


def trajectory_rng(seed_hex):
    return np.random.RandomState(np.frombuffer(bytes.fromhex(seed_hex)[:16], dtype=np.uint32))


def _draw(rng, value_range):
    return None if value_range is None else rng.uniform(value_range[0], value_range[1])


def draw_schedule(sampler_config, rng):
    """Draw one trajectory's schedule parameters, as ``simple_mlmd_sampling_task`` does."""
    return {'Tamp': _draw(rng, sampler_config['amp_temp']), 'Tper': _draw(rng, sampler_config['per_temp']),
            'Tsrt': _draw(rng, sampler_config['srt_temp']), 'Tend': _draw(rng, sampler_config['end_temp']),
            'Ramp': _draw(rng, sampler_config['amp_dens']), 'Rper': _draw(rng, sampler_config['per_dens']),
            'Rend': _draw(rng, sampler_config['end_dens'])}


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

    Engine contract used by :func:`run_mlmd_trajectories`: ``step(indices, n)``
    returns ``{index: exception}`` for members that failed; ``observe(i)``
    returns the unbiased per-member prediction and minimum distance at the
    current frame; ``atoms(i)``, ``set_temperature(i, T)``,
    ``scale_cell(i, scalar)``, and ``finish(i)``. A batched engine implements
    the same methods over packed systems, freezing members on ``finish``.
    An optional ``synchronize()`` waits for device work before a timing checkpoint.
    """

    def __init__(self, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, friction=0.02):
        self._states = isinstance(models, SelectedStateEvaluator)
        self.n_models = models.n_models if self._states else len(models)
        self._atoms = list(atoms_list)
        self._dynamics = []
        self._writers = {}
        self.steps = [0] * len(self._atoms)
        for atoms, temperature, rng in zip(self._atoms, temperatures_K, rngs):
            # MLMD_calculator appends its well potential to the list and edits its options.
            atoms.calc = _StateMeanCalculator(models, calculator_options) if self._states else \
                MLMD_calculator(list(models), **copy.deepcopy(calculator_options or {}))
            self._dynamics.append(Langevin(atoms, dt * units.fs, friction=friction,
                                           temperature_K=temperature, rng=rng))

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
        atoms = self._atoms[index]
        if self._states:
            atoms.calc.calculate(atoms)
            return dict(prediction=atoms.calc.prediction, min_distance=minimum_distance(atoms))
        atoms.calc.calculate(atoms, properties=list(OBSERVED_PROPERTIES))
        results = atoms.calc.results
        prediction = {'energies': np.array(results['energy_contributions'][:self.n_models], dtype=float)[:, None],
                      'forces': [np.array(results['forces_contributions'][:self.n_models], dtype=float)]}
        return {'prediction': prediction, 'min_distance': minimum_distance(atoms)}

    def atoms(self, index):
        return self._atoms[index]

    def set_temperature(self, index, temperature_K):
        self._dynamics[index].set_temperature(temperature_K=temperature_K)

    def scale_cell(self, index, scalar):
        atoms = self._atoms[index]
        atoms.set_cell(scalar * atoms.get_cell(), scale_atoms=True)

    def finish(self, index):
        writer = self._writers.pop(index, None)
        if writer is not None:
            writer.close()
        self._atoms[index].calc = None


def run_mlmd_trajectories(molecule_objects, engine, schedules, sampler_config, seeds=None, *, topology=None, screen_audit=None):
    """Run trajectories in lockstep with independent first-trigger stopping.

    Args:
        molecule_objects (list): Inputs, whose atoms are those driven by ``engine``.
        engine: An engine implementing the :class:`ASEEngine` contract.
        schedules (list): Per-trajectory ``Tamp``, ``Tper``, ``Tsrt``, ``Tend``,
            ``Ramp``, ``Rper``, ``Rend``.
        sampler_config (dict): ``dt``, ``maxt``, ``Ncheck``, ``Escut``,
            ``Fscut`` and optional ``distcut``, ``min_time``, ``meta_dir``,
            ``acquisition``.
        seeds (list): Optional per-trajectory seeds to record.

    Returns:
        (list): The input ``MoleculesObject`` instances, updated in place.
    """
    settings = acquisition_settings(sampler_config)
    cohort = settings['selection']['mode'] == 'cohort_top_k'
    keep_running = settings['trajectory_policy'] == 'continue'
    dt, maxt, Ncheck = sampler_config['dt'], sampler_config['maxt'], sampler_config['Ncheck']
    meta_dir = sampler_config.get('meta_dir')
    n_checks = int(np.ceil((1000 * maxt) / (dt * Ncheck)))
    count = len(molecule_objects)
    started = time.time()
    state = []
    for index, schedule in enumerate(schedules):
        atoms = engine.atoms(index)
        rho0 = None if schedule['Rend'] is None else \
            (1.66054e-24 / 1.0e-24) * (np.sum(atoms.get_masses()) / atoms.get_volume())
        state.append({'temps': [], 'denss': [], 'rho0': rho0, 'decision': None, 'signals': None,
                      'min_distance': None, 'check': 0, 'outcome': None, 'error': None,
                      'candidates': [], 'peaks': {}, 'finished_at': None,
                      'step_times': {}})

    def finish(index, outcome, error=None):
        state[index]['outcome'] = outcome
        state[index]['error'] = error
        state[index]['finished_atoms'] = engine.atoms(index)
        state[index]['finished_at'] = time.time()
        engine.finish(index)
        active.remove(index)

    last_step_mark = None

    def advance(n):
        """Step to timing boundaries without adding uncertainty or acquisition checks."""
        nonlocal last_step_mark
        synchronize = getattr(engine, 'synchronize', None)
        if last_step_mark is None:
            if synchronize is not None:
                synchronize()
            last_step_mark = time.perf_counter()
        while active and n > 0:
            completed = engine.steps[active[0]]
            chunk = min(n, STEP_TIMING_INTERVAL - completed % STEP_TIMING_INTERVAL)
            for index, exc in engine.step(list(active), chunk).items():
                finish(index, 'failed', exc)
            n -= chunk
            if active and engine.steps[active[0]] % STEP_TIMING_INTERVAL == 0:
                if synchronize is not None:
                    synchronize()
                now = time.perf_counter()
                elapsed = now - last_step_mark
                # Every surviving trajectory has completed the same batch steps.
                for index in active:
                    state[index]['step_times'][engine.steps[index]] = elapsed
                last_step_mark = now

    if 'sampling_window' in sampler_config:
        for molecule in molecule_objects:
            molecule.update_metadata(dict(sampling_window=sampler_config['sampling_window']))
    seeking = gap_seeking_settings(sampler_config)
    if seeking is not None:
        for molecule in molecule_objects:
            molecule.update_metadata(dict(gap_seeking_enabled=True, gap_seeking_current_mode='selected_state',
                                          gap_seeking_switched=False))

    def seek_gap(index, prediction, time_ps):
        """Switch a running trajectory to the gap-seeking surface once its gap is small enough."""
        molecule = molecule_objects[index]
        if molecule.get_metadata()['gap_seeking_switched']:
            return
        state_energies = np.asarray(prediction['state_energies'], dtype=float)
        pairs = adjacent_pairs(prediction['selected_state'], state_energies.shape[1])
        trigger = select_trigger(state_energies[:, :, 0].mean(0), pairs, seeking['trigger_gap_threshold_eV'])
        if trigger is None:
            return
        engine.set_gap_seeking(index, trigger['pair'], seeking['sigma'], seeking['alpha_eV'])
        molecule.update_metadata(dict(gap_seeking_current_mode='gap_seeking', gap_seeking_switched=True,
                                      gap_seeking_pair=list(trigger['pair']), gap_seeking_trigger_gap_eV=trigger['gap_eV'],
                                      gap_seeking_trigger_step=int(engine.steps[index]),
                                      gap_seeking_trigger_time_ps=float(time_ps)))

    cutoff = sampler_config.get('max_force_cutoff')
    screening = topology is not None or cutoff is not None
    if cutoff is not None:
        from alframework.tools.force_screen import validated_force_cutoff
        cutoff = validated_force_cutoff(cutoff)

    def inspect_frame(index, observation, time_ps):
        from alframework.tools.molecular_topology import validate_fixed_topology, topology_metadata
        from alframework.tools.force_screen import physical_force_diagnostics, physical_force_exceeds, force_screen_metadata
        molecule = molecule_objects[index]
        atoms = engine.atoms(index)
        report = topology_metadata(validate_fixed_topology(atoms, topology)) if topology is not None else {}
        force = physical_force_diagnostics(observation['prediction']) if cutoff is not None else None
        if force is not None:
            report.update(force_screen_metadata(force))
            report['physical_force_valid'] = not physical_force_exceeds(force, cutoff)
            report['max_force_cutoff'] = cutoff
        reason = report.get('topology_reject_reason')
        if reason is None and force is not None and not report['physical_force_valid']:
            reason = 'physical_force'
        report['screen_reject_reason'] = reason
        molecule.update_metadata(report)
        if screen_audit is not None:
            diagnostics = dict(report, **({'physical_forces': force['physical_forces']} if force is not None else {}))
            if engine.steps[index] == 0:
                screen_audit.record(molecule, 'initial', atoms, 0, 0., diagnostics)
            screen_audit.record(molecule, 'rejected' if reason else 'last_valid', atoms,
                                engine.steps[index], engine.steps[index] * dt / 1000, diagnostics)
        return report, reason

    active = list(range(count))
    if screening:
        for index in list(active):
            try:
                observation = engine.observe(index)
                report, reason = inspect_frame(index, observation, 0.)
                if reason:
                    state[index].update(decision=dict(outcome='rejected'),
                                        signals={k:v for k,v in report.items() if k == 'Fmeanmax' or k.startswith('physical_F')})
                    finish(index, 'rejected')
            except Exception as exc:
                finish(index, 'failed', exc)
    for molecule in molecule_objects:
        molecule.update_metadata(dict(sampling_entered_dynamics_width=len(active)))
    advance(1)
    for check in range(n_checks):
        time_ps = check * Ncheck * dt / 1000
        for index in list(active):
            entry, schedule = state[index], schedules[index]
            try:
                observation = engine.observe(index)
                if screening:
                    report, reason = inspect_frame(index, observation, time_ps)
                    if reason:
                        # Keep the last valid uncertainty, since this frame's is physically suspect.
                        entry.update(decision=dict(outcome='rejected'), check=check,
                                     signals={**(entry['signals'] or {}),
                                              **{k:v for k,v in report.items() if k == 'Fmeanmax' or k.startswith('physical_F')}})
                        finish(index, 'rejected')
                        continue
                signals = ensemble_statistics(observation['prediction'])[0]
                if screening:
                    signals.update({k:v for k,v in report.items() if k == 'Fmeanmax' or k.startswith('physical_F')})
                decision = assess_frame(signals, observation['min_distance'], time_ps, sampler_config, settings)
                entry.update(signals=signals, decision=decision, min_distance=observation['min_distance'],
                             check=check)
                for key in PEAK_SIGNALS:
                    if key in signals:  # gap_std only with a gap_uncertainty pair.
                        entry['peaks'][key] = max(signals[key], entry['peaks'].get(key, signals[key]))
                if cohort and decision['outcome'] == 'eligible':
                    entry['candidates'].append(_candidate(molecule_objects[index], engine, index, signals,
                                                          decision, time_ps))
                    # A frame outside this trajectory's best can never be selected, so don't hold its geometry.
                    entry['candidates'] = _prune(entry['candidates'], settings['selection'])
                    if keep_running:
                        # Keep sampling; the cohort selects among this trajectory's frames later.
                        decision = dict(decision, outcome='continue')
                if decision['outcome'] != 'continue':
                    finish(index, decision['outcome'])
                    continue
                if seeking is not None:
                    seek_gap(index, observation['prediction'], time_ps)
                set_temp = annealing_schedule(time_ps, maxt, schedule['Tamp'], schedule['Tper'],
                                              schedule['Tsrt'], schedule['Tend'])
                engine.set_temperature(index, set_temp)
                if entry['rho0'] is not None:
                    set_dens = (1.0e-24 / 1.66054e-24) * annealing_schedule(
                        time_ps, maxt, schedule['Ramp'], schedule['Rper'], entry['rho0'], schedule['Rend'])
                    atoms = engine.atoms(index)
                    engine.scale_cell(index, np.power(np.sum(atoms.get_masses()) /
                                                      (atoms.get_volume() * set_dens), 1. / 3.))
                    entry['denss'].append(set_dens)
                entry['temps'].append(set_temp)
            except Exception as exc:
                finish(index, 'failed', exc)
        if not active:
            break
        advance(Ncheck)
    for index in list(active):
        finish(index, 'completed')
    if cohort:
        for entry in state:
            if entry['outcome'] in ('completed', 'eligible') and entry['candidates']:
                entry['outcome'] = 'eligible'
            entry['candidates'] = _prune(entry['candidates'], settings['selection'])

    for index, molecule in enumerate(molecule_objects):
        _record(molecule, state[index], schedules[index], sampler_config, settings, engine,
                index, started, None if seeds is None else seeds[index], meta_dir)
    return list(molecule_objects)


PEAK_SIGNALS = ('Es', 'Fs', 'Fsmax', 'gap_std')

# Each ``delta_realtime_step_<n>`` is the wall time of the STEP_TIMING_INTERVAL steps that end at step n, as the
# CAAR FOM analyzer reads it. This includes checks between those steps, but excludes initial screening.
# In a batch all trajectories step together, so it is the batch wall time, shared by its active trajectories.
STEP_TIMING_INTERVAL = 100


def _record(molecule, entry, schedule, sampler_config, settings, engine, index, started, seed, meta_dir):
    atoms = entry['finished_atoms']
    signals = entry['signals'] or {}
    decision = entry['decision'] or {}
    dt, Ncheck = sampler_config['dt'], sampler_config['Ncheck']
    step = engine.steps[index]
    meta_dict = {"realtime_simulation": entry['finished_at'] - started,
                 "Es": signals.get('Es'), "Fs": signals.get('Fs'), "Fsmax": signals.get('Fsmax'),
                 "Fsrms": signals.get('Fsrms'),
                 **{f"peak_{key}": entry['peaks'].get(key) for key in PEAK_SIGNALS},
                 "distmin": entry['min_distance'],
                 "simulationtime": (entry['check'] * Ncheck * dt) / 1000,
                 "temps": entry['temps'], "denss": entry['denss'],
                 "system_comp": np.unique(atoms.get_chemical_symbols(), return_counts=True),
                 "Tamp": schedule['Tamp'], "Tper": schedule['Tper'], "Tsrt": schedule['Tsrt'],
                 "Tend": schedule['Tend'], "Ramp": schedule['Ramp'], "Rper": schedule['Rper'],
                 "Rsrt": entry['rho0'], "Rend": schedule['Rend'],
                 "Ecrit": decision.get('Ecrit'), "Fcrit": decision.get('Fcrit'),
                 "Fmcrit": decision.get('Fmcrit'), "distcrit": decision.get('distcrit'),
                 "chemical_symbols": atoms.get_chemical_symbols(),
                 "positions": atoms.get_positions(wrap=True),
                 "cell": atoms.get_cell(),
                 "trajectory_outcome": entry['outcome'],
                 "acquisition_strategy": settings['strategy'],
                 "step": step,
                 "frame_id": frame_id(molecule.get_moleculeid(), step),
                 "random_seed": seed,
                 **{f"delta_realtime_step_{steps}": seconds for steps, seconds in entry['step_times'].items()}}
    cohort = settings['selection']['mode'] == 'cohort_top_k'
    if entry['outcome'] == 'eligible' and not cohort:
        meta_dict['ranking_score'] = decision['ranking_score']
    if cohort:
        meta_dict['acquisition_candidates'] = entry['candidates']
    if entry['error'] is not None:
        meta_dict['sampling_error'] = "{}: {}".format(type(entry['error']).__name__, entry['error'])
    meta_dict.update(molecule.get_metadata())
    meta_dict.update(signals)
    for key in ('acquisition_trigger_flags', 'acquisition_cutoffs', 'acquisition_score_terms'):
        if key in decision:
            meta_dict[key] = decision[key]
    if sampler_config.get('gap_uncertainty') is not None:
        gap = sampler_config['gap_uncertainty']
        meta_dict['gap_pair'] = gap['pair'] if 'pair' in gap else gap['pairs']
    if meta_dir is not None:
        with open(meta_dir + "/metadata-" + molecule.get_moleculeid() + '.p', "wb") as handle:
            pkl.dump(meta_dict, handle)
    molecule.update_metadata(meta_dict)
    # In cohort mode frames travel as candidates; nothing is labeled until the cohort selects.
    molecule.update_atoms(atoms if entry['outcome'] == 'eligible' and not cohort else None)


def _candidate(molecule, engine, index, signals, decision, time_ps):
    atoms = engine.atoms(index)
    step = engine.steps[index]
    return {"trajectory_id": molecule.get_moleculeid(), "step": int(step),
            "frame_id": frame_id(molecule.get_moleculeid(), step),
            "time_ps": float(time_ps), "ranking_score": float(decision['ranking_score']),
            "signals": {key: float(value) for key, value in signals.items()},
            "flags": {key: decision[key] for key in ('Ecrit', 'Fcrit', 'Fmcrit')},
            "numbers": atoms.get_atomic_numbers().copy(), "positions": atoms.get_positions().copy(),
            "cell": np.array(atoms.get_cell()), "pbc": atoms.get_pbc().copy(),
            "metadata": {**copy.deepcopy(molecule.get_metadata()),
                         **{key: decision[key] for key in ('acquisition_trigger_flags', 'acquisition_cutoffs',
                                                          'acquisition_score_terms') if key in decision}},
            "topology_atom_ids": None if 'topology_atom_ids' not in atoms.arrays else
                atoms.arrays['topology_atom_ids'].copy()}


def _prune(candidates, selection):
    """Keep a trajectory's best frames; never more than the cohort can select from one trajectory."""
    return top_k(candidates, worker_retention_limit(selection))


def make_engine(engine, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, device=None, friction=0.02):
    """Construct the engine; optional GPU libraries are imported only when selected."""
    if engine == 'torchsim':
        from alframework.samplers.torchsim_engine import TorchSimEngine
        return TorchSimEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                              device=device or 'cpu', friction=friction)
    if engine == 'alchemi':
        from alframework.samplers.alchemi_engine import AlchemiEngine
        return AlchemiEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                             device=device or 'cpu', friction=friction)
    if engine != 'ase':
        raise ValueError(f'Unsupported dynamics engine: {engine!r}.')
    return ASEEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options, friction=friction)


def sample_trajectories(molecule_objects, models, sampler_config, engine_class=None, device=None):
    """Seed, schedule, and run MLMD for a group of molecules with a loaded ensemble.

    ``engine_class`` overrides the configured ``driver.engine`` (for tests).
    """
    driver = driver_settings(sampler_config)
    states = state_settings(sampler_config)
    from alframework.tools.molecular_topology import load_reference_topology, validate_fixed_topology, topology_metadata
    from alframework.tools.force_screen import RuntimeScreenAudit, validated_force_cutoff
    topology = load_reference_topology(sampler_config)
    if sampler_config.get('max_force_cutoff') is not None:
        validated_force_cutoff(sampler_config['max_force_cutoff'])
    screening = topology is not None or sampler_config.get('max_force_cutoff') is not None
    original_molecules = list(molecule_objects)
    audit = RuntimeScreenAudit(original_molecules, sampler_config) if screening else None
    if states is not None:
        resolved = {resolve_selected_state(m, states) for m in original_molecules}
        if len(resolved) != 1:
            raise ValueError('A sampler task must share one selected state')
    if topology is not None:
        valid = []
        for molecule in original_molecules:
            atoms = molecule.get_atoms()
            report = topology_metadata(validate_fixed_topology(atoms, topology))
            audit.record(molecule, 'initial', atoms, diagnostics=report)
            molecule.update_metadata(report)
            if not report['topology_valid']:
                molecule.update_metadata(dict(trajectory_outcome='rejected', step=0, simulationtime=0.,
                    acquisition_candidates=[], screen_reject_reason=report['topology_reject_reason']))
                audit.record(molecule, 'rejected', atoms, diagnostics=report)
                molecule.update_atoms(None)
            else:
                valid.append(molecule)
        molecule_objects = valid
    for molecule in original_molecules:
        molecule.update_metadata(dict(sampling_initial_active_width=len(molecule_objects)))
    if not molecule_objects:
        if audit is not None:audit.write()
        for molecule in original_molecules:
            molecule.update_metadata(dict(sampling_entered_dynamics_width=0))
        return original_molecules
    if states is not None:
        from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, validated_atom_ids
        selected = {resolve_selected_state(m, states) for m in molecule_objects}
        if len(selected) != 1:
            raise ValueError('A sampler task must share one selected state')
        if not hasattr(models, 'evaluate_states'):
            raise ValueError('Excited-state sampling requires a multi-state evaluator')
        policy = states['policy']
        configured = [] if policy is None else ([policy['state']] if policy['mode'] == 'fixed' else policy['states'])
        if any(s >= models.n_states for s in configured):
            raise ValueError('Configured state is absent from ensemble')
        for molecule in molecule_objects:
            atoms = molecule.get_atoms()
            if states['gap_pair'] is not None:
                gap_pair = states['gap_pair']
                molecule.update_metadata(dict(gap_pair=gap_pair if isinstance(gap_pair, str) else list(gap_pair)))
            if atoms.pbc.any():
                raise ValueError('Excited-state sampling supports nonperiodic molecules only')
            if TOPOLOGY_ATOM_IDS in atoms.arrays:
                validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
        models = SelectedStateEvaluator(models, selected.pop(), states['gap_pair'])
        if states['gap_seeking'] is not None:
            adjacent_pairs(models.selected_state, models.evaluator.n_states)
    seeds, rngs, schedules, trajectory_files = [], [], [], []
    for molecule in molecule_objects:
        assert isinstance(molecule, MoleculesObject), 'molecule_objects must contain MoleculesObject instances'
        seed = trajectory_seed(molecule.get_moleculeid(), sampler_config.get('random_seed'))
        rng = trajectory_rng(seed)
        schedules.append(draw_schedule(sampler_config, rng))
        trajectory_files.append(rng.rand() <= sampler_config.get('trajectory_frequency', 0))
        seeds.append(seed)
        rngs.append(rng)
        if sampler_config.get('translate_to_center', False):
            atoms = molecule.get_atoms()
            atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())

    maxt = sampler_config['maxt']
    temperatures = [annealing_schedule(0.0, maxt, s['Tamp'], s['Tper'], s['Tsrt'], s['Tend']) for s in schedules]
    engine_args = ([molecule.get_atoms() for molecule in molecule_objects], models, sampler_config['dt'],
                   temperatures, rngs, sampler_config.get('MLMD_calculator_options'))
    friction_options = {}
    if 'friction_per_fs' in sampler_config:
        gamma = sampler_config['friction_per_fs']
        if isinstance(gamma, bool) or not np.isfinite(float(gamma)) or float(gamma) < 0:
            raise ValueError('friction_per_fs must be finite and nonnegative')
        friction_options['friction'] = float(gamma) / units.fs
    if engine_class is None:
        engine = make_engine(driver['engine'], *engine_args, device=device, **friction_options)
    else:
        engine = engine_class(*engine_args, **friction_options)
    interval = sampler_config.get('trajectory_interval')
    meta_dir = sampler_config.get('meta_dir')
    for index, molecule in enumerate(molecule_objects):
        if trajectory_files[index] and interval is not None and meta_dir is not None:
            engine.attach_trajectory(index, meta_dir + "/metadata-" + molecule.get_moleculeid() + '.traj', interval)
    try:
        run_mlmd_trajectories(molecule_objects, engine, schedules, sampler_config, seeds,
                              topology=topology, screen_audit=audit)
        entered = molecule_objects[0].get_metadata()['sampling_entered_dynamics_width']
        for molecule in original_molecules:
            molecule.update_metadata(dict(sampling_entered_dynamics_width=entered))
        return original_molecules
    finally:
        if audit is not None:audit.write()


@python_app(executors=['alf_sampler_executor'])
@timed('sampler')
def mlmd_trajectories_task(molecule_objects, sampler_config, model_path, current_model_id, gpus_per_node=None):
    """Run MLMD for several trajectories in one task.

    Args:
        molecule_objects (list): MoleculesObject instances to sample.
        sampler_config (dict): ``simple_mlmd_sampling_task`` settings plus the
            optional ``driver``, ``batching``, ``acquisition``,
            ``random_seed``, and ``device`` keys described in this module.
        model_path (str): Model path pattern from the master configuration.
        current_model_id (int): Model generation to sample with.
        gpus_per_node (int): GPU assignment cap when worker visibility is unmasked.

    Returns:
        (list): One MoleculesObject per input. ``atoms`` is the selected frame
        for eligible trajectories and ``None`` otherwise.
    """
    if sampler_config.get('use_potential_specific_code') is not None:
        raise ValueError("mlmd_trajectories_task does not support use_potential_specific_code; "
                         "use simple_mlmd_sampling_task.")
    device = sampling_device(sampler_config, gpus_per_node)
    if state_settings(sampler_config) is not None:
        for molecule in molecule_objects:
            molecule.update_metadata(dict(sampling_task_width=len(molecule_objects),
                                           sampling_engine=driver_settings(sampler_config)['engine']))
    models = worker_ensemble(sampler_config, model_path.format(current_model_id) + '/', device)
    return sample_trajectories(list(molecule_objects), models, sampler_config, device=device)


# The ensemble this worker loaded last. A promoted model directory never changes, so later tasks
# with the same model, loader, and device reuse it instead of reading every checkpoint again.
_WORKER_ENSEMBLE = {}


def worker_ensemble(sampler_config, model_directory, device):
    """Load the sampler's ensemble, or return the one this worker already loaded for the same model."""
    import json
    batched = sampler_config.get('batched_calculator') is not None
    options = sampler_config.get('batched_calculator_options') or {}
    key = (sampler_config.get('batched_calculator') if batched else sampler_config.get('ase_calculator'),
           str(model_directory), str(device), json.dumps(options, sort_keys=True, default=str))
    if key not in _WORKER_ENSEMBLE:
        _WORKER_ENSEMBLE.clear()  # Free the previous model's memory before loading the new one.
        if batched:
            loader = load_module_from_config(sampler_config, 'batched_calculator')
            _WORKER_ENSEMBLE[key] = loader(model_directory, device=device, **options)
        else:
            calc_class = load_module_from_config(sampler_config, 'ase_calculator')
            _WORKER_ENSEMBLE[key] = calc_class(model_directory, device=device)
    return _WORKER_ENSEMBLE[key]
