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

from alframework.ml_interfaces.ensemble_inference import ensemble_statistics, SelectedStateEvaluator
from alframework.samplers.acquisition import acquisition_settings, assess_frame, frame_id, top_k
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
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


class _StateMeanCalculator(Calculator):
    """ASE propagation view retaining unbiased member/state observations."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.prediction = self.evaluator.predict([atoms])
        self.results = dict(energy=float(self.prediction['energies'].mean()),
                            forces=self.prediction['forces'][0].mean(0))


class ASEEngine:
    """One ASE Langevin integrator per trajectory.

    Engine contract used by :func:`run_mlmd_trajectories`: ``step(indices, n)``
    returns ``{index: exception}`` for members that failed; ``observe(i)``
    returns the unbiased per-member prediction and minimum distance at the
    current frame; ``atoms(i)``, ``set_temperature(i, T)``,
    ``scale_cell(i, scalar)``, and ``finish(i)``. A batched engine implements
    the same methods over packed systems, freezing members on ``finish``.
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
            atoms.calc = _StateMeanCalculator(models) if self._states else \
                MLMD_calculator(list(models), **copy.deepcopy(calculator_options or {}))
            self._dynamics.append(Langevin(atoms, dt * units.fs, friction=friction,
                                           temperature_K=temperature, rng=rng))

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


def run_mlmd_trajectories(molecule_objects, engine, schedules, sampler_config, seeds=None):
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
                      'candidates': []})

    def finish(index, outcome, error=None):
        state[index]['outcome'] = outcome
        state[index]['error'] = error
        state[index]['finished_atoms'] = engine.atoms(index)
        engine.finish(index)
        active.remove(index)

    active = list(range(count))
    for index, exc in engine.step(list(active), 1).items():
        finish(index, 'failed', exc)
    for check in range(n_checks):
        time_ps = check * Ncheck * dt / 1000
        for index in list(active):
            entry, schedule = state[index], schedules[index]
            try:
                observation = engine.observe(index)
                signals = ensemble_statistics(observation['prediction'])[0]
                decision = assess_frame(signals, observation['min_distance'], time_ps, sampler_config, settings)
                entry.update(signals=signals, decision=decision, min_distance=observation['min_distance'],
                             check=check)
                if cohort and decision['outcome'] == 'eligible':
                    entry['candidates'].append(_candidate(molecule_objects[index], engine, index, signals,
                                                          decision, time_ps))
                    if keep_running:
                        # Keep sampling; the cohort selects among this trajectory's frames later.
                        decision = dict(decision, outcome='continue')
                if decision['outcome'] != 'continue':
                    finish(index, decision['outcome'])
                    continue
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
        for index, exc in engine.step(list(active), Ncheck).items():
            finish(index, 'failed', exc)
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


def _record(molecule, entry, schedule, sampler_config, settings, engine, index, started, seed, meta_dir):
    atoms = entry['finished_atoms']
    signals = entry['signals'] or {}
    decision = entry['decision'] or {}
    dt, Ncheck = sampler_config['dt'], sampler_config['Ncheck']
    step = engine.steps[index]
    meta_dict = {"realtime_simulation": time.time() - started,
                 "Es": signals.get('Es'), "Fs": signals.get('Fs'), "Fsmax": signals.get('Fsmax'),
                 "Fsrms": signals.get('Fsrms'),
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
                 "random_seed": seed}
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
        meta_dict['gap_pair'] = sampler_config['gap_uncertainty']['pair']
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
    limit = selection['k']
    if selection['max_candidates_per_trajectory'] is not None:
        limit = min(limit, selection['max_candidates_per_trajectory'])
    return top_k(candidates, limit)


def make_engine(engine, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, device=None):
    """Construct the engine; optional GPU libraries are imported only when selected."""
    if engine == 'torchsim':
        from alframework.samplers.torchsim_engine import TorchSimEngine
        return TorchSimEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                              device=device or 'cpu')
    if engine == 'alchemi':
        from alframework.samplers.alchemi_engine import AlchemiEngine
        return AlchemiEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                             device=device or 'cpu')
    if engine != 'ase':
        raise ValueError(f'Unsupported dynamics engine: {engine!r}.')
    return ASEEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options)


def sample_trajectories(molecule_objects, models, sampler_config, engine_class=None, device=None):
    """Seed, schedule, and run MLMD for a group of molecules with a loaded ensemble.

    ``engine_class`` overrides the configured ``driver.engine`` (for tests).
    """
    driver = driver_settings(sampler_config)
    states = state_settings(sampler_config)
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
            if atoms.pbc.any():
                raise ValueError('Excited-state sampling supports nonperiodic molecules only')
            if TOPOLOGY_ATOM_IDS in atoms.arrays:
                validated_atom_ids(atoms.arrays[TOPOLOGY_ATOM_IDS], len(atoms))
        models = SelectedStateEvaluator(models, selected.pop(), states['gap_pair'])
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
    if engine_class is None:
        engine = make_engine(driver['engine'], *engine_args, device=device)
    else:
        engine = engine_class(*engine_args)
    interval = sampler_config.get('trajectory_interval')
    meta_dir = sampler_config.get('meta_dir')
    for index, molecule in enumerate(molecule_objects):
        if trajectory_files[index] and interval is not None and meta_dir is not None:
            engine.attach_trajectory(index, meta_dir + "/metadata-" + molecule.get_moleculeid() + '.traj', interval)
    return run_mlmd_trajectories(molecule_objects, engine, schedules, sampler_config, seeds)


@python_app(executors=['alf_sampler_executor'])
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
    if sampler_config.get('batched_calculator') is not None:
        loader = load_module_from_config(sampler_config, 'batched_calculator')
        models = loader(model_path.format(current_model_id) + '/', device=device,
                        **(sampler_config.get('batched_calculator_options') or {}))
    else:
        calc_class = load_module_from_config(sampler_config, 'ase_calculator')
        models = calc_class(model_path.format(current_model_id) + '/', device=device)
    return sample_trajectories(list(molecule_objects), models, sampler_config, device=device)
