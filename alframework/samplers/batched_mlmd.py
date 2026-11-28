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
"alframework.ml_interfaces.hippynn.load_hippynn_batched_ensemble"``
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

from alframework.ml_interfaces.ensemble_inference import atomic_uncertainty_batch, ensemble_statistics, SelectedStateEvaluator
from alframework.samplers.acquisition import (acquisition_settings, assess_frame, custom_terms, frame_id,
                                              require_members, resolve_force_atom_indices,
                                              selected_force_diagnostics, top_k, worker_retention_limit)
# Keep the old paths importable, including for previously pickled calculators.
from alframework.samplers.engines.ase import (ASEEngine, OBSERVED_PROPERTIES, _PredictedMember,
                                             _StateMeanCalculator, has_fixed_atoms)
from alframework.samplers.engines.factory import make_engine
from alframework.tools.gap_seeking import adjacent_pairs, gap_seeking_settings, select_trigger
from alframework.tools.geometry import minimum_distance
from alframework.tools.molecules import MoleculesObject
from alframework.tools.gpu import sampling_device
from alframework.tools.sampler_coordinator import driver_settings, state_settings, resolve_selected_state
from alframework.tools.annealing import annealing_schedule
from alframework.tools.config import load_module_from_config
from alframework.tools.tasks import alf_task
from alframework.tools.velocities import initialize_velocities, kinetic_temperature


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


def run_mlmd_trajectories(molecule_objects, engine, schedules, sampler_config, seeds=None, *, topology=None,
                          surface_checks=None, screen_audit=None, bundle_dir=None):
    """Run trajectories in lockstep with first-trigger or final-frame selection.

    With ``acquisition.distcut_frames`` "label", a ``distcut`` violation is a frame to label: under first-trigger
    selection the trajectory stops there, and under final-frame selection it runs to its duration as usual but
    labels its first such frame.

    Args:
        molecule_objects (list): Inputs, whose atoms are those driven by ``engine``.
        engine: An engine implementing the :class:`~alframework.samplers.engines.ase.ASEEngine` contract.
        schedules (list): Per-trajectory ``Tamp``, ``Tper``, ``Tsrt``, ``Tend``,
            ``Ramp``, ``Rper``, ``Rend``.
        sampler_config (dict): ``dt``, ``maxt``, ``Ncheck``, ``Escut``,
            ``Fscut`` and optional ``distcut``, ``min_time``, ``meta_dir``,
            ``metadata_files``, ``acquisition``.
        seeds (list): Optional per-trajectory seeds to record.
        surface_checks (list): Optional per-trajectory adsorbate screens prepared at step zero.
        bundle_dir (str): Where ``metadata_files`` "per_task" writes its one file (default ``meta_dir``).

    Returns:
        (list): The input ``MoleculesObject`` instances, updated in place.
    """
    settings = acquisition_settings(sampler_config)
    # The model's own uncertainty the run's acquisition terms read, whose largest values each trajectory records.
    own_uncertainty = tuple(sorted({term['uncertainty'] for term in custom_terms(sampler_config).values()}))
    cohort = settings['selection']['mode'] == 'cohort_top_k'
    adaptive_ntk = settings['selection'].get('selector') == 'adaptive_ntk'
    final_frame = settings['selection']['mode'] == 'final_frame'
    # final_frame with distcut_frames "label": the first close contact, not the final frame, goes to QM.
    contact_frames = final_frame and settings['distcut_frames'] == 'label'
    if contact_frames and not callable(getattr(engine, 'snapshot', None)):
        raise ValueError("acquisition.distcut_frames 'label' with selection.mode 'final_frame' needs an engine "
                         "with snapshot(i).")
    keep_running = settings['trajectory_policy'] == 'continue'
    dt, maxt, Ncheck = sampler_config['dt'], sampler_config['maxt'], sampler_config['Ncheck']
    meta_dir = sampler_config.get('meta_dir')
    n_checks = int(np.ceil((1000 * maxt) / (dt * Ncheck)))
    if final_frame:
        final_step = max(1, int(np.ceil(1000 * maxt / dt)))
        n_checks = 1 + (final_step - 1 + Ncheck - 1) // Ncheck
    count = len(molecule_objects)
    force_atoms = settings.get("force_atoms")
    selected_sources = settings.get("selected_force_signals", ())
    force_atom_indices = None if force_atoms is None else [
        resolve_force_atom_indices(force_atoms, molecule.get_atoms(), molecule.get_moleculeid())
        for molecule in molecule_objects]
    set_force_atoms = getattr(engine, "set_force_atom_indices", None)
    if callable(set_force_atoms):
        set_force_atoms(force_atom_indices)
    force_atom_array = None if force_atoms is None else force_atoms.get("array")
    started = time.time()
    state = []
    for index, schedule in enumerate(schedules):
        atoms = engine.atoms(index)
        rho0 = None if schedule['Rend'] is None else \
            (1.66054e-24 / 1.0e-24) * (np.sum(atoms.get_masses()) / atoms.get_volume())
        state.append({'temps': [], 'denss': [], 'rho0': rho0, 'decision': None, 'signals': None,
                      'min_distance': None, 'check': 0, 'outcome': None, 'error': None,
                      'candidates': [], 'candidates_pruned': 0, 'peaks': {}, 'finished_at': None,
                      'step_times': {}})

    def finish(index, outcome, error=None):
        state[index]['outcome'] = outcome
        # Keep the message, not the exception: its traceback holds the failed evaluation's frames and their
        # tensors, which would stay allocated for the rest of the task.
        state[index]['error'] = None if error is None else "{}: {}".format(type(error).__name__, error)
        state[index]['finished_atoms'] = engine.atoms(index)
        # While the engine still holds the center of mass fixed, so it counts the same degrees of freedom.
        state[index]['finished_kinetic_temperature_K'] = kinetic_temperature(state[index]['finished_atoms'])
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
    screening = topology is not None or surface_checks is not None or cutoff is not None
    if cutoff is not None:
        from alframework.tools.force_screen import validated_force_cutoff
        cutoff = validated_force_cutoff(cutoff)

    def inspect_frame(index, observe):
        from alframework.tools.molecular_topology import validate_fixed_topology, topology_metadata
        from alframework.tools.force_screen import physical_force_diagnostics, physical_force_exceeds, force_screen_metadata
        molecule = molecule_objects[index]
        atoms = engine.atoms(index)
        reference = topology[index] if isinstance(topology, list) else topology
        report = topology_metadata(validate_fixed_topology(atoms, reference)) if reference is not None else {}
        step = engine.steps[index]
        time_ps = step * dt / 1000
        surface_check = surface_checks[index] if surface_checks is not None else None
        if surface_check is not None:
            report.update(surface_check.inspect(atoms, step, time_ps))
        observation = None if report.get('surface_reject_reason') else observe(index)
        force = physical_force_diagnostics(observation['prediction']) if cutoff is not None and observation else None
        if force is not None:
            report.update(force_screen_metadata(force))
            report['physical_force_valid'] = not physical_force_exceeds(force, cutoff)
            report['max_force_cutoff'] = cutoff
        reason = report.get('topology_reject_reason') or report.get('surface_reject_reason')
        if reason is None and force is not None and not report['physical_force_valid']:
            reason = 'physical_force'
        report['screen_reject_reason'] = reason
        molecule.update_metadata(report)
        if screen_audit is not None:
            diagnostics = dict(report, **({'physical_forces': force['physical_forces']} if force is not None else {}))
            if step == 0:
                screen_audit.record(molecule, 'initial', atoms, 0, 0., diagnostics)
            screen_audit.record(molecule, 'rejected' if reason else 'last_valid', atoms, step, time_ps, diagnostics)
        return report, reason, observation

    active = list(range(count))
    if topology is not None or cutoff is not None:
        observe_physical = getattr(engine, "observe_physical", engine.observe)
        for index in list(active):
            try:
                report, reason, _ = inspect_frame(index, observe_physical)
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
        check_step = check * Ncheck
        if final_frame:
            check_step = min(check_step + 1, final_step)
        time_ps = check_step * dt / 1000
        for index in list(active):
            entry, schedule = state[index], schedules[index]
            try:
                if screening:
                    report, reason, observation = inspect_frame(index, engine.observe)
                    if reason:
                        # Keep the last valid uncertainty, since this frame's is physically suspect.
                        entry.update(decision=dict(outcome='rejected'), check=check,
                                     signals={**(entry['signals'] or {}),
                                              **{k:v for k,v in report.items() if k == 'Fmeanmax' or k.startswith('physical_F')}})
                        finish(index, 'rejected')
                        continue
                else:
                    observation = engine.observe(index)
                prediction = observation['prediction']
                signals = ensemble_statistics(prediction)[0]
                diagnostics = {}
                if selected_sources:
                    diagnostics = selected_force_diagnostics(
                        signals, selected_sources, force_atom_indices[index], member_forces=prediction["forces"][0],
                        atomic=(prediction.get("atomic_uncertainty") or [None])[0],
                        selected=(prediction.get("selected_force_uncertainty") or [None])[0])
                if screening:
                    signals.update({k:v for k,v in report.items() if k == 'Fmeanmax' or k.startswith('physical_F')})
                decision = assess_frame(
                    signals, observation['min_distance'], time_ps, sampler_config, settings,
                    trajectory_id=molecule_objects[index].get_moleculeid(), step=engine.steps[index])
                entry.update(signals=signals, decision=decision, min_distance=observation['min_distance'],
                             check=check)
                for key in (*PEAK_SIGNALS, *own_uncertainty):
                    if key in signals:  # gap_std only with a gap_uncertainty pair.
                        entry['peaks'][key] = max(signals[key], entry['peaks'].get(key, signals[key]))
                returning_frame = decision['outcome'] == 'eligible' or (final_frame and check_step == final_step)
                if returning_frame and prediction.get('atomic_uncertainty') is not None:
                    [atomic] = atomic_uncertainty_batch(prediction['atomic_uncertainty'],
                                                        [len(engine.atoms(index))])
                    if atomic:
                        diagnostics['atomic_uncertainty'] = {name: values.copy() for name, values in atomic.items()}
                if final_frame:
                    threshold_violation = decision['outcome'] != 'continue' and (
                        decision['uncertain'] or decision['distcrit'])
                    if threshold_violation and 'first_threshold_violation' not in entry:
                        atoms = engine.atoms(index)
                        entry['first_threshold_violation'] = {
                            'step': check_step, 'simulationtime': time_ps, 'distmin': observation['min_distance'],
                            **{key: decision[key] for key in ('distcrit', 'Ecrit', 'Fcrit', 'Fmcrit')},
                            'acquisition_trigger_flags': decision['acquisition_trigger_flags'],
                            'positions': atoms.get_positions(wrap=True),
                            'chemical_symbols': atoms.get_chemical_symbols(), 'cell': atoms.get_cell(),
                        }
                    if contact_frames and decision['distcrit'] and threshold_violation \
                            and 'contact_frame' not in entry:
                        entry['contact_frame'] = {'atoms': engine.snapshot(index), 'step': check_step,
                                                  'signals': signals, 'decision': decision, 'diagnostics': diagnostics,
                                                  'min_distance': observation['min_distance']}
                    if check_step == final_step:
                        entry.update(diagnostics)
                        finish(index, 'eligible')
                        continue
                    decision = dict(decision, outcome='continue')
                if decision['outcome'] == 'eligible':
                    if cohort:
                        candidate = _candidate(
                            molecule_objects[index], engine, index, signals, decision, time_ps,
                            entry['temps'][-1] if entry['temps'] else schedule['Tsrt'], diagnostics=diagnostics,
                            force_atom_array=force_atom_array)
                        if adaptive_ntk:
                            from alframework.samplers.adaptive_ntk import energy_ntk_row
                            basis, ridge, row = energy_ntk_row(prediction.get('energy_ntk', [None])[0])
                            candidate['energy_ntk'] = {'basis_id': basis, 'ridge': ridge, 'row': row.copy()}
                        entry['candidates'].append(candidate)
                        before = len(entry['candidates'])
                        entry['candidates'] = _prune(entry['candidates'], settings['selection'])
                        entry['candidates_pruned'] += before - len(entry['candidates'])
                        if keep_running:
                            # Keep sampling; the cohort selects among this trajectory's frames later.
                            decision = dict(decision, outcome='continue')
                    else:
                        entry.update(diagnostics)
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
        advance(min(Ncheck, final_step - check_step) if final_frame else Ncheck)
    for index in list(active):
        finish(index, 'completed')
    if cohort:
        for entry in state:
            if entry['outcome'] in ('completed', 'eligible') and entry['candidates']:
                entry['outcome'] = 'eligible'
            entry['candidates'] = _prune(entry['candidates'], settings['selection'])

    files, _ = metadata_settings(sampler_config)
    compact = returned_metadata(sampler_config) == 'compact'
    momenta = returned_momenta(sampler_config)
    records = []
    for index, molecule in enumerate(molecule_objects):
        records.append(_record(molecule, state[index], schedules[index], sampler_config, settings, engine, index,
                               started, None if seeds is None else seeds[index],
                               meta_dir if files == 'per_trajectory' else None, compact=compact,
                               momenta=momenta))
    bundle_dir = meta_dir if bundle_dir is None else bundle_dir
    if files == 'per_task' and bundle_dir is not None and molecule_objects:
        # One file for the whole task: the same dicts as the per-trajectory files, each with its molecule id.
        name = 'metadata-task-' + molecule_objects[0].get_moleculeid() + '.p'
        with open(os.path.join(bundle_dir, name), 'wb') as handle:
            pkl.dump([dict(record, moleculeid=molecule.get_moleculeid())
                      for record, molecule in zip(records, molecule_objects)], handle)
    return list(molecule_objects)


PEAK_SIGNALS = ('Es', 'Fs', 'Fsmax', 'gap_std')

# Each ``delta_realtime_step_<n>`` is the wall time of the STEP_TIMING_INTERVAL steps that end at step n, for
# analyses that extrapolate MD cost from step times. This includes checks between those steps, but excludes initial
# screening.
# In a batch all trajectories step together, so it is the batch wall time, shared by its active trajectories.
STEP_TIMING_INTERVAL = 100


def _record(molecule, entry, schedule, sampler_config, settings, engine, index, started, seed, meta_dir,
            compact=False, momenta=True):
    # With distcut_frames "label", a final-frame trajectory's labeled frame is its first close contact; the
    # frame fields describe that frame, and the trajectory-wide ones (timings, temperatures, peaks) the whole run.
    contact = entry.get('contact_frame') if entry['outcome'] == 'eligible' else None
    frame = contact or {'atoms': entry['finished_atoms'], 'signals': entry['signals'], 'decision': entry['decision'],
                        'min_distance': entry['min_distance'], 'step': engine.steps[index]}
    atoms, step = frame['atoms'], frame['step']
    signals = frame['signals'] or {}
    decision = frame['decision'] or {}
    dt, Ncheck = sampler_config['dt'], sampler_config['Ncheck']
    meta_dict = {"realtime_simulation": entry['finished_at'] - started,
                 "Es": signals.get('Es'), "Fs": signals.get('Fs'), "Fsmax": signals.get('Fsmax'),
                 "Fsrms": signals.get('Fsrms'),
                 **{f"peak_{key}": entry['peaks'].get(key) for key in PEAK_SIGNALS},
                 "distmin": frame['min_distance'],
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
    if settings['selection']['mode'] == 'final_frame':
        meta_dict['simulationtime'] = step * dt / 1000
        if contact is not None:
            meta_dict['trajectory_final_step'] = engine.steps[index]
        if 'first_threshold_violation' in entry:
            meta_dict['first_threshold_violation'] = entry['first_threshold_violation']
    if entry['outcome'] == 'eligible' and not cohort and 'ranking_score' in decision:
        meta_dict['ranking_score'] = decision['ranking_score']
    if cohort:
        meta_dict['acquisition_candidates'] = entry['candidates']
        if settings['selection'].get('selector') == 'adaptive_ntk':
            meta_dict['acquisition_candidates_pruned'] = entry['candidates_pruned']
    if entry['error'] is not None:
        meta_dict['sampling_error'] = entry['error']
    meta_dict.update(molecule.get_metadata())
    meta_dict.update(kinetic_temperature_K=entry.get('finished_kinetic_temperature_K'),
                     thermostat_temperature_K=entry['temps'][-1] if entry['temps'] else schedule['Tsrt'])
    meta_dict.update(signals)
    if entry['outcome'] == 'eligible':
        diagnostics = contact['diagnostics'] if contact is not None else entry
        for key in ('atomic_uncertainty', 'force_atom_indices', 'selected_force_uncertainty'):
            if key in diagnostics:
                meta_dict[key] = diagnostics[key]
    own = {name: peak for name, peak in entry['peaks'].items() if name not in PEAK_SIGNALS}
    if own:
        meta_dict['max_uncertainty'] = own
    for key in ('acquisition_trigger_flags', 'acquisition_cutoffs', 'acquisition_score_terms'):
        if key in decision:
            meta_dict[key] = decision[key]
    if sampler_config.get('gap_uncertainty') is not None:
        gap = sampler_config['gap_uncertainty']
        meta_dict['gap_pair'] = gap['pair'] if 'pair' in gap else gap['pairs']
    if meta_dir is not None:
        with open(meta_dir + "/metadata-" + molecule.get_moleculeid() + '.p', "wb") as handle:
            pkl.dump(meta_dict, handle)
    molecule.update_metadata({k: v for k, v in meta_dict.items() if k not in COMPACT_DROPPED} if compact else meta_dict)
    # In cohort mode frames travel as candidates; nothing is labeled until the cohort selects.
    if entry['outcome'] == 'eligible' and not cohort and not momenta and atoms.has('momenta'):
        atoms = atoms.copy()  # the engine keeps its own
        del atoms.arrays['momenta']
    molecule.update_atoms(atoms if entry['outcome'] == 'eligible' and not cohort else None)
    return meta_dict


def _candidate(molecule, engine, index, signals, decision, time_ps, thermostat_temperature_K=None, *,
               diagnostics=None, force_atom_array=None):
    atoms = engine.atoms(index)
    step = engine.steps[index]
    candidate = {"trajectory_id": molecule.get_moleculeid(), "step": int(step),
            "frame_id": frame_id(molecule.get_moleculeid(), step),
            "time_ps": float(time_ps), "ranking_score": float(decision['ranking_score']),
            "signals": {key: float(value) for key, value in signals.items()},
            "flags": {key: decision[key] for key in ('Ecrit', 'Fcrit', 'Fmcrit')},
            "numbers": atoms.get_atomic_numbers().copy(), "positions": atoms.get_positions().copy(),
            "cell": np.array(atoms.get_cell()), "pbc": atoms.get_pbc().copy(),
            "metadata": {**copy.deepcopy(molecule.get_metadata()),
                         **(diagnostics or {}),
                         "kinetic_temperature_K": kinetic_temperature(atoms),
                         "thermostat_temperature_K": thermostat_temperature_K,
                         **{key: decision[key] for key in ('acquisition_trigger_flags', 'acquisition_cutoffs',
                                                          'acquisition_score_terms') if key in decision}},
            "topology_atom_ids": None if 'topology_atom_ids' not in atoms.arrays else
                atoms.arrays['topology_atom_ids'].copy()}
    if force_atom_array is not None:
        candidate["force_atom_mask"] = (force_atom_array, atoms.arrays[force_atom_array].copy())
    return candidate


def _prune(candidates, selection):
    """Keep a trajectory's best frames within the selector's retention limit."""
    limit = worker_retention_limit(selection)
    if selection.get('selector') == 'adaptive_ntk':
        from alframework.samplers.adaptive_ntk import initial_energy_score
        return sorted(candidates, key=lambda c: (-initial_energy_score(c), c['trajectory_id'], c['step']))[:limit]
    return top_k(candidates, limit)


METADATA_FILES = ('per_trajectory', 'per_task', 'none')
META_DIR_LAYOUTS = ('per_task', 'flat')


def metadata_settings(sampler_config):
    """``(metadata_files, meta_dir_layout)``, checked: what a task writes to ``meta_dir``, and where.

    ``metadata_files``: ``"per_trajectory"`` (default) writes ``metadata-<id>.p`` per trajectory;
    ``"per_task"`` writes one ``metadata-task-<first id>.p`` per task, a list of the same dicts with a
    ``moleculeid`` key; ``"none"`` writes no metadata file (it still returns with each structure).
    ``meta_dir_layout``: ``"per_task"`` (default) puts a task's per-trajectory files (metadata and trajectories)
    in ``task-<first id>/``; ``"flat"`` puts them in ``meta_dir`` itself, for readers that expect one directory.
    """
    files = sampler_config.get('metadata_files', 'per_trajectory')
    layout = sampler_config.get('meta_dir_layout', 'per_task')
    if files not in METADATA_FILES:
        raise ValueError(f"metadata_files must be one of {METADATA_FILES}, not {files!r}")
    if layout not in META_DIR_LAYOUTS:
        raise ValueError(f"meta_dir_layout must be one of {META_DIR_LAYOUTS}, not {layout!r}")
    return files, layout


RETURNED_METADATA = ('full', 'compact')
# Per-trajectory arrays and lists the metadata file keeps but the coordinator doesn't read: with
# returned_metadata "compact" the returned molecule carries everything else.
COMPACT_DROPPED = ('positions', 'chemical_symbols', 'temps', 'denss', 'system_comp', 'first_threshold_violation')


def returned_metadata(sampler_config):
    """``"full"`` (default) or ``"compact"``: what metadata each returned molecule carries, checked.

    The coordinator holds every returned molecule until its labels are stored. ``"compact"`` leaves out
    ``COMPACT_DROPPED`` (the final positions and symbols again, the per-check temperature and density series,
    the composition), about two thirds of a trajectory's metadata in memory. The metadata file keeps the full
    record, so compact needs one: ``metadata_files`` "per_trajectory" or "per_task", and a ``meta_dir``.
    """
    mode = sampler_config.get('returned_metadata', 'full')
    if mode not in RETURNED_METADATA:
        raise ValueError(f"returned_metadata must be one of {RETURNED_METADATA}, not {mode!r}")
    if mode == 'compact':
        files, _ = metadata_settings(sampler_config)
        if files == 'none' or sampler_config.get('meta_dir') is None:
            raise ValueError('returned_metadata "compact" leaves fields out of the returned molecules, so the '
                             'metadata files must keep them: set meta_dir and metadata_files "per_trajectory" or '
                             '"per_task".')
    return mode


def returned_momenta(sampler_config):
    """Whether each returned structure keeps its MD momenta (default True), checked.

    Labeling and storage don't read them, and for 201-atom structures they are about 40% of what each
    returned structure costs the coordinator, and Parsl's interchange, until its labels are stored.
    False returns the final positions and cell without them; the engine's own atoms keep theirs.
    """
    keep = sampler_config.get('returned_momenta', True)
    if not isinstance(keep, bool):
        raise ValueError(f'returned_momenta must be true or false, not {keep!r}')
    return keep


def task_meta_dir(sampler_config, molecule_objects):
    """The directory this task writes its per-trajectory files (metadata, trajectories) to.

    With ``meta_dir_layout`` "per_task", a subdirectory ``task-<first molecule id>`` of ``meta_dir``, so that the
    tasks of a large run don't all create files in one directory at once, which a parallel filesystem
    serializes (on Lustre at 8 nodes, a task waited about 7 minutes to write its files). It is made only when
    the task may write such files.
    """
    meta_dir = sampler_config.get('meta_dir')
    files, layout = metadata_settings(sampler_config)
    trajectories = bool(sampler_config.get('trajectory_frequency')) and sampler_config.get('trajectory_interval') is not None
    if meta_dir is None or layout == 'flat' or not molecule_objects or not (files == 'per_trajectory' or trajectories):
        return meta_dir
    path = os.path.join(meta_dir, 'task-' + molecule_objects[0].get_moleculeid())
    os.makedirs(path, exist_ok=True)
    return path


def sample_trajectories(molecule_objects, models, sampler_config, engine_class=None, device=None):
    """Seed, schedule, and run MLMD for a group of molecules with a loaded ensemble.

    ``engine_class`` overrides the configured ``driver.engine`` (for tests).
    """
    bundle_dir = sampler_config.get('meta_dir')
    original_molecules = list(molecule_objects)
    molecule_objects = original_molecules
    sampler_config = dict(sampler_config, meta_dir=task_meta_dir(sampler_config, original_molecules))
    driver = driver_settings(sampler_config)
    states = state_settings(sampler_config)
    selection = acquisition_settings(sampler_config)['selection']
    if selection.get('selector') == 'adaptive_ntk':
        has_observations = any(callable(getattr(models, hook, None))
                               for hook in ('observe_uncertainty_batch', 'observe_uncertainty'))
        if not has_observations or states is not None or (driver['engine'] == 'ase' and engine_class is None):
            raise ValueError("Adaptive NTK needs a ground-state packed evaluator that reports energy_ntk rows.")
    require_members(models, sampler_config)
    from alframework.tools.molecular_topology import load_reference_topologies, validate_fixed_topology, topology_metadata
    from alframework.tools.molecules import MOLECULE_ID
    from alframework.tools.force_screen import validated_force_cutoff
    from alframework.tools.screen_audit import RuntimeScreenAudit
    from alframework.tools.surface_screen import anomaly_check_settings, prepare_surface_anomaly_check
    references = load_reference_topologies(
        sampler_config, [molecule.get_metadata().get(MOLECULE_ID) for molecule in original_molecules])
    topology = references if any(reference is not None for reference in references) else None
    surface_settings = anomaly_check_settings(sampler_config)
    check_surfaces = surface_settings is not None and any(
        'surface_builder' in molecule.get_metadata() for molecule in original_molecules)
    if sampler_config.get('max_force_cutoff') is not None:
        validated_force_cutoff(sampler_config['max_force_cutoff'])
    screening = topology is not None or check_surfaces or sampler_config.get('max_force_cutoff') is not None
    audit = RuntimeScreenAudit(original_molecules, sampler_config) if screening else None
    if states is not None:
        resolved = {resolve_selected_state(m, states) for m in original_molecules}
        if len(resolved) != 1:
            raise ValueError('A sampler task must share one selected state')
    surface_checks = None
    if topology is not None or check_surfaces:
        valid = []
        valid_references = []
        valid_surface_checks = []
        for molecule, reference in zip(original_molecules, references):
            atoms = molecule.get_atoms()
            # With topology checks on, every structure gets topology metadata, as before surface checks.
            report = topology_metadata(validate_fixed_topology(atoms, reference)) if topology is not None else {}
            try:
                surface_check = prepare_surface_anomaly_check(molecule, surface_settings)
            except ValueError as exc:
                surface_check = None
                report.update(surface_reject_reason='adsorbate_identity', anomaly_check_error=str(exc))
            if surface_check is not None:
                report.update(surface_check.inspect(atoms, 0, 0.))
            reason = report.get('topology_reject_reason') or report.get('surface_reject_reason')
            if report:
                report['screen_reject_reason'] = reason
                audit.record(molecule, 'initial', atoms, diagnostics=report)
                molecule.update_metadata(report)
                if surface_check is not None and reason is None:
                    audit.record(molecule, 'last_valid', atoms, diagnostics=report)
            if reason:
                molecule.update_metadata(dict(trajectory_outcome='rejected', step=0, simulationtime=0.,
                                              acquisition_candidates=[]))
                audit.record(molecule, 'rejected', atoms, diagnostics=report)
                molecule.update_atoms(None)
            else:
                valid.append(molecule)
                valid_references.append(reference)
                valid_surface_checks.append(surface_check)
        molecule_objects = valid
        topology = valid_references if topology is not None else None
        surface_checks = valid_surface_checks if any(check is not None for check in valid_surface_checks) else None
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
        atoms = molecule.get_atoms()
        fixed = has_fixed_atoms(atoms)
        if sampler_config.get('end_dens') is not None and ('surface_builder' in molecule.get_metadata() or fixed):
            raise ValueError('Sampling slabs or structures with FixAtoms requires a fixed cell. Set end_dens to null.')
        if fixed and sampler_config.get('translate_to_center', False):
            raise ValueError('Sampling with FixAtoms requires translate_to_center=false.')
        seed = trajectory_seed(molecule.get_moleculeid(), sampler_config.get('random_seed'))
        rng = trajectory_rng(seed)
        schedules.append(draw_schedule(sampler_config, rng))
        trajectory_files.append(rng.rand() <= sampler_config.get('trajectory_frequency', 0))
        seeds.append(seed)
        rngs.append(rng)
        if sampler_config.get('translate_to_center', False):
            atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())

    maxt = sampler_config['maxt']
    temperatures = [annealing_schedule(0.0, maxt, s['Tamp'], s['Tper'], s['Tsrt'], s['Tend']) for s in schedules]
    for molecule, temperature in zip(molecule_objects, temperatures):
        molecule.update_metadata(initialize_velocities(
            molecule.get_atoms(), temperature, sampler_config.get('velocity_initialization', 'preserve'),
            molecule_id=molecule.get_moleculeid(), random_seed=sampler_config.get('random_seed'),
            initial_velocity_temperature_K=sampler_config.get('initial_velocity_temperature_K')))
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
                              topology=topology, surface_checks=surface_checks, screen_audit=audit, bundle_dir=bundle_dir)
        entered = molecule_objects[0].get_metadata()['sampling_entered_dynamics_width']
        for molecule in original_molecules:
            molecule.update_metadata(dict(sampling_entered_dynamics_width=entered))
        return original_molecules
    finally:
        if audit is not None:audit.write()


@alf_task('sampler', declares=('returned_metadata', 'returned_momenta', 'metadata_files', 'frame_spool'))
def mlmd_trajectories_task(molecule_objects, sampler_config, model_path, current_model_id, gpus_per_node=None,
                           frame_spool=None):
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
    results = sample_trajectories(list(molecule_objects), models, sampler_config, device=device)
    if frame_spool is not None:
        from alframework.tools.sampled_frames import stage_sampled_frames

        return stage_sampled_frames(results, *frame_spool)
    return results


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
