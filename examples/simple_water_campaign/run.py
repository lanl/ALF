#!/usr/bin/env python3
"""Prepare or supervise the opt-in twelve-case Darwin validation campaign."""
import argparse
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_support import (HERE, SOURCE, Case, atomic_json, cleanup_jobs, completion, darwin_site,
    event, events, fingerprint, h5_inventory, matrix, owned_jobs, prepare_run, read_json,
    seed_case, source_identity, submitted_width, validate_storage)


def audit_submissions(dfk, run):
    original = dfk.submit
    signature = inspect.signature(original)

    def submit(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        func = bound['func']
        width, ids = submitted_width(func, bound['app_args'], bound['app_kwargs'])
        future = original(*args, **kwargs)
        values = dict(task_id=future.tid, name=func.__name__, width=width, ids=ids)
        event('task_submitted', run, **values)

        def done(completed):
            event('task_done', run, success=not completed.cancelled() and completed.exception() is None, **values)

        future.add_done_callback(done)
        return future

    dfk.submit = submit


def worker(run, operation, seed=None):
    """Run the real entrypoint, with audit and guaranteed resource cleanup."""
    import parsl
    run = Path(run).resolve()
    os.chdir(run)
    os.environ['ALF_CAMPAIGN_RUN'] = str(run)
    original_load = parsl.load
    original_sleep = time.sleep

    def observed_sleep(seconds):
        frame = sys._getframe(1)
        if frame.f_globals.get('__name__') == '__main__' and 'master_loop_iter' in frame.f_globals:
            queues = {}
            for name in ('builder_task_queue', 'sampler_task_queue', 'QM_task_queue', 'ML_task_queue'):
                queue = frame.f_globals.get(name)
                if queue is not None:
                    queues[name] = dict(total=queue.get_number(), queued=queue.get_queued_number(),
                                        running=queue.get_running_number(), done=queue.get_completed_number())
            event('queue_poll', run, queues=queues)
        return original_sleep(seconds)

    def load(config):
        dfk = original_load(config)
        audit_submissions(dfk, run)
        return dfk

    parsl.load = load
    time.sleep = observed_sleep
    try:
        if operation == 'coordinator':
            sys.argv = ['alframework', '--master', str(run / 'master_config.json')]
            runpy.run_module('alframework', run_name='__main__')
        else:
            from campaign_resources import config
            parsl.load(config)
            if operation == 'preflight':
                preflight(run)
            elif operation == 'prediction':
                from campaign_tasks import prediction_check
                report = prediction_check(str(Path(seed) / 'models/model-0000'),
                                          str(Path(seed) / 'preflight.extxyz')).result()
                atomic_json(run / 'prediction.json', report)
            else:
                raise ValueError(operation)
    except KeyboardInterrupt:
        event('coordinator_interrupted', run)
    except BaseException as exc:
        event('worker_error', run, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        parsl.load = original_load
        time.sleep = original_sleep
        try:
            dfk = parsl.dfk()
        except Exception:
            pass
        else:
            event('queue_snapshot', run, tasks=[dict(task_id=key, state=str(value['status']))
                                                for key, value in dfk.tasks.items()])
            try:
                dfk.cleanup()
            finally:
                parsl.clear()
        event('worker_stopped', run)


def preflight(run):
    import numpy as np
    from ase.io import write
    from parsl import python_app
    from alframework.tools.labeling import label_sequentially
    from alframework.tools.tools import build_input_dict, load_config_file
    from campaign_tasks import build_water, label_water
    master = load_config_file(run / 'master_config.json')
    builder = load_config_file(master['builder_config_path'], master['master_directory'])
    qm = load_config_file(master['QM_config_path'], master['master_directory'])
    ids = [f'preflight-{i}' for i in range(4)]
    arguments = build_input_dict(build_water.func, [dict(moleculeids=ids, builder_config=builder), builder],
                                 raise_on_fail=True)
    molecules = build_water(**arguments).result()
    assert [m.get_moleculeid() for m in molecules] == ids
    from alframework.samplers.geometry import minimum_distance
    assert all(minimum_distance(m.get_atoms()) >= .7 for m in molecules)
    write(run / 'preflight.extxyz', [m.get_atoms() for m in molecules])
    task_inputs = dict(QM_config=qm, QM_scratch_dir=str(run / 'preflight_orca'),
                       properties_list=master['properties_list'])
    singles = [label_water(molecule_object=copy.deepcopy(m), **task_inputs) for m in molecules]
    single_results = [future.result() for future in singles]
    packed_task = python_app(executors=['alf_QM_executor'])(label_sequentially)
    task_inputs['QM_scratch_dir'] = str(run / 'preflight_orca_packed')
    packed = packed_task(copy.deepcopy(molecules), label_water.func, task_inputs).result()
    comparisons = []
    for original, first, second in zip(molecules, single_results, packed):
        assert first.check_convergence() and second.check_convergence()
        assert first.get_moleculeid() == second.get_moleculeid() == original.get_moleculeid()
        np.testing.assert_array_equal(first.get_atoms().pbc, original.get_atoms().pbc)
        np.testing.assert_array_equal(second.get_atoms().cell, original.get_atoms().cell)
        a, b = first.get_results(), second.get_results()
        de, df = abs(a['energy']-b['energy']), float(np.max(np.abs(a['forces']-b['forces'])))
        assert de <= qm['Ediff'] and df <= qm['Fdiff']
        comparisons.append(dict(id=first.get_moleculeid(), energy_error_hartree=de, force_error_hartree_bohr=df))
    atomic_json(run / 'preflight.json', dict(passed=True, comparisons=comparisons))


def validate_checkpoints(run, result):
    """Load every newly trained member and evaluate a recorded structure on CPU."""
    import numpy as np
    from ase import Atoms
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_load_ensemble
    checked = validate_storage(run)
    assert checked >= len(result['qualifying_ids'])
    if (Path(run) / 'seed.json').exists():
        seed = read_json(Path(run) / 'seed.json')
        assert all(fingerprint(Path(run) / name) == expected for name, expected in seed['files_sha256'].items())
    records = events(run)
    rec = next(r for r in records if r['kind'] == 'builder_end')
    geometry = rec['structures'][0]
    atoms = Atoms(numbers=geometry['numbers'], positions=geometry['positions'],
                  cell=geometry['cell'], pbc=geometry['pbc'])
    directory = Path(run) / 'models' / f"model-{result['model_id']:04d}"
    models = HIPNN_ASE_load_ensemble(str(directory), device='cpu')
    assert len(models) == 4
    for model in models:
        model.calculate(atoms, properties=['energy', 'forces'])
        assert np.isfinite(model.results['energy'])
        assert np.isfinite(model.results['forces']).all()
    return True


def stop_process(process, run, grace=90):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            # The group contains only this worker and its local Parsl processes.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
    jobs = cleanup_jobs(run)
    atomic_json(Path(run) / 'cleanup.json', dict(verified=True, job_ids=jobs, returncode=process.returncode))


def launch(run, operation, deadline, bootstrap=False, seed=None):
    run = Path(run)
    spec = read_json(run / 'run.json')
    (run / 'cleanup.json').unlink(missing_ok=True)
    environment = dict(os.environ, ALF_CAMPAIGN_RUN=str(run), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', HIPPYNN_USE_CUSTOM_KERNELS='False',
                       PYTHONPATH=spec['source'] + ':' + str(Path(spec['source']) / 'examples/simple_water_campaign'))
    command = [spec['site']['base_python'], '-B', str(Path(spec['source']) / 'examples/simple_water_campaign/run.py'),
               '--worker', operation, '--run-root', str(run)]
    if seed is not None:
        command += ['--seed', str(seed)]
    with (run / (operation + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=run, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        outcome = None
        try:
            while time.time() < deadline:
                if operation == 'coordinator':
                    outcome = completion(run, bootstrap=bootstrap)
                    if outcome is not None:
                        validate_checkpoints(run, outcome)
                        break
                else:
                    marker = run / ('preflight.json' if operation == 'preflight' else 'prediction.json')
                    if marker.is_file() and process.poll() == 0:
                        outcome = read_json(marker)
                        break
                if process.poll() is not None:
                    raise RuntimeError(f'{operation} exited with {process.returncode}; see {log.name}')
                time.sleep(spec['profile']['poll_seconds'])
        finally:
            stop_process(process, run)
    if outcome is None:
        raise TimeoutError(f'{operation} reached its deadline')
    return outcome


def freeze_seed(bootstrap, destination, accepted):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Seed already exists: {destination}')
    destination.mkdir(parents=True)
    files = {}
    for original, expected in accepted['training']['input']['files_sha256'].items():
        original = Path(original)
        assert fingerprint(original) == expected
        relative = Path('h5store') / original.name
        (destination / relative).parent.mkdir(exist_ok=True)
        shutil.copy2(original, destination / relative)
        files[str(relative)] = expected
    model = Path(bootstrap) / 'models/model-0000'
    shutil.copytree(model, destination / 'models/model-0000')
    for path in sorted((destination / 'models').rglob('*')):
        if path.is_file():
            files[str(path.relative_to(destination))] = fingerprint(path)
    shutil.copy2(Path(bootstrap) / 'preflight.extxyz', destination / 'preflight.extxyz')
    files['preflight.extxyz'] = fingerprint(destination / 'preflight.extxyz')
    atomic_json(destination / 'seed.json', dict(files_sha256=files, ids=accepted['training']['input']['ids']))


def case_metrics(run):
    records = events(run)
    started = {r['id']: r['time'] for r in records if r['kind'] == 'qm_start'}
    returned = {name: r['time'] for r in records if r['kind'] == 'sampler_end' for name in r['selected_ids']}
    latencies = {name: started[name]-value for name, value in returned.items() if name in started}
    spans = [r['time'] for r in records]
    elapsed = max(spans)-min(spans) if len(spans) > 1 else 0.
    converged = sum(r['kind'] == 'qm_end' and r['converged'] for r in records)
    return dict(qm_attempts=sum(r['kind'] == 'qm_start' for r in records),
                qm_converged=converged, converged_labels_per_elapsed_second=converged/elapsed if elapsed else None,
                stored=len(h5_inventory(Path(run) / 'h5store')['ids']),
                sampler_widths=[r['width'] for r in records if r['kind'] == 'sampler_end'],
                qm_task_widths=[r['width'] for r in records if r['kind'] == 'task_done' and
                               r['name'] in ('label_water', 'label_sequentially') and r['success']],
                sampler_return_to_qm_start_seconds=latencies,
                queue_occupancy=[r for r in records if r['kind'] == 'queue_poll'])


def suite(root, selected=None):
    root = Path(root).resolve()
    site = read_json(root / 'site.json')
    profile = read_json(root / 'profile.json')
    outcomes = {}
    if (root / 'execution.json').exists():
        raise ValueError('This root has already executed. Preserve it and choose a fresh run root.')
    atomic_json(root / 'execution.json', dict(start=time.time(), selected=selected))
    atomic_json(root / 'result.json', dict(status='running', phase='bootstrap', cases={}))
    print('Starting shared bootstrap and real ORCA preflight.', flush=True)
    stage_start = time.time()
    bootstrap = root / 'bootstrap'
    try:
        launch(bootstrap, 'preflight', stage_start + profile['stage_seconds'] - profile['cleanup_seconds'])
        accepted = launch(bootstrap, 'coordinator', stage_start + profile['stage_seconds'] - profile['cleanup_seconds'],
                          bootstrap=True)
        freeze_seed(bootstrap, root / 'seed', accepted)
        atomic_json(bootstrap / 'result.json', dict(status='passed', accepted=accepted, elapsed=time.time()-stage_start))
        print('Shared bootstrap accepted; initial ensemble and training inputs frozen.', flush=True)
    except Exception as exc:
        atomic_json(bootstrap / 'result.json', dict(status='incomplete' if isinstance(exc, TimeoutError) else 'failed',
                                                   error=str(exc), elapsed=time.time()-stage_start))
        atomic_json(root / 'result.json', dict(status='incomplete', cases={}, bootstrap_error=str(exc)))
        raise
    for engine in ('ase', 'torchsim', 'alchemi'):
        cases = [case for case in matrix() if case.engine == engine and (selected is None or case.name == selected)]
        if not cases:
            continue
        stage_deadline = time.time() + profile['stage_seconds']
        for case in cases:
            run = root / 'cases' / case.name
            started = time.time()
            print(f'Starting {case.name}.', flush=True)
            atomic_json(root / 'result.json', dict(status='running', phase=engine, current_case=case.name, cases=outcomes))
            deadline = min(started + profile['case_seconds'], stage_deadline)
            if deadline - started <= profile['cleanup_seconds']:
                outcome = dict(status='incomplete', reason='engine stage deadline reached before execution')
            else:
                try:
                    seed_case(root / 'seed', run)
                    # Each case verifies the actual sampler environment against common recorded frames.
                    launch(run, 'prediction', deadline-profile['cleanup_seconds'], seed=root / 'seed')
                    accepted = launch(run, 'coordinator', deadline-profile['cleanup_seconds'])
                    outcome = dict(status='passed', accepted=accepted)
                except Exception as exc:
                    outcome = dict(status='incomplete' if isinstance(exc, TimeoutError) else 'failed', error=str(exc))
                    if not (run / 'cleanup.json').exists() or not read_json(run / 'cleanup.json')['verified']:
                        raise RuntimeError(f'Cleanup unverified for {case.name}; refusing further allocations') from exc
                outcome.update(elapsed=time.time()-started, metrics=case_metrics(run))
            atomic_json(run / 'result.json', outcome)
            print(f'{case.name}: {outcome["status"]}.', flush=True)
            outcomes[case.name] = outcome
            atomic_json(root / 'result.json', dict(status='running', cases=outcomes))
    expected = {case.name for case in matrix()} if selected is None else {selected}
    passed = set(outcomes) == expected and all(value['status'] == 'passed' for value in outcomes.values())
    atomic_json(root / 'result.json', dict(status='passed' if passed else 'incomplete', cases=outcomes,
                                          complete_matrix=passed and len(expected) == 12))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--case', choices=[case.name for case in matrix()])
    parser.add_argument('--site', type=Path, help='JSON overrides for Darwin executable/resource defaults')
    parser.add_argument('--worker', choices=['coordinator', 'preflight', 'prediction'], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.run_root, args.worker, args.seed)
        return
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profile = read_json(HERE / 'profile.json')
        site = darwin_site()
        if args.site:
            site.update(read_json(args.site))
        for name, value in [('profile.json', profile), ('site.json', site), ('source.json', source_identity())]:
            if (root / name).exists():
                if read_json(root / name) != value:
                    raise ValueError(f'Existing campaign manifest differs: {name}')
            else:
                atomic_json(root / name, value)
        prepare_run(root / 'bootstrap', None, profile, site)
        for case in matrix():
            prepare_run(root / 'cases' / case.name, case, profile, site)
        if args.prepare_only:
            print(f'Prepared 12 immutable cases under {root}; no jobs submitted.')
            return
        for key in ('base_python', 'torchsim_python', 'alchemi_python', 'orca'):
            if not Path(site[key]).is_file():
                raise FileNotFoundError(site[key])
        if not suite(root, args.case):
            sys.exit(1)


if __name__ == '__main__':
    main()
