"""Opt-in evidence collection and acceptance for the remaining release gates."""
from contextlib import contextmanager
from pathlib import Path
import copy
import resource

from campaign_support import atomic_json, event, fingerprint, h5_inventory, read_json


def memory():
    """Process lifetime peak RSS, not a sum of process or GPU allocations."""
    return dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def verify_seed(seed, destination):
    """Validate a closed, actually trained water seed before any submission."""
    import numpy as np
    from ase.io import read
    from alframework.ml_interfaces.hippynn_interface import HIPNN_ASE_calculator
    seed, destination = Path(seed).resolve(), Path(destination).resolve()
    manifest = read_json(seed / 'seed.json')
    for relative, expected in manifest['files_sha256'].items():
        path = (seed / relative).resolve()
        if not path.is_relative_to(seed) or fingerprint(path) != expected:
            raise ValueError(f'Invalid seed file: {relative}')
    inventory = h5_inventory(seed / 'h5store')
    if set(inventory['ids']) != set(manifest['ids']) or len(inventory['ids']) < 32:
        raise ValueError('Seed needs at least 32 unique bootstrap IDs')
    if not all(name.startswith('mol-boot-') for name in inventory['ids']):
        raise ValueError('Seed contains non-bootstrap structures')
    bootstrap = seed.parent / 'bootstrap'
    accepted = read_json(bootstrap / 'result.json')
    training = accepted['accepted']['training']
    if accepted['status'] != 'passed' or training['model_id'] != 0:
        raise ValueError('Seed bootstrap was not accepted')
    if set(training['input']['ids']) != set(inventory['ids']):
        raise ValueError('Seed differs from the initial training input')
    if {Path(p).name: h for p, h in training['input']['files_sha256'].items()} != {
            Path(p).name: h for p, h in inventory['files_sha256'].items()}:
        raise ValueError('Seed HDF5 fingerprints differ from training input')
    members = training['members']
    if len(members) != 4 or not all(m['complete'] and m['epochs'] > 0 for m in members):
        raise ValueError('Seed needs four actually trained members')
    for name in ('hippynn_config.json', 'mlmd_config.json'):
        if read_json(bootstrap / name) != read_json(destination / name):
            raise ValueError(f'Incompatible seed configuration: {name}')
    a, b = (read_json(p / 'builder_config.json') for p in (bootstrap, destination))
    library_a, library_b = (Path(v.pop('molecule_library_path')) for v in (a, b))
    if a != b or {str(p.relative_to(library_a)): fingerprint(p) for p in library_a.rglob('*') if p.is_file()} != {
            str(p.relative_to(library_b)): fingerprint(p) for p in library_b.rglob('*') if p.is_file()}:
        raise ValueError('Incompatible seed builder or fragment library')
    original = read_json(bootstrap / 'master_config.json')
    target = read_json(destination / 'master_config.json')
    if original['properties_list'] != target['properties_list']:
        raise ValueError('Incompatible seed property units')
    original_qm, target_qm = (read_json(p / 'orca_config.json') for p in (bootstrap, destination))
    for key in ('orcasimpleinput', 'orcablocks', 'Ediff', 'Fdiff'):
        if original_qm.get(key) != target_qm.get(key):
            raise ValueError(f'Incompatible seed labeling: {key}')
    structures = read(seed / 'preflight.extxyz', index=':')
    directories = sorted((seed / 'models/model-0000').glob('model-*'))
    if [d.name for d in directories] != [m['member'] for m in members]:
        raise ValueError('Initial member identity mismatch')
    for directory, member in zip(directories, members):
        for name, expected in member['files_sha256'].items():
            if fingerprint(directory / name) != expected:
                raise ValueError('Initial checkpoint differs from training report')
        model = HIPNN_ASE_calculator(str(directory), device='cpu')
        for atoms in structures:
            model.calculate(atoms, properties=['energy', 'forces'])
            if not np.isfinite(model.results['energy']) or not np.isfinite(model.results['forces']).all():
                raise ValueError('Initial member predictions are not finite')
    return dict(source=str(seed), seed_manifest_sha256=fingerprint(seed / 'seed.json'),
                initial_training=training, verified_members=4)


def install_cohort_audit(run):
    """Observe the real coordinator without changing selection or admission."""
    from alframework.tools.cohorts import CohortPool
    original_admit, original_record, original_ready = CohortPool.admit, CohortPool.record, CohortPool.ready

    def admit(pool, molecule, model_id):
        pinned = original_admit(pool, molecule, model_id)
        event('cohort_admitted', run, id=molecule.get_moleculeid(), current_model_id=model_id,
              pinned_model_id=pinned, metadata=molecule.get_metadata(), **memory())
        return pinned

    def record(pool, molecule):
        original_record(pool, molecule)
        event('cohort_terminal', run, id=molecule.get_moleculeid(), metadata=molecule.get_metadata(), **memory())

    def ready(pool):
        previous = len(pool.finished)
        result = original_ready(pool)
        for summary in pool.finished[previous:]:
            from campaign_tasks import geometry
            frames = [dict(**geometry(m), metadata=m.get_metadata()) for m in result
                      if m.get_metadata()['cohort_id'] == summary['cohort_id']]
            event('cohort_selected', run, summary=summary, frames=frames, **memory())
        return result

    CohortPool.admit, CohortPool.record, CohortPool.ready = admit, record, ready


@contextmanager
def sampling_audit(molecules, fault=''):
    """Capture candidates before pruning; optionally fail one stepped trajectory."""
    from alframework.samplers import batched_mlmd as module
    original_candidate, original_engine = module._candidate, module.make_engine
    target = 'mol-0000-0000000000'
    eligible = set()
    arrays_bytes = 0
    original_count = 0

    def candidate(*args, **kwargs):
        nonlocal arrays_bytes, original_count
        result = original_candidate(*args, **kwargs)
        eligible.add(result['trajectory_id'])
        arrays_bytes += sum(result[key].nbytes for key in ('numbers', 'positions', 'cell', 'pbc'))
        original_count += 1
        event('candidate_created', candidate=result, generated_count=original_count,
              generated_geometry_bytes=arrays_bytes, **memory())
        return result

    def engine(*args, **kwargs):
        value = original_engine(*args, **kwargs)
        step = value.step
        injected = False

        def observed_step(indices, steps):
            nonlocal injected
            indices = list(indices)
            failures = {}
            if fault == 'trajectory_failure' and target in eligible and not injected:
                for index in indices:
                    if molecules[index].get_moleculeid() == target:
                        failures[index] = RuntimeError('controlled campaign trajectory failure')
                        indices.remove(index)
                        injected = True
                        event('fault_injected', id=target, fault=fault)
                        break
            if indices:
                failures.update(step(indices, steps))
            return failures

        value.step = observed_step
        return value

    module._candidate, module.make_engine = candidate, engine
    try:
        yield
    finally:
        module._candidate, module.make_engine = original_candidate, original_engine


def replay_cohorts(records, sampler_config=None):
    """Replay exact recorded candidates; comparison does not rerun stochastic MD."""
    from ase import Atoms
    from alframework.tools.cohorts import CohortPool
    from alframework.tools.molecules_class import MoleculesObject
    from alframework.samplers.acquisition import top_k
    if sampler_config is not None:
        import numpy as np
        from alframework.samplers.acquisition import acquisition_settings, assess_frame
        from alframework.samplers.geometry import minimum_distance
        settings = acquisition_settings(sampler_config)
    admissions = [r for r in records if r['kind'] == 'cohort_admitted']
    terminals = {r['id']: r for r in records if r['kind'] == 'cohort_terminal'}
    generated = {}
    for record in records:
        if record['kind'] == 'candidate_created':
            candidate = record['candidate']
            if sampler_config is not None:
                atoms = Atoms(numbers=candidate['numbers'], positions=candidate['positions'],
                              cell=candidate['cell'], pbc=candidate['pbc'])
                assessed = assess_frame(candidate['signals'], minimum_distance(atoms),
                                        candidate['time_ps'], sampler_config, settings)
                if assessed['outcome'] != 'eligible' or not np.isclose(
                        assessed['ranking_score'], candidate['ranking_score'], rtol=1e-12, atol=1e-12):
                    raise ValueError('Recorded candidate eligibility/ranking differs from configured policy')
            generated.setdefault(candidate['trajectory_id'], []).append(candidate)
    reports = []
    for selection in (r for r in records if r['kind'] == 'cohort_selected'):
        summary = selection['summary']
        names = [r['id'] for r in admissions if r['metadata']['cohort_id'] == summary['cohort_id']]
        expected = [c['frame_id'] for c in top_k([c for n in names for c in generated.get(n, [])], 4)]
        if expected != summary['selected']:
            raise ValueError('Selection differs from unpruned global top-K')
        for width in (1, 2, 4, 8, 32, 50):
            for reverse in (False, True):
                pool = CohortPool(dict(cohort_size=8, k=4))
                for name in names:
                    pool.admit(MoleculesObject(Atoms('H'), name), summary['model_id'])
                groups = [names[i:i+width] for i in range(0, len(names), width)]
                for group in reversed(groups) if reverse else groups:
                    for name in reversed(group) if reverse else group:
                        m = MoleculesObject(Atoms('H'), name)
                        m.update_metadata(copy.deepcopy(terminals[name]['metadata']))
                        pool.record(m)
                selected = pool.ready()
                if [m.get_metadata()['frame_id'] for m in selected] != expected:
                    raise ValueError('Replay changed selected identities')
        reports.append(dict(cohort_id=summary['cohort_id'], selected=expected))
    return reports


def campaign_accepted(case, records, good, stored, inputs):
    if case.campaign == 'cohort' or case.fault == 'trajectory_failure':
        selections = [r for r in records if r['kind'] == 'cohort_selected']
        if len(selections) < 2:
            return False
        selected = {f['id']: f for r in selections for f in r['frames']}
        if not good <= set(selected):
            raise ValueError('Unselected candidate reached the accepted lineage')
        generated = {r['candidate']['frame_id']: r['candidate'] for r in records if r['kind'] == 'candidate_created'}
        for name in good:
            frame = selected[name]
            candidate = generated.get(frame['metadata']['frame_id'])
            if candidate is None or candidate['trajectory_id'] != frame['metadata']['trajectory_id']:
                raise ValueError('Selected frame has no generated candidate')
            import numpy as np
            for key in ('numbers', 'positions', 'cell', 'pbc'):
                np.testing.assert_array_equal(frame[key], candidate[key])
        if case.policy == 'continue':
            counts = {}
            for record in records:
                if record['kind'] == 'candidate_created':
                    name = record['candidate']['trajectory_id']
                    counts[name] = counts.get(name, 0) + 1
            if max(counts.values(), default=0) < 2:
                return False
    if case.fault:
        faults = [r for r in records if r['kind'] == 'fault_injected' and r['fault'] == case.fault]
        if len(faults) != 1:
            return False
        failed = faults[0]['id']
        if case.fault.startswith('label_'):
            if case.fault == 'label_unconverged' and not faults[0].get('backend_converged'):
                return False
            if failed in stored or failed in inputs:
                raise ValueError('Controlled failed label reached storage/training')
            # At least one sibling in the same real packed task survives into training.
            packs = [r for r in records if r['kind'] == 'task_submitted' and failed in r.get('ids', [])]
            if not any(set(r['ids']) & good for r in packs if r['name'] == 'label_sequentially'):
                return False
        elif not any(r['kind'] == 'cohort_selected' and any(
                f['id'] in good and f['metadata']['trajectory_id'] == failed for f in r['frames']) for r in records):
            return False
    if case.campaign == 'pyseqm' and case.qm_width == 4:
        return any(r['kind'] == 'native_solve' and r['width'] == 4 and r['success'] for r in records)
    return True
