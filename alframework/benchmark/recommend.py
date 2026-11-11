"""Turn benchmark measurements into suggested settings. Pure functions, with no Torch."""
import math

NEAR_BEST = 0.9  # A setting within 10% of the best throughput is good enough, and smaller is safer.


def _fits(record, memory_gib, memory_fraction):
    if record.get('result') != 'passed':
        return False
    reserved = record.get('peak_reserved_gib')
    return memory_gib is None or reserved is None or reserved <= memory_fraction * memory_gib


def _smallest_near_best(rows, size_key):
    """The smallest size whose throughput is within 10% of the best, from (size, throughput) rows."""
    if not rows:
        return None
    best = max(rate for _, rate in rows)
    return min(size for size, rate in rows if rate >= NEAR_BEST * best)


def training_suggestions(records, *, memory_gib=None, memory_fraction=0.8, train_rows=40000, epochs=300,
                         walltime_hours=16.0, run_batch=None):
    """One suggestion per profile: a batch size, its epoch time, whether training fits the walltime,
    and, for the run's own batch size, how many GPUs per member it needs."""
    suggestions = {}
    for profile in sorted({r['profile'] for r in records if r['kind'] == 'training'}):
        rows = [r for r in records if r['kind'] == 'training' and r['profile'] == profile]
        fitting = {r['batch']: r for r in rows if _fits(r, memory_gib, memory_fraction)}
        batch = _smallest_near_best([(b, b / r['train_step_s']) for b, r in fitting.items()], 'batch')
        suggestion = dict(profile=profile, batch_size=batch, fitting_batches=sorted(fitting))
        if batch is not None:
            minutes = math.ceil(train_rows / batch) * fitting[batch]['train_step_s'] / 60
            suggestion.update(epoch_minutes=minutes, training_hours=epochs * minutes / 60,
                              fits_walltime=epochs * minutes / 60 <= walltime_hours)
        if run_batch:
            suggestion['run_batch_size'] = run_batch
            suggestion['gpus_per_member'] = gpus_for_batch(run_batch, fitting)
        suggestions[profile] = suggestion
    return suggestions


def gpus_for_batch(batch, fitting, largest=8):
    """The fewest GPUs per member, dividing ``batch``, whose per-GPU share was measured to fit; else None.

    An estimate: it leaves out the time GPUs spend exchanging gradients.
    """
    for gpus in range(1, largest + 1):
        if batch % gpus == 0 and batch // gpus in fitting:
            return gpus
    return None


def sampling_suggestions(records, *, memory_gib=None, memory_fraction=0.8, sampler_gpus=1):
    """One suggestion per profile: trajectories per GPU, and the matching parallel_samplers."""
    suggestions = {}
    for profile in sorted({r['profile'] for r in records if r['kind'] == 'sampling'}):
        rows = [r for r in records if r['kind'] == 'sampling' and r['profile'] == profile]
        fitting = {r['trajectories']: r for r in rows if _fits(r, memory_gib, memory_fraction)}
        per_gpu = _smallest_near_best([(n, n / r['step_s']) for n, r in fitting.items()], 'trajectories')
        suggestion = dict(profile=profile, trajectories_per_gpu=per_gpu, fitting=sorted(fitting))
        if per_gpu is not None:
            suggestion.update(structures_per_second=per_gpu / fitting[per_gpu]['step_s'],
                              parallel_samplers=per_gpu * sampler_gpus, sampler_gpus=sampler_gpus)
        suggestions[profile] = suggestion
    return suggestions
