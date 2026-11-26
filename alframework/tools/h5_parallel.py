"""Read HDF5 label groups from their backing parts, optionally in parallel.

A stored label file holds one group per composition, compressed with gzip. Reading it decompresses
each dataset in one thread, so a large shard reads slowly. Here each group is read by its own worker
process, which returns the group's datasets exactly as hippynn's ``AniDataLoader`` gives them, in the
file's group order.
The file is read in place; nothing is written.

``ALF_H5_READ_WORKERS`` sets the number of processes (1 reads serially, as before). By default files
of at least ``PARALLEL_MIN_BYTES`` use one process per available core, up to one per group. A label
snapshot counts the parts it references, and its virtual datasets are read from those parts directly.
"""
import math
import os
from collections import defaultdict, deque
from contextlib import nullcontext

READ_WORKERS = 'ALF_H5_READ_WORKERS'
PARALLEL_MIN_BYTES = 256 * 2**20


def _label_file_groups(handle, group_keys):
    """The structure groups of a label file ALF wrote, by link names alone.

    ALF's label shards and snapshots hold one level of formula groups, each of datasets only, so this selects the
    same groups as the walk in :func:`_structure_groups`, in the same order, without opening every dataset. In a
    snapshot each one is a virtual dataset, and opening it parses its mapping: with one group per label, that walk
    took longer than reading the labels.
    """
    for name in handle:
        member = handle[name]
        if (any(key in member for key in group_keys) if group_keys is not None else len(member)):
            yield f'/{name}', member


def _structure_groups(group, path='/', group_keys=None):
    import h5py
    from alframework.tools.label_hdf5 import SCHEMA_VERSION
    if path == '/' and SCHEMA_VERSION in group.attrs:
        yield from _label_file_groups(group, group_keys)
        return
    if group_keys is not None:
        if any(isinstance(group.get(key), h5py.Dataset) for key in group_keys):
            yield path, group
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                yield from _structure_groups(item, f'{path.rstrip("/")}/{name}', group_keys)
        return

    # Keep the inexpensive grouped-shard traversal for ALF's usual layout.
    if isinstance(next(iter(group.values()), None), h5py.Dataset):
        yield path, group
        return
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            yield from _structure_groups(item, f'{path.rstrip("/")}/{name}')


def group_paths(file):
    """The structure groups of ``file``, in the order hippynn's ``AniDataLoader`` visits them."""
    import h5py
    with h5py.File(file, 'r') as handle:
        return [path for path, _ in _structure_groups(handle)]


def _row_mappings(dataset, parts, directory):
    """Recognize a column assembled from whole datasets placed in nonoverlapping row ranges."""
    import h5py

    if not dataset.shape or dataset.dtype.kind not in 'biufcS':
        return None
    rows = []
    for mapping in dataset.virtual_sources():
        path = os.path.normpath(os.path.join(directory, mapping.file_name))
        if path not in parts or mapping.vspace.get_select_type() != h5py.h5s.SEL_HYPERSLABS:
            return None
        start, end = mapping.vspace.get_select_bounds()
        stop = end[0] + 1
        if (any(start[1:]) or tuple(n + 1 for n in end[1:]) != dataset.shape[1:]
                or (stop - start[0]) * math.prod(dataset.shape[1:]) != mapping.vspace.get_select_npoints()):
            return None
        rows.append((start[0], stop, path, mapping))
    rows.sort(key=lambda row: row[0])
    if any(previous[1] > following[0] for previous, following in zip(rows, rows[1:])):
        return None
    return rows


def read_datasets(handle, group, keys=None):
    """Read selected group columns, opening each declared part once for all its columns."""
    import h5py
    import json
    import numpy as np
    from alframework.tools.label_hdf5 import LABEL_PARTS

    columns = {key: value for key, value in group.items()
               if isinstance(value, h5py.Dataset) and (keys is None or key in keys)}
    part_names = json.loads(handle.attrs.get(LABEL_PARTS, '[]'))
    if not part_names:
        return {key: dataset[()] for key, dataset in columns.items()}

    directory = os.path.dirname(os.path.abspath(handle.filename))
    parts = {os.path.normpath(os.path.join(directory, name)) for name in part_names}
    pending = defaultdict(list)
    arrays = {}
    fallback = set()
    for key, dataset in columns.items():
        rows = _row_mappings(dataset, parts, directory) if dataset.is_virtual else None
        if rows is None:
            fallback.add(key)
            continue
        array = arrays[key] = np.empty(dataset.shape, dtype=dataset.dtype)
        last = 0
        for start, stop, part, mapping in rows:
            if last < start:
                array[last:start] = dataset.fillvalue
            pending[part].append((key, start, stop, mapping))
            last = stop
        if last < len(array):
            array[last:] = dataset.fillvalue

    for part, copies in pending.items():
        with h5py.File(part, 'r') as source_file:
            for key, start, stop, mapping in copies:
                if key in fallback:
                    continue
                source = source_file[mapping.dset_name]
                source_space = mapping.src_space
                whole = (source_space.get_select_type() == h5py.h5s.SEL_ALL
                         or source_space.get_select_npoints() == source.size)
                destination = arrays[key]
                source_dtype = source.dtype
                if (not whole or source.shape != (stop - start, *destination.shape[1:])
                        or not np.can_cast(source_dtype, destination.dtype, casting='safe')):
                    fallback.add(key)
                elif source_dtype == destination.dtype and source_dtype.metadata == destination.dtype.metadata:
                    source.read_direct(destination[start:stop])
                else:
                    destination[start:stop] = source[()]
    return {key: dataset[()] if key in fallback else arrays[key] for key, dataset in columns.items()}


def _read_group(handle, group, path, keys):
    import numpy as np
    data = {'path': path}
    for key, raw in read_datasets(handle, group, keys).items():
        dataset = np.asarray(raw)
        if dataset.ndim and dataset.size and isinstance(dataset[0], (bytes, np.bytes_)):
            dataset = [item.decode('utf-8') for item in dataset]
        data[key] = dataset
    return data


def read_group(file, path, keys=None):
    """One group's datasets as ``AniDataLoader.populate_data_dict`` returns them: arrays, byte strings
    decoded to lists of strings, and the group's ``path``. Virtual datasets are read from their
    sources directly (see :func:`read_datasets`)."""
    import h5py
    with h5py.File(file, 'r') as handle:
        return _read_group(handle, handle[path], path, keys)


def stored_bytes(file, *, handle=None):
    """The bytes ``file`` holds, counting the label parts a snapshot references rather than the snapshot's own
    small file."""
    import json
    import h5py
    from alframework.tools.label_hdf5 import LABEL_PARTS
    size = os.path.getsize(file)
    try:
        with (h5py.File(file, 'r') if handle is None else nullcontext(handle)) as shard:
            parts = json.loads(shard.attrs.get(LABEL_PARTS, '[]'))
    except OSError:
        return size
    directory = os.path.dirname(os.path.abspath(file))
    return size + sum(os.path.getsize(os.path.join(directory, part)) for part in parts
                      if os.path.exists(os.path.join(directory, part)))


def _positive_or_none(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f'{name} must be a positive integer or null; received {value!r}.')
    return value


def read_workers_setting(value, name='h5_read_workers'):
    """A checked ``h5_read_workers`` setting: a positive integer, or ``None`` to decide by file size."""
    return _positive_or_none(value, name)


def read_budget_setting(value, name='h5_read_max_inflight_bytes'):
    """A checked ``h5_read_max_inflight_bytes`` setting: a positive number of bytes, or ``None`` for no limit."""
    return _positive_or_none(value, name)


def _worker_limit(file, requested, *, handle=None):
    setting = os.environ.get(READ_WORKERS, '')
    if requested is not None:
        return read_workers_setting(requested)
    if setting:
        return int(setting)
    if stored_bytes(file, handle=handle) < PARALLEL_MIN_BYTES:
        return 1
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1


def workers_for(file, groups, requested=None):
    """How many processes to read ``file``'s ``groups`` groups with; 1 or less means serially.

    ``requested`` (a configuration's ``h5_read_workers``) comes first, then ``ALF_H5_READ_WORKERS``; with
    neither, files of at least ``PARALLEL_MIN_BYTES`` use one process per available core."""
    return max(1, min(_worker_limit(file, requested), groups))


def _decoded_group_bytes(file, paths, keys, budget):
    import h5py

    with h5py.File(file, 'r') as handle:
        for path in paths:
            size = 0
            for key, column in handle[path].items():
                if isinstance(column, h5py.Dataset) and (keys is None or key in keys):
                    if column.dtype.hasobject:
                        # HDF5 metadata does not give the size of variable-length values.
                        size = budget + 1
                        break
                    size += column.nbytes
            yield path, size


def _read_in_order(pool, file, jobs, workers, keys, budget):
    pending = deque()
    item = next(jobs, None)
    reserved = 0

    def refill():
        nonlocal item, reserved
        while item is not None and len(pending) < workers:
            path, size = item
            if pending and budget is not None and reserved + size > budget:
                break
            pending.append((pool.submit(read_group, file, path, keys), size))
            reserved += size
            item = next(jobs, None)

    try:
        refill()
        while pending:
            future, size = pending.popleft()
            result = future.result()
            del future
            reserved -= size
            if budget is None:
                # Keep readers busy while the caller uses the group it just received.
                refill()
            yield result
            del result
            if budget is not None:
                refill()
    finally:
        for future, _ in pending:
            future.cancel()


def _use_array_pickle_protocol():
    from multiprocessing.reduction import ForkingPickler
    from pickle import DEFAULT_PROTOCOL

    if DEFAULT_PROTOCOL >= 5:
        return
    # This runs only in the reader children. ProcessPoolExecutor has no protocol option, and protocol 4
    # copies every NumPy array into a temporary bytes object before writing it to the result pipe.
    original_dumps = ForkingPickler.dumps

    def dumps(value, protocol=None):
        return original_dumps(value, 5 if protocol is None else protocol)

    ForkingPickler.dumps = staticmethod(dumps)


def read_groups(file, paths, workers, keys=None, *, max_inflight_bytes=None):
    """Read groups in order, using a process pool if more than one reader is available.

    The processes are spawned because the caller may already hold a GPU context. The optional byte
    limit estimates decoded columns queued or being read. Larger or variable-length groups run alone.
    """
    budget = read_budget_setting(max_inflight_bytes)
    if workers == 1:
        import h5py
        with h5py.File(file, 'r') as handle:
            for path in paths:
                yield _read_group(handle, handle[path], path, keys)
        return
    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context('spawn')
    if budget is None:
        jobs = ((path, 0) for path in paths)
    else:
        jobs = iter(list(_decoded_group_bytes(file, paths, keys, budget)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                                initializer=_use_array_pickle_protocol) as pool:
        yield from _read_in_order(pool, file, jobs, workers, keys, budget)


def read_shard(file, keys=None, workers=None, *, group_keys=None, sort_groups=False, max_inflight_bytes=None):
    """Every structure group of a stored label file, in order, as :func:`read_group` gives them, read in
    parallel when the file is large or ``workers`` asks for it (see :func:`workers_for`).

    Serial reads reuse one file handle and yield one group at a time. A snapshot is checked before
    yielding any groups, so a missing part raises ``FileNotFoundError``. Virtual datasets are read
    from their parts directly. ``group_keys`` only selects groups containing one of those datasets,
    including the root. ``sort_groups`` selects lexical path order for readers that require it.
    """
    import h5py
    from alframework.tools.label_hdf5 import validate_label_parts
    read_budget_setting(max_inflight_bytes)
    with h5py.File(file, 'r') as handle:
        validate_label_parts(handle)
        limit = _worker_limit(file, workers, handle=handle)
        groups = _structure_groups(handle, group_keys=group_keys)
        if sort_groups:
            groups = sorted(groups, key=lambda entry: entry[0])
        if limit <= 1:
            for path, group in groups:
                yield _read_group(handle, group, path, keys)
            return
        paths = [path for path, _ in groups]
        if len(paths) <= 1:
            for path in paths:
                yield _read_group(handle, handle[path], path, keys)
            return
    options = {} if max_inflight_bytes is None else {'max_inflight_bytes': max_inflight_bytes}
    yield from read_groups(file, paths, min(limit, len(paths)), keys, **options)
