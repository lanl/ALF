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
from collections import defaultdict
from contextlib import nullcontext
from itertools import repeat

READ_WORKERS = 'ALF_H5_READ_WORKERS'
PARALLEL_MIN_BYTES = 256 * 2**20


def _structure_groups(group, prefix=''):
    import h5py
    for key, item in group.items():
        path = f'{prefix}/{key}'
        first = next(iter(item.values()), None)
        if isinstance(first, h5py.Dataset):
            yield path, item
        else:
            yield from _structure_groups(item, path)


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
            dataset = [item.decode('ascii') for item in dataset]
        data[key] = dataset
    return data


def read_group(file, path, keys=None):
    """One group's datasets as ``AniDataLoader.populate_data_dict`` returns them: arrays, byte strings
    decoded to lists of ASCII strings, and the group's ``path``. Virtual datasets are read from their
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


def read_workers_setting(value, name='h5_read_workers'):
    """A checked ``h5_read_workers`` setting: a positive integer, or ``None`` to decide by file size."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f'{name} must be a positive integer or null; received {value!r}.')
    return value


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


def read_groups(file, paths, workers, keys=None):
    """Read groups in order, using a process pool if more than one reader is available.

    The processes are spawned, not forked: the caller may already hold a GPU context.
    """
    if workers == 1:
        import h5py
        with h5py.File(file, 'r') as handle:
            for path in paths:
                yield _read_group(handle, handle[path], path, keys)
        return
    import concurrent.futures
    import multiprocessing

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        yield from pool.map(read_group, repeat(file), paths, repeat(keys))


def read_shard(file, keys=None, workers=None):
    """Every structure group of a stored label file, in order, as :func:`read_group` gives them, read in
    parallel when the file is large or ``workers`` asks for it (see :func:`workers_for`).

    Serial reads reuse one file handle and yield one group at a time. A snapshot is checked before
    yielding any groups, so a missing part raises ``FileNotFoundError``. Virtual datasets are read
    from their parts directly.
    """
    import h5py
    from alframework.tools.label_hdf5 import validate_label_parts
    with h5py.File(file, 'r') as handle:
        validate_label_parts(handle)
        limit = _worker_limit(file, workers, handle=handle)
        if limit <= 1:
            for path, group in _structure_groups(handle):
                yield _read_group(handle, group, path, keys)
            return
        paths = [path for path, _ in _structure_groups(handle)]
        if len(paths) <= 1:
            for path in paths:
                yield _read_group(handle, handle[path], path, keys)
            return
    yield from read_groups(file, paths, min(limit, len(paths)), keys)
