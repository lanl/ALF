"""Read the structure groups of a large HDF5 label file in several processes.

A stored label file holds one group per composition, compressed with gzip. Reading it decompresses
each dataset in one thread, so a large shard reads slowly. Here each group is read by its own worker
process, which returns the group's datasets exactly as hippynn's ``AniDataLoader`` gives them, in the
file's group order.
The file is read in place; nothing is written.

``ALF_H5_READ_WORKERS`` sets the number of processes (1 reads serially, as before). By default files
of at least ``PARALLEL_MIN_BYTES`` use one process per available core, up to one per group. A label
snapshot counts the parts it references, and its virtual datasets are read from those parts directly.
"""
import concurrent.futures
import multiprocessing
import os

READ_WORKERS = 'ALF_H5_READ_WORKERS'
PARALLEL_MIN_BYTES = 256 * 2**20


def group_paths(file):
    """The structure groups of ``file``, in the order hippynn's ``AniDataLoader`` visits them."""
    import h5py
    paths = []

    def walk(group, prefix):
        for key, item in group.items():
            path = f'{prefix}/{key}'
            first = item[next(iter(item.keys()))]
            if isinstance(first, h5py.Dataset):
                paths.append(path)
            else:
                walk(item, path)

    with h5py.File(file, 'r') as handle:
        walk(handle, '')
    return paths


def read_virtual(dataset, directory, sources):
    """A virtual dataset's values, read from its source datasets directly.

    ALF's label snapshots join their parts' datasets end to end along the first axis. Reading each part's
    dataset and placing it where the snapshot maps it gives the same array as reading through HDF5's
    virtual layer, which is several times slower. ``sources`` caches the open source files. Any mapping
    that isn't a whole source dataset placed in a block of rows is read through HDF5 instead.
    """
    import h5py
    import numpy as np
    mappings = dataset.virtual_sources()
    out = np.empty(dataset.shape, dtype=dataset.dtype)
    covered = 0
    for mapping in mappings:
        name = os.path.join(directory, mapping.file_name) if mapping.file_name != '.' else None
        if name is None or mapping.vspace.get_select_type() != h5py.h5s.SEL_HYPERSLABS:
            return dataset[()]
        if name not in sources:
            sources[name] = h5py.File(name, 'r')
        source = sources[name][mapping.dset_name]
        start, end = mapping.vspace.get_select_bounds()
        rows = end[0] - start[0] + 1
        whole = mapping.src_space.get_select_type() == h5py.h5s.SEL_ALL or \
            mapping.src_space.get_select_npoints() == source.size
        if not whole or source.shape[0] != rows or source.shape[1:] != dataset.shape[1:] or \
                any(start[1:]) or rows * int(np.prod(dataset.shape[1:], dtype=np.int64)) != \
                mapping.vspace.get_select_npoints():
            return dataset[()]
        out[start[0]:end[0] + 1] = source[()]
        covered += rows
    # Rows no part supplies (none in ALF's snapshots) hold the dataset's fill value, as HDF5 gives them.
    return out if covered == dataset.shape[0] else dataset[()]


def read_group(file, path):
    """One group's datasets as ``AniDataLoader.populate_data_dict`` returns them: arrays, byte strings
    decoded to lists of ASCII strings, and the group's ``path``. Virtual datasets are read from their
    sources directly (see :func:`read_virtual`)."""
    import h5py
    import numpy as np
    data = {'path': path}
    sources = {}
    with h5py.File(file, 'r') as handle:
        try:
            for key, value in handle[path].items():
                if isinstance(value, h5py.Group):
                    continue
                raw = read_virtual(value, os.path.dirname(os.path.abspath(file)), sources) if value.is_virtual \
                    else value[()]
                dataset = np.asarray(raw)
                if isinstance(dataset, np.ndarray) and dataset.size != 0 and type(dataset[0]) is np.bytes_:
                    dataset = [item.decode('ascii') for item in dataset]
                data[key] = dataset
        finally:
            for source in sources.values():
                source.close()
    return data


def stored_bytes(file):
    """The bytes ``file`` holds, counting the label parts a snapshot references rather than the snapshot's own
    small file."""
    import json
    import h5py
    from alframework.tools.pyanitools import LABEL_PARTS
    size = os.path.getsize(file)
    try:
        with h5py.File(file, 'r') as handle:
            parts = json.loads(handle.attrs.get(LABEL_PARTS, '[]'))
    except OSError:
        return size
    directory = os.path.dirname(os.path.abspath(file))
    return size + sum(os.path.getsize(os.path.join(directory, part)) for part in parts
                      if os.path.exists(os.path.join(directory, part)))


def workers_for(file, groups):
    """How many processes to read ``file``'s ``groups`` groups with; 1 or less means serially."""
    setting = os.environ.get(READ_WORKERS, '')
    if setting:
        workers = int(setting)
    elif stored_bytes(file) < PARALLEL_MIN_BYTES:
        workers = 1
    else:
        workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return max(1, min(workers, groups))


def read_groups(file, paths, workers):
    """Each group of ``paths`` read by a pool of ``workers`` processes, yielded in order.

    The processes are spawned, not forked: the caller may already hold a GPU context.
    """
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        yield from pool.map(read_group, [file] * len(paths), paths)
