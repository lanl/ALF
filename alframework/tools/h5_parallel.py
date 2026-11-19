"""Read the structure groups of a large HDF5 label file in several processes.

A stored label file holds one group per composition, compressed with gzip. Reading it decompresses
each dataset in one thread, so a large shard reads slowly. Here each group is read by its own worker
process, which returns the group's datasets exactly as hippynn's ``AniDataLoader`` gives them, in the
file's group order.
The file is read in place; nothing is written.

``ALF_H5_READ_WORKERS`` sets the number of processes (1 reads serially, as before). By default files
of at least ``PARALLEL_MIN_BYTES`` use one process per available core, up to one per group.
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


def read_group(file, path):
    """One group's datasets as ``AniDataLoader.populate_data_dict`` returns them: arrays, byte strings
    decoded to lists of ASCII strings, and the group's ``path``."""
    import h5py
    import numpy as np
    data = {'path': path}
    with h5py.File(file, 'r') as handle:
        for key, value in handle[path].items():
            if isinstance(value, h5py.Group):
                continue
            dataset = np.asarray(value[()])
            if isinstance(dataset, np.ndarray) and dataset.size != 0 and type(dataset[0]) is np.bytes_:
                dataset = [item.decode('ascii') for item in dataset]
            data[key] = dataset
    return data


def workers_for(file, groups):
    """How many processes to read ``file``'s ``groups`` groups with; 1 or less means serially."""
    setting = os.environ.get(READ_WORKERS, '')
    if setting:
        workers = int(setting)
    elif os.path.getsize(file) < PARALLEL_MIN_BYTES:
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
