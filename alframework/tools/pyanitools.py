"""Read and write ALF's HDF5 training shards, in the ANI data format.

Every shard ALF stores is written with ``datapacker`` and read with ``anidataloader``: one HDF5 group per
empirical formula, holding that formula's frames as arrays. The module keeps the name of the ANI tools it
comes from, which scripts and training code import; it is not legacy code. The legacy ANI model helper is
:mod:`alframework.tools.legacy_ani`.
"""
import h5py
import json
import numpy as np
import platform
import os
from pathlib import Path


LABEL_PARTS = 'alf_label_parts'


def validate_label_parts(handle):
    """Check a virtual label shard before HDF5 can treat a missing source as fill values."""
    parts = json.loads(handle.attrs.get(LABEL_PARTS, '[]'))
    directory = Path(handle.filename).parent
    for relative_path in parts:
        source = directory / relative_path
        if not source.is_file():
            raise FileNotFoundError(f'{handle.filename}: required label part {source} is missing')

# Preserve the previous import path.
from alframework.tools.legacy_ani import build_ANI_info

# Determine python version
PY_VERSION = int(platform.python_version().split('.')[0]) > 3

'''                ANI data packer class
    Class for storing data supplied as a dictionary.
'''
class datapacker(object):
    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6, shuffle=False, frame_chunk=None,
                 **file_options):
        """Wrapper to store arrays within HFD5 file

        ``complib`` and ``complevel`` choose the lossless compression ('gzip' with a level, 'lzf', or None).
        ``shuffle`` adds HDF5's byte-shuffle filter to numeric arrays, and ``frame_chunk`` stores arrays in
        chunks of that many whole frames instead of h5py's guess. Neither changes any stored value.
        """
        # opening file
        self.store = h5py.File(store_file, mode=mode, **file_options)
        self.clib = complib
        self.clev = None if complib in (None, 'lzf') else complevel
        self.shuffle = shuffle
        self.frame_chunk = frame_chunk

    def store_data(self, store_loc, **kwargs):
        """Put arrays to store
        """
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            if type(v) == list:
                if len(v) != 0:
                    if type(v[0]) is np.str_ or type(v[0]) is str:
                        v = [a.encode('utf8') for a in v]

            #print(k)
            options = dict(compression=self.clib, compression_opts=self.clev)
            array = np.asarray(v)
            if self.shuffle and array.dtype.kind in 'biuf':
                options['shuffle'] = True
            if self.frame_chunk and array.ndim >= 1 and array.size and (self.clib or self.shuffle):
                options['chunks'] = (min(int(self.frame_chunk), array.shape[0]),) + array.shape[1:]
            g.create_dataset(k, data=v, **options)

    def cleanup(self):
        """Wrapper to close HDF5 file
        """
        self.store.close()


'''                 ANI data loader class
    Class for loading data stored with the datapacker class.
'''
class anidataloader(object):

    ''' Contructor '''
    def __init__(self, store_file):
        if not os.path.exists(store_file):
            raise FileNotFoundError('file ' + store_file + 'not found.')
        self.store = h5py.File(store_file,'r')
        try:
            validate_label_parts(self.store)
        except BaseException:
            self.store.close()
            raise

    ''' Group recursive iterator (iterate through all groups in all branches and return datasets in dicts) '''
    def h5py_dataset_iterator(self,g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset): # test for dataset
                data = {'path':path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])

                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if isinstance(dataset[0], (bytes, np.bytes_)):
                                    dataset = [a.decode('ascii') for a in dataset]

                        data.update({k:dataset})

                yield data
            else: # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        for data in self.h5py_dataset_iterator(g):
            yield data

    ''' Returns the requested dataset '''
    def get_data(self, path, prefix=''):
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k][()])

                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if isinstance(dataset[0], (bytes, np.bytes_)):
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    ''' Returns the number of groups '''
    def group_size(self):
        return len(self.get_group_list())

    ''' Returns the number of items in the entire file '''
    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()
