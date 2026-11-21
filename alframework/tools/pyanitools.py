"""Compatibility API for scripts that use ALF's ANI-style HDF5 utilities.

ALF writes label files with :class:`alframework.tools.label_hdf5.LabelH5Writer` (``datapacker`` here) and reads
them with :func:`alframework.tools.h5_parallel.read_shard`. ``anidataloader`` keeps the older reader's interface
and reads through the same code.
"""
import os
import platform

import h5py

from alframework.tools.h5_parallel import _read_group, _structure_groups
from alframework.tools.label_hdf5 import LABEL_PARTS, LabelH5Writer as datapacker, validate_label_parts
from alframework.tools.legacy_ani import build_ANI_info

# Determine python version
PY_VERSION = int(platform.python_version().split('.')[0]) > 3

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
    def h5py_dataset_iterator(self, g, prefix=''):
        for path, group in _structure_groups(g, prefix):
            yield _read_group(self.store, group, path, None)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        yield from self.h5py_dataset_iterator(self.store)

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        yield from self.h5py_dataset_iterator(g)

    ''' Returns the requested dataset '''
    def get_data(self, path, prefix=''):
        return _read_group(self.store, self.store[path], '{}/{}'.format(prefix, path), None)

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
