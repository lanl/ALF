"""Write HDF5 label groups and check the backing files of virtual label shards."""

import json
from pathlib import Path

import h5py
import numpy as np


LABEL_PARTS = 'alf_label_parts'


def validate_label_parts(handle):
    """Check a virtual label shard before HDF5 can treat a missing source as fill values."""
    parts = json.loads(handle.attrs.get(LABEL_PARTS, '[]'))
    directory = Path(handle.filename).parent
    for relative_path in parts:
        source = directory / relative_path
        if not source.is_file():
            raise FileNotFoundError(f'{handle.filename}: required label part {source} is missing')


class LabelH5Writer:
    """Write one group of label arrays per composition.

    ``complib`` and ``complevel`` select lossless compression. ``shuffle`` adds the byte-shuffle filter
    to numeric arrays, and ``frame_chunk`` groups that many whole frames into each chunk.
    """

    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6, shuffle=False, frame_chunk=None,
                 **file_options):
        self.store = h5py.File(store_file, mode=mode, **file_options)
        self.clib = complib
        self.clev = None if complib in (None, 'lzf') else complevel
        self.shuffle = shuffle
        self.frame_chunk = frame_chunk

    def store_data(self, store_loc, **kwargs):
        """Create a group and write its datasets."""
        group = self.store.create_group(store_loc)
        for name, values in kwargs.items():
            if type(values) is list and values and type(values[0]) in (str, np.str_):
                values = [value.encode('utf8') for value in values]

            options = dict(compression=self.clib, compression_opts=self.clev)
            array = np.asarray(values)
            if self.shuffle and array.dtype.kind in 'biuf':
                options['shuffle'] = True
            if self.frame_chunk and array.ndim >= 1 and array.size and (self.clib or self.shuffle):
                options['chunks'] = (min(int(self.frame_chunk), array.shape[0]),) + array.shape[1:]
            group.create_dataset(name, data=values, **options)

    def cleanup(self):
        """Close the HDF5 file."""
        self.store.close()
