"""Chemical symbols and atomic numbers, converted for whole arrays at once."""

import numpy as np


def symbol_text(value):
    """A stored symbol as text: HDF5 gives bytes, ASE and NumPy give str."""
    return value.decode('utf-8') if isinstance(value, (bytes, np.bytes_)) else str(value)


def atomic_numbers_of(values):
    """Atomic numbers, as int64, for an array of symbols (bytes or text, any shape); atomic numbers pass through.

    Symbols stored as bytes of up to two characters are looked up through a table indexed by their bytes, so a
    million frames of species convert at array speed; other symbols convert once per distinct symbol. An unknown
    symbol raises ``KeyError``.
    """
    from ase.data import atomic_numbers
    array = np.asarray(values)
    if array.dtype.kind in 'iu':
        return array
    if array.dtype.kind == 'S' and array.dtype.itemsize <= 2:
        pairs = array.astype('S2').view(np.uint8).reshape(*array.shape, 2)
        codes = pairs[..., 0].astype(np.int32) << 8 | pairs[..., 1]
        table = np.zeros(1 << 16, dtype=np.int64)
        for code in np.flatnonzero(np.bincount(codes.ravel(), minlength=1 << 16)):
            table[code] = atomic_numbers[bytes([code >> 8, code & 255]).rstrip(b'\0').decode('utf-8')]
        return table[codes]
    if array.dtype.kind not in 'SUO':
        raise ValueError(f'Species must be chemical symbols or atomic numbers, not {array.dtype}.')
    distinct, inverse = np.unique(array.ravel(), return_inverse=True)
    numbers = np.array([atomic_numbers[symbol_text(symbol)] for symbol in distinct], dtype=np.int64)
    return numbers[inverse].reshape(array.shape)
