"""Write files so that a crash never leaves a half-written one, and hash files once per worker.

A reader sees either the old file or the new one: each write goes to a temporary name in the same
directory, is flushed to disk, and is then renamed over the final name.
"""
import hashlib
import json
import os
from pathlib import Path


def fsync_file(path):
    """Flush a closed file's contents to disk, so a rename can't expose an empty file after a node crash."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path, values):
    """Replace ``path`` with ``values`` as indented JSON, all at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(values, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_key(path):
    """Identify one version of a file without reading it.

    An atomic replace gives a new inode, and an edit in place changes the size or modified time.
    """
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


_HASHES = {}


def cached_sha256(path):
    """Return a file's SHA-256, reading it only the first time this worker sees this version of it."""
    key = file_key(path)
    if _HASHES.get(key[0], (None,))[0] != key:
        digest = hashlib.sha256()
        with open(key[0], 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        # Only the latest version of each path is kept.
        _HASHES[key[0]] = (key, digest.hexdigest())
    return _HASHES[key[0]][1]
