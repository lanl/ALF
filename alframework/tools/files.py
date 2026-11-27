"""Write files so that a crash never leaves a half-written one, hash files once per worker, lock a run, and run
a block in a directory with its output copied to a log.

A reader sees either the old file or the new one: each write goes to a temporary name in the same
directory, is flushed to disk, and is then renamed over the final name.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def fsync_file(path):
    """Flush a closed file's contents to disk, so a rename can't expose an empty file after a node crash."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_atomically(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with open(temporary, 'wb') as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_json_atomic(path, values):
    """Replace ``path`` with ``values`` as indented JSON, all at once."""
    _replace_atomically(path, lambda handle: handle.write((json.dumps(values, indent=2) + '\n').encode()))


def write_bytes_atomic(path, data):
    """Replace ``path`` with ``data``, all at once."""
    _replace_atomically(path, lambda handle: handle.write(data))


_LOCKS = {}


def hold_lock(path):
    """Take an exclusive lock on ``path`` for the rest of this process, or raise if another process holds it.

    ``flock`` is released when the holder exits or is killed, so a crash never leaves a stale lock.
    The host and process recorded in the file are only there to name the holder in the error.
    """
    path = str(Path(path).resolve())
    if path in _LOCKS:
        return _LOCKS[path]
    handle = open(path, 'a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or 'an unknown process'
        handle.close()
        raise RuntimeError(f'Another ALF coordinator is using this run: {path} is locked by {holder}. '
                           'Two coordinators would overwrite each other\'s status and shards.') from None
    handle.seek(0)
    handle.truncate()
    handle.write(f'host {socket.gethostname()}, process {os.getpid()}, since {time.ctime()}\n')
    handle.flush()
    _LOCKS[path] = handle
    return handle


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


@contextlib.contextmanager
def working_directory(path, create=None):
    """Run the block in ``path``, then return to the previous working directory.

    ``create`` None makes ``path`` if it is missing; True requires it to be new; False requires it to exist.
    """
    if create not in (None, True, False):
        raise ValueError("create must be None, True or False.")
    exists = os.path.exists(path)
    if create is True and exists:
        raise FileExistsError(f"Directory '{path}' exists already.")
    if create is False and not exists:
        raise FileNotFoundError(f"Directory '{path}' not found.")
    if not exists:
        os.makedirs(path)
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _Tee:
    """A text stream writing to several streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged_output(path, mode='wt'):
    """Copy everything the block prints, to stdout and to stderr, into the log at ``path`` as well; an uncaught
    exception's traceback goes to the log too. Yields the open log file, which the block may also write to."""
    import sys
    import traceback
    with open(path, mode) as log:
        teed_stderr = _Tee(log, sys.stderr)
        with contextlib.redirect_stderr(teed_stderr), contextlib.redirect_stdout(_Tee(log, sys.stdout)):
            try:
                yield log
            except BaseException:
                print('Uncaught exception, logging interrupted.', file=teed_stderr)
                traceback.print_exc(file=log)
                raise
