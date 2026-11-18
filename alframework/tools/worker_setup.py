"""Worker setup lines for running at scale: a node-local Python environment, caches, and HDF5 settings.

``worker_setup`` returns shell lines for a Parsl provider's ``worker_init``, and the interpreter the
workers should launch with. Everything is opt-in, so with the defaults it returns no lines.

- With ``env_tarball``, an archive of a relocatable Python environment on shared storage, each node
  unpacks it once into node-local storage, and workers start from that copy instead of importing
  every library from shared storage. ``conda-pack`` archives work, and so does a uv-managed
  standalone Python with its packages installed into it; ``env_python`` is the interpreter's path
  inside the archive. A ``bin/conda-unpack`` script, when present, runs after unpacking. The directory name includes a key from the archive's path, size, and modification
  time, so a rebuilt archive unpacks afresh. Blocks on the same node take turns under a lock, and a
  marker written last shows the unpack finished; an unfinished one is redone.
- With ``local_caches``, package caches that default to ``$HOME`` (CuPy kernels, Matplotlib, MIOpen,
  Numba, Triton) go to node-local storage, and HDF5 file locking is turned off. That is safe for ALF, whose
  shards are written once and renamed into place.

Only the standard library is used, since this runs where the coordinator builds its Parsl config.
"""
import hashlib
import os
import shlex
from pathlib import Path

DONE = '.alf-unpacked'
LOCK_WAIT_SECONDS = 1800


def _root(node_dir):
    return shlex.quote(str(node_dir)) if node_dir else '"${TMPDIR:-/tmp}"'


def environment_key(env_tarball):
    path = Path(env_tarball).expanduser().resolve(strict=True)
    stat = path.stat()
    return hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:12]


def worker_setup(env_tarball=None, node_dir=None, local_caches=False, cupy_cache=None, env_python='bin/python',
                 all_nodes=False):
    """Return ``(lines, python)``: ``worker_init`` lines, and a shell expression for the worker interpreter.

    ``python`` is ``None`` without ``env_tarball``, meaning the shared interpreter. ``node_dir`` is the
    node-local root; by default ``$TMPDIR`` on the node, or ``/tmp``. With an archive, the lines set
    ``ALF_ENV_BIN`` to the directory holding its interpreter, for ``PATH``.

    ``worker_init`` runs once per block, on its first node. For blocks of several nodes whose launcher
    starts workers on all of them (``SrunLauncher``), set ``all_nodes``: the setup then also runs on every
    node of the block, as one ``srun`` step, before the first node exports what the workers inherit.
    ``all_nodes`` may also be the command that runs one task per node, in place of ``ALL_NODES_COMMAND``.
    """
    lines, python = _setup_lines(env_tarball, node_dir, local_caches, cupy_cache, env_python)
    if all_nodes and lines:
        command = ALL_NODES_COMMAND if all_nodes is True else str(all_nodes)
        # The lines are idempotent (a finished unpack is reused, directories are created with -p), so the
        # first node runs them a second time in its own shell, to define and export the variables.
        # read -d '' returns nonzero at the end of its input, hence || true (worker_init may run under set -e).
        lines = ["IFS= read -r -d '' ALF_NODE_SCRIPT <<'ALF_NODE_SETUP' || true", *lines, 'ALF_NODE_SETUP',
                 f'{command} bash -c "$ALF_NODE_SCRIPT" || '
                 '{ echo "ALF: node-local setup failed on a node of this block" >&2; exit 1; }',
                 'eval "$ALF_NODE_SCRIPT"']
    return lines, python


# One task on each node of the current Slurm allocation, alongside any other step.
ALL_NODES_COMMAND = 'srun --nodes="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 --overlap'


def _setup_lines(env_tarball, node_dir, local_caches, cupy_cache, env_python):
    lines, python = [], None
    if env_tarball:
        tarball = shlex.quote(str(Path(env_tarball).expanduser().resolve(strict=True)))
        inside = Path(env_python)
        if inside.is_absolute() or '..' in inside.parts:
            raise ValueError(f'The environment interpreter must be a path inside the archive, not {env_python!r}')
        interpreter = shlex.quote(inside.as_posix())
        lines += [
            f'ALF_NODE_DIR={_root(node_dir)}/alf-"${{USER:-user}}"-{environment_key(env_tarball)}',
            'mkdir -p "$ALF_NODE_DIR" || { echo "ALF: cannot create $ALF_NODE_DIR" >&2; exit 1; }',
            # conda-unpack rewrites paths to where the environment is, so it runs in place, and the marker
            # written last is what says the environment is ready.
            'alf_unpack() {',
            f'  [ -e "$ALF_NODE_DIR/env/{DONE}" ] && return 0',
            '  rm -rf "$ALF_NODE_DIR/env" && mkdir "$ALF_NODE_DIR/env" &&',
            f'  tar -xf {tarball} -C "$ALF_NODE_DIR/env" &&',
            '  { [ ! -e "$ALF_NODE_DIR/env/bin/conda-unpack" ] || "$ALF_NODE_DIR/env/bin/conda-unpack"; } &&',
            f'  [ -x "$ALF_NODE_DIR/env/"{interpreter} ] && touch "$ALF_NODE_DIR/env/{DONE}"',
            '}',
            'alf_unpack_locked() {',
            '  if command -v flock >/dev/null 2>&1; then',
            '    ( flock 9 && alf_unpack ) 9>"$ALF_NODE_DIR.lock"',
            '  else',
            '    alf_waited=0',
            '    until mkdir "$ALF_NODE_DIR.lockdir" 2>/dev/null; do',
            f'      alf_waited=$((alf_waited + 1)); [ "$alf_waited" -gt {LOCK_WAIT_SECONDS} ] && return 1; sleep 1',
            '    done',
            '    alf_unpack; alf_status=$?; rmdir "$ALF_NODE_DIR.lockdir"; return $alf_status',
            '  fi',
            '}',
            f'alf_unpack_locked || {{ echo "ALF: could not unpack "{tarball}" into $ALF_NODE_DIR" >&2; exit 1; }}',
            f'ALF_ENV_BIN="$ALF_NODE_DIR/env/"{shlex.quote(inside.parent.as_posix())}',
        ]
        python = f'"$ALF_ENV_BIN/"{shlex.quote(inside.name)}'
    if local_caches:
        cache = '"$ALF_NODE_DIR/cache"' if env_tarball else f'{_root(node_dir)}/alf-"${{USER:-user}}"/cache'
        cupy = shlex.quote(str(cupy_cache)) if cupy_cache else '"$ALF_CACHE_DIR/cupy"'
        lines += [
            f'ALF_CACHE_DIR={cache}',
            'mkdir -p "$ALF_CACHE_DIR" || { echo "ALF: cannot create $ALF_CACHE_DIR" >&2; exit 1; }',
            'export XDG_CACHE_HOME="$ALF_CACHE_DIR" MPLCONFIGDIR="$ALF_CACHE_DIR/matplotlib"',
            'export NUMBA_CACHE_DIR="$ALF_CACHE_DIR/numba" TRITON_CACHE_DIR="$ALF_CACHE_DIR/triton"',
            f'export CUPY_CACHE_DIR={cupy}',
            # MIOpen (PyTorch on AMD GPUs) keeps its kernel database under $HOME by default.
            'export MIOPEN_USER_DB_PATH="$ALF_CACHE_DIR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$ALF_CACHE_DIR/miopen"',
            # Shards are written once and renamed into place, so readers need no locks.
            'export HDF5_USE_FILE_LOCKING=FALSE',
        ]
    return lines, python


def setup_from_environment(role_prefix, environ=os.environ, all_nodes=False):
    """Read the templates' variables: ``ALF_ENV_TARBALL`` (or ``<role_prefix>_ENV_TARBALL``),
    ``ALF_ENV_PYTHON``, ``ALF_NODE_LOCAL_DIR``, ``ALF_NODE_LOCAL_CACHES``, and ``ALF_CUPY_CACHE_DIR``.
    ``all_nodes`` is passed to ``worker_setup``, for multi-node blocks."""
    caches = environ.get('ALF_NODE_LOCAL_CACHES', '').strip().lower()
    if caches not in {'', '0', 'false', 'no', '1', 'true', 'yes'}:
        raise ValueError(f'ALF_NODE_LOCAL_CACHES must be 1 or 0, not {caches!r}')
    return worker_setup(environ.get(role_prefix + '_ENV_TARBALL') or environ.get('ALF_ENV_TARBALL') or None,
                        environ.get('ALF_NODE_LOCAL_DIR') or None, caches in {'1', 'true', 'yes'},
                        environ.get('ALF_CUPY_CACHE_DIR') or None, environ.get('ALF_ENV_PYTHON') or 'bin/python',
                        all_nodes=all_nodes)
