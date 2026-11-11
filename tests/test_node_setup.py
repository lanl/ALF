"""Worker setup for running at scale: unpacking an environment archive on each node, and node-local caches.

The shell lines run for real in bash, on a tiny fake archive, so these check what the lines do, including
two blocks starting on one node at once.
"""

import os
import shutil
import subprocess
import tarfile

import pytest

from alframework.tools.node_setup import DONE, setup_from_environment, worker_setup

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _archive(tmp_path, name="env.tar.gz", python="bin/python", payload="one"):
    """A fake packed environment: an executable interpreter stub and a conda-unpack that counts its runs."""
    source = tmp_path / f"source-{name}"
    (source / os.path.dirname(python)).mkdir(parents=True, exist_ok=True)
    (source / python).write_text("#!/bin/sh\necho " + payload + "\n")
    unpack = source / "bin" / "conda-unpack"
    unpack.parent.mkdir(exist_ok=True)
    # Outside env/, so a redone unpack can't erase the count.
    unpack.write_text('#!/bin/sh\necho ran >> "$(dirname "$0")/../../unpack-count"\n')
    for path in (source / python, unpack):
        path.chmod(0o755)
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=".")
    return archive


def _run(lines, python, node, *, background=1):
    script = "\n".join(lines + [f"{python}"]) + "\n"
    env = dict(os.environ, TMPDIR=str(node), USER="alf")
    runs = [subprocess.Popen(["bash", "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(background)]
    return [(run.wait(timeout=120), *run.communicate()) for run in runs]


def test_defaults_add_nothing():
    assert worker_setup() == ([], None)
    assert setup_from_environment("ALF_QM", environ={}) == ([], None)


def test_local_caches_alone_set_caches_and_hdf5_without_unpacking(tmp_path):
    lines, python = worker_setup(local_caches=True)
    assert python is None and not any("tar -xf" in line for line in lines)
    script = "\n".join(lines) + '\necho "$CUPY_CACHE_DIR|$MPLCONFIGDIR|$HDF5_USE_FILE_LOCKING"\n'
    out = subprocess.run(["bash", "-c", script], env=dict(os.environ, TMPDIR=str(tmp_path), USER="alf"),
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == f"{tmp_path}/alf-alf/cache/cupy|{tmp_path}/alf-alf/cache/matplotlib|FALSE"


def test_two_blocks_on_one_node_unpack_once(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    lines, python = worker_setup(_archive(tmp_path))
    results = _run(lines, python, node, background=2)
    assert all(code == 0 and out.strip() == "one" for code, out, _ in results), results
    [directory] = node.glob("alf-alf-*")
    assert (directory / "env" / DONE).exists()
    assert (directory / "unpack-count").read_text().count("ran") == 1, "one block unpacks, the other waits"
    _run(lines, python, node)
    assert (directory / "unpack-count").read_text().count("ran") == 1, "a later block reuses the copy"


def test_a_rebuilt_archive_unpacks_into_a_new_directory(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    archive = _archive(tmp_path)
    first, python = worker_setup(archive)
    _run(first, python, node)
    archive.unlink()
    second, python = worker_setup(_archive(tmp_path, payload="rebuilt-with-more"))
    [(code, out, _)] = _run(second, python, node)
    assert code == 0 and out.strip() == "rebuilt-with-more"
    assert len(list(node.glob("alf-alf-*"))) == 2


def test_an_interpreter_elsewhere_in_the_archive(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    lines, python = worker_setup(_archive(tmp_path, python="cpython-3.12/bin/python3"), env_python="cpython-3.12/bin/python3")
    [(code, out, _)] = _run(lines + ['echo "$ALF_ENV_BIN"'], python, node)
    assert code == 0 and out.split() == [f"{next(node.glob('alf-alf-*'))}/env/cpython-3.12/bin", "one"]
    with pytest.raises(ValueError, match="inside the archive"):
        worker_setup(_archive(tmp_path, name="other.tar.gz"), env_python="/usr/bin/python3")


def test_an_archive_missing_on_the_node_stops_the_block(tmp_path):
    node = tmp_path / "node"
    node.mkdir()
    archive = _archive(tmp_path)
    lines, python = worker_setup(archive)
    archive.unlink()
    [(code, _, err)] = _run(lines, python, node)
    assert code == 1 and "could not unpack" in err


def test_templates_launch_workers_from_the_unpacked_copy_but_not_the_interchange(tmp_path, monkeypatch):
    pytest.importorskip("parsl")
    import importlib.util
    from pathlib import Path

    archive = _archive(tmp_path)
    monkeypatch.setattr("parsl.addresses.address_by_hostname", lambda: "127.0.0.1")
    monkeypatch.setenv("ALF_ENV_TARBALL", str(archive))
    monkeypatch.setenv("ALF_RUN_ROOT", str(tmp_path))
    for example in ("CPP_excited_gpu4pyscf", "AcAc_excited_pyseqm"):
        path = Path(__file__).resolve().parents[1] / "examples" / example / "parsl_configs.py"
        spec = importlib.util.spec_from_file_location(f"parsl_configs_{example}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for executor in module.config.executors:
            assert executor.launch_cmd.startswith('"$ALF_ENV_BIN/"python'), executor.label
            assert "ALF_ENV_BIN" not in " ".join(executor.interchange_launch_cmd), executor.label
            assert 'export PATH="$ALF_ENV_BIN":' in executor.provider.worker_init
