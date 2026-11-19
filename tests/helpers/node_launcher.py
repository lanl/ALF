"""A stand-in for ``srun`` with one task per node, for multi-node member tests on one machine.

``python node_launcher.py N [--fail-node K] -- command...`` runs ``command`` N times at once, each with
``ALF_NODE_RANK`` set to its node number. With ``--fail-node K``, node K exits with code 3 at once instead.
When one fails, the others are stopped with SIGTERM, as ``srun --kill-on-bad-exit`` does, and this exits
with the first failure's code.
"""
import os
import signal
import subprocess
import sys
import time


def main(argv):
    split = argv.index('--')
    options, command = argv[:split], argv[split + 1:]
    nodes = int(options[0])
    fail = int(options[options.index('--fail-node') + 1]) if '--fail-node' in options else None
    processes = []
    for node in range(nodes):
        run = [sys.executable, '-c', 'raise SystemExit(3)'] if node == fail else command
        processes.append(subprocess.Popen(run, env=dict(os.environ, ALF_NODE_RANK=str(node))))
    while True:
        codes = [process.poll() for process in processes]
        failed = next((code for code in codes if code not in (None, 0)), None)
        if failed is not None or all(code == 0 for code in codes):
            break
        time.sleep(0.2)
    failed = failed or 0
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return failed


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
