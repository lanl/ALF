"""Run the configured progress-plot utility in its own fresh process, outside the coordinator's memory.

The coordinator used to fork itself for each update. A forked child shares the coordinator's pages until
it writes to them, and Python's reference counting writes to most of them, so at scale each update copied
a large part of a coordinator holding many labels. Here each update is a new ``python -m`` process that
imports only the plotting utility and receives its arguments as JSON, and a new update is skipped while the
previous one is still running.
"""
import json
import os
import subprocess
import sys


class ProgressPlots:
    """Starts updates of ``utility`` (an import string such as ``alframework.tools.plotting.analysis_plot``)."""

    def __init__(self, utility):
        self.utility = utility
        self.process = None

    def update(self, arguments):
        """Start an update with ``arguments`` (JSON-serializable keyword arguments), unless one is still running.
        Returns whether one was started."""
        if self.process is not None:
            if self.process.poll() is None:
                print('Progress plots: the previous update is still running, so this one is skipped.', flush=True)
                return False
            self.process = None
        # The child imports what the coordinator can: its own import path, not only the environment's.
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
        self.process = subprocess.Popen([sys.executable, '-m', __name__, self.utility, json.dumps(arguments)],
                                        env=environment)
        return True

    def wait(self, timeout=None):
        """Wait for a running update, for example before the coordinator exits."""
        if self.process is not None:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass


def main(argv):
    from alframework.tools.config import load_module_from_string
    utility, arguments = argv
    load_module_from_string(utility)(**json.loads(arguments))


if __name__ == '__main__':
    main(sys.argv[1:])
