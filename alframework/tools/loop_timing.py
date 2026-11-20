"""How long each phase of the coordinator's loop takes, pass by pass, for finding what limits a large run.

Set ``coordinator_loop_path`` in the master configuration to turn it on; unset, timing adds nothing. Each pass of
the bootstrap and active-learning loops appends one CSV row, flushed as it is written, so a run that stops keeps
every finished pass. Columns, in order:

- ``pass`` (counting from 1 in each coordinator process), ``loop`` (``bootstrap`` or ``active``), ``start`` (Unix
  seconds) and ``seconds``, the pass's time without its sleep;
- the seconds spent in each phase of ``PHASES`` (0 for a phase the pass skipped), and the pass's ``sleep``;
- ``results_held``, the finished QM results the pass moved to held storage, and at the end of the pass, how many
  builder, sampler, QM and ML tasks or structures were in flight (``builder_queue`` to ``ml_queue``) and how many
  finished labels were held (``held_labels``).

A restarted coordinator appends to the same file, starting again from pass 1.
"""
import contextlib
import csv
import os
import time

PHASES = ('reload_config', 'builders', 'builder_harvest', 'sampler_submit', 'sampler_harvest', 'qm_submit', 'hold',
          'store', 'ml', 'plots', 'print_status', 'write_status', 'executor_check', 'retire_frames')
COUNTS = ('results_held', 'builder_queue', 'sampler_queue', 'qm_queue', 'ml_queue', 'held_labels')
COLUMNS = ('pass', 'loop', 'start', 'seconds', *PHASES, 'sleep', *COUNTS)
LOOPS = ('bootstrap', 'active')


class LoopTimer:
    """Times the phases of each coordinator pass; does nothing without a ``path``."""

    def __init__(self, path=None, *, clock=time.perf_counter, wall_clock=time.time):
        self.enabled = bool(path)
        self.path = path
        self._clock, self._wall_clock = clock, wall_clock
        self._file = self._writer = None
        self._pass = 0
        self._reset()

    def _reset(self):
        self._phases = dict.fromkeys(PHASES, 0.0)
        self._counts = dict.fromkeys(COUNTS, 0)
        self._started = self._wall_started = None

    def _begin(self):
        if self._started is None:
            self._started, self._wall_started = self._clock(), self._wall_clock()

    def phase(self, name):
        """A context that adds its time to phase ``name`` of this pass."""
        if not self.enabled:
            return contextlib.nullcontext()
        if name not in self._phases:
            raise ValueError(f'Unknown coordinator loop phase {name!r}; phases are {PHASES}.')
        return self._timed(name)

    @contextlib.contextmanager
    def _timed(self, name):
        self._begin()
        started = self._clock()
        try:
            yield
        finally:
            self._phases[name] += self._clock() - started

    def count(self, **values):
        """Record end-of-pass counts, such as ``results_held`` (added up within a pass) or queue sizes."""
        if not self.enabled:
            return
        for name, value in values.items():
            if name not in self._counts:
                raise ValueError(f'Unknown coordinator loop count {name!r}; counts are {COUNTS}.')
            self._counts[name] = self._counts[name] + value if name == 'results_held' else value

    def end_pass(self, loop, sleep):
        """Write this pass's row; ``sleep`` is the delay before the next pass."""
        if not self.enabled:
            return
        if loop not in LOOPS:
            raise ValueError(f'Unknown coordinator loop {loop!r}; loops are {LOOPS}.')
        self._begin()
        self._pass += 1
        row = {'pass': self._pass, 'loop': loop, 'start': round(self._wall_started, 3),
               'seconds': round(self._clock() - self._started, 6),
               **{name: round(value, 6) for name, value in self._phases.items()}, 'sleep': round(float(sleep), 3),
               **self._counts}
        try:
            self._open().writerow(row)
            self._file.flush()
        except OSError as exc:
            print(f'Could not write the coordinator loop timing {self.path}: {exc}; stopping it.', flush=True)
            self.enabled = False
        self._reset()

    def _open(self):
        if self._writer is None:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            self._file = open(self.path, 'a', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
            if new:
                self._writer.writeheader()
        return self._writer

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = self._writer = None


def coordinator_loop_timer(master_config):
    """The master configuration's loop timer: on with ``coordinator_loop_path``, off without it."""
    path = master_config.get('coordinator_loop_path')
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError('coordinator_loop_path must name a file.')
    return LoopTimer(path)
