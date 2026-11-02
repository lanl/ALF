"""TorchSim engine for batched multi-trajectory MLMD (``driver.engine: torchsim``).

:class:`TorchSimEngine` implements the engine contract of
:class:`alframework.samplers.batched_mlmd.ASEEngine`, so
``run_mlmd_trajectories`` drives it unchanged: the same checks, schedules,
stopping, acquisition, and cohorts. All active trajectories are packed into one
TorchSim ``MDState`` and every MD step makes one batched ensemble evaluation.

Integration is Langevin BAOAB (Leimkuhler and Matthews) built from TorchSim's
``momentum_step`` and ``position_step``. The stochastic step draws each
trajectory's noise from that trajectory's own random stream (seeded from its
molecule ID, as in the ASE engine). With exact inference a trajectory's result
therefore does not depend on which other trajectories share its task.
Finished trajectories are removed from the packed state rather than frozen.
As in ASE's ``Langevin`` (``fixcm=True``) the noise carries no net momentum,
and the friction is the legacy MLMD value of 0.02 inverse ASE time units.

Units are those of ASE: Angstrom, eV, amu, and ASE time; ``dt`` is in fs.

Forces used for propagation are the ensemble mean plus the optional well
potential and UDD bias (``MLMD_calculator_options``), with the formulas of
``MLMD_calculator``. Screening reads the unbiased per-member predictions.

Inference comes from a batched evaluator (``evaluate(numbers, positions,
cells, pbc, counts) -> (energies [models, systems], forces [models, atoms,
3])``, e.g. ``hippynn_interface.load_hippynn_batched_ensemble``), selected
by the sampler's ``batched_calculator``. Without one, the ``ase_calculator``
ensemble is evaluated one structure at a time.
"""

from alframework.samplers.packed_engine import PackedMLMDEngine, ASECalculatorBatch, _calculator_options


def require_torchsim():
    """Import TorchSim only when its engine is selected."""
    try:
        from torch_sim.integrators.md import MDState, momentum_step, position_step
    except ImportError as exc:
        raise ImportError("driver.engine 'torchsim' requires torch and torch-sim-atomistic (Python 3.12+) in the "
                          "sampler worker environment; install ALF's 'torchsim' extra.") from exc
    return MDState, momentum_step, position_step


class TorchSimEngine(PackedMLMDEngine):
    """TorchSim deterministic updates and ALF's per-trajectory thermostat."""
    engine_name = "TorchSim"

    def __init__(self, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, friction=0.02,
                 device="cpu"):
        self.state_class, self._momentum_step, self._position_step = require_torchsim()
        super().__init__(atoms_list, models, dt, temperatures_K, rngs, calculator_options, friction, device)

    def _baoab(self):
        half = self._dt / 2
        self._momentum_step(self.state, half)
        self._position_step(self.state, half)
        self._ou()
        self._position_step(self.state, half)
        bad = self._evaluate()
        self._momentum_step(self.state, half)
        return bad
