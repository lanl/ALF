"""ALCHEMI 0.2 deterministic updates with ALF's per-trajectory BAOAB noise.

The native velocity-finalization kernel supplies each half kick. Its unit-mass
form updates momentum directly (p += dt * F / 2), preserving ASE momenta and
avoiding a round trip through velocity. Drifts and the shared stochastic step
complete BAOAB. The stock ALCHEMI Langevin RNG is deliberately not used: its
packed atom indices would change a trajectory's noise when batches change.
"""
from importlib.metadata import PackageNotFoundError, version

from alframework.samplers.packed_engine import PackedMLMDEngine


def require_alchemi():
    """Import only the qualified toolkit version, when this engine is selected."""
    try:
        if version('nvalchemi-toolkit') != '0.2.0':
            raise ImportError('The ALCHEMI engine requires nvalchemi-toolkit==0.2.0.')
        import torch
        from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize
    except (ImportError, PackageNotFoundError) as exc:
        raise ImportError("driver.engine 'alchemi' requires ALF's 'alchemi' extra "
                          "(nvalchemi-toolkit==0.2.0, Python 3.11+) in the sampler worker.") from exc
    return torch, vv_velocity_finalize


class AlchemiEngine(PackedMLMDEngine):
    """Packed CPU/NVIDIA ground-state MLMD with the existing engine contract."""
    engine_name = 'ALCHEMI'

    def __init__(self, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, friction=0.02,
                 device='cpu'):
        torch, self._velocity_update = require_alchemi()
        selected = torch.device(device)
        if selected.type not in {'cpu', 'cuda'} or (selected.type == 'cuda' and torch.version.hip is not None):
            raise NotImplementedError('ALCHEMI supports CPU or NVIDIA CUDA; use TorchSim for ROCm.')
        if selected.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('The requested ALCHEMI CUDA device is unavailable.')
        super().__init__(atoms_list, models, dt, temperatures_K, rngs, calculator_options, friction, device)

    def _layout(self):
        super()._layout()
        if self.state is not None:
            self._unit_masses = self._torch.ones_like(self.state.masses)
            self._batch_idx = self.state.system_idx.to(self._torch.int32)
            # These low-level kernels take internal time. The base already
            # converted fs to ASE time; the high-level ALCHEMI fs conversion
            # must not be applied a second time.
            self._time_steps = self.state.energy.new_full((len(self._rows),), self._dt)

    def _kick(self):
        self._velocity_update(self.state.momenta, self.state.forces, self._unit_masses,
                              self._time_steps, self._batch_idx)

    def _drift(self):
        self.state.positions += (self._dt / 2) * self.state.momenta / self.state.masses[:, None]

    def _baoab(self):
        self._kick()
        self._drift()
        self._ou()
        self._drift()
        bad = self._evaluate()
        self._kick()
        return bad
