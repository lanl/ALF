"""Construct a dynamics engine for the selected sampler backend.

``driver.engine`` names a built-in engine (``ase``, ``torchsim``, ``alchemi``) or, for an engine of your own, the
import path of its class, such as ``"my_package.engines.MyEngine"``. An engine class takes the arguments
:func:`make_engine` passes, ``(atoms_list, models, dt, temperatures_K, rngs, calculator_options, device=...,
friction=...)``, and provides what the sampler calls: ``step``, ``observe``, ``atoms``, ``finish``,
``set_temperature``, ``scale_cell``, ``attach_trajectory`` and the per-trajectory ``steps`` counts (see
:class:`~alframework.samplers.engines.packed.PackedMLMDEngine`, which a packed engine can subclass, and
``alframework.testing.check_engine``). Set ``accepts_batched_calculator = True`` on a class that evaluates the
ensemble a ``batched_calculator`` loads; otherwise the sampler gives it the ``ase_calculator`` ensemble. The class's
module is imported on the coordinator to read that attribute, so it should import GPU libraries inside the class.
"""

from alframework.samplers.engines.ase import ASEEngine


def engine_class(engine):
    """The engine class an import path names."""
    from alframework.tools.config import load_module_from_string
    try:
        found = load_module_from_string(engine)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f'driver.engine {engine!r} could not be imported: {exc}') from exc
    if not isinstance(found, type):
        raise ValueError(f'driver.engine {engine!r} names {type(found).__name__}, not an engine class.')
    return found


def make_engine(engine, atoms_list, models, dt, temperatures_K, rngs, calculator_options=None, device=None, friction=0.02):
    """Construct the engine; optional GPU libraries are imported only when selected."""
    if engine == 'torchsim':
        from alframework.samplers.engines.torchsim import TorchSimEngine
        return TorchSimEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                              device=device or 'cpu', friction=friction)
    if engine == 'alchemi':
        from alframework.samplers.engines.alchemi import AlchemiEngine
        return AlchemiEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                             device=device or 'cpu', friction=friction)
    if engine == 'ase':
        return ASEEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options, friction=friction)
    if isinstance(engine, str) and '.' in engine:
        return engine_class(engine)(atoms_list, models, dt, temperatures_K, rngs, calculator_options,
                                    device=device or 'cpu', friction=friction)
    raise ValueError(f'Unsupported dynamics engine: {engine!r}.')
