"""Construct a dynamics engine for the selected sampler backend."""

from alframework.samplers.engines.ase import ASEEngine


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
    if engine != 'ase':
        raise ValueError(f'Unsupported dynamics engine: {engine!r}.')
    return ASEEngine(atoms_list, models, dt, temperatures_K, rngs, calculator_options, friction=friction)
