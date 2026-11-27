"""Optional sampler-level thermal starts; dynamics engines preserve input state."""
import hashlib
import math
import os
from numbers import Real

import numpy as np
from ase import units

MODES = ('preserve', 'maxwell_boltzmann_if_missing')


def velocity_initialization_mode(value='preserve'):
    """Validate a policy without importing a model or a dynamics engine."""
    if not isinstance(value, str) or value not in MODES:
        raise ValueError(f'velocity_initialization must be one of {MODES}, not {value!r}.')
    return value


def validate_velocity_initialization(mode, temperature):
    """Check the sampler's velocity policy and optional Kelvin override."""
    mode = velocity_initialization_mode(mode)
    if temperature is None:
        return mode, None
    if mode == 'preserve':
        raise ValueError('initial_velocity_temperature_K needs velocity_initialization maxwell_boltzmann_if_missing.')
    error = 'initial_velocity_temperature_K must be a finite, nonnegative number of Kelvin.'
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError(error)
    try:
        temperature = float(temperature)
    except OverflowError as exc:
        raise ValueError(error) from exc
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(error)
    return mode, temperature


def velocity_initialization_settings(sampler_config):
    """Check a sampler configuration's velocity settings; return the policy and the optional temperature."""
    return validate_velocity_initialization(sampler_config.get('velocity_initialization', 'preserve'),
                                            sampler_config.get('initial_velocity_temperature_K'))


def kinetic_temperature(atoms):
    """ASE's instantaneous kinetic temperature, or None when it is undefined."""
    if atoms.get_number_of_degrees_of_freedom() <= 0:
        return None
    value = float(atoms.get_temperature())
    return value if math.isfinite(value) else None


def held_kinetic_temperature(atoms):
    """The kinetic temperature the samplers start from: the center of mass held at rest, as by ASE's ``FixCom``.

    The center-of-mass momentum is left out of the kinetic energy and its 3 degrees of freedom out of the count
    (3N - 3, less any other constraint's), as every engine removes it before the first step. ``None`` when no
    degree of freedom is left, such as for one atom.
    """
    from ase.constraints import FixCom
    if any(isinstance(constraint, FixCom) for constraint in atoms.constraints):
        return kinetic_temperature(atoms)
    dof = atoms.get_number_of_degrees_of_freedom() - 3
    if dof <= 0:
        return None
    momenta, masses = atoms.get_momenta(), atoms.get_masses()
    momenta = momenta - masses[:, None] * momenta.sum(axis=0) / masses.sum()
    value = float((momenta ** 2 / masses[:, None]).sum() / (dof * units.kB))
    return value if math.isfinite(value) else None


def initialize_velocities(atoms, temperature_K, mode='preserve', *, molecule_id, random_seed=None,
                          initial_velocity_temperature_K=None):
    """Draw missing momenta at the override or at the thermostat's initial target by default.

    An explicit momentum array, even all zeros, is always preserved. Independent
    Gaussian momentum components have variance m*kB*T, as in ASE's thermalizer,
    and the draw's center-of-mass momentum is removed, as the samplers hold the
    center of mass at rest: what remains is Maxwell-Boltzmann at T over the
    3N - 3 degrees of freedom they integrate, and ``initial_kinetic_temperature_K``
    counts those (:func:`held_kinetic_temperature`). No exact-temperature
    rescaling or angular-momentum removal is applied. This local NumPy draw avoids
    MPI broadcasts and works across ASE versions. Constraints are applied by
    Atoms.set_momenta as usual.

    A separate molecule-ID random stream leaves existing schedule and Langevin
    streams untouched. The preserve path consumes no randomness at all.
    """
    mode, override = validate_velocity_initialization(mode, initial_velocity_temperature_K)
    initialized = mode == 'maxwell_boltzmann_if_missing' and not atoms.has('momenta')
    seed = None
    temperature = None
    if initialized:
        temperature = float(temperature_K) if override is None else override
        masses = atoms.get_masses()
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Drawing initial velocities needs a finite, nonnegative temperature.')
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError('Drawing initial velocities needs finite, positive atomic masses.')
        material = (os.urandom(32) if random_seed is None else
                    repr(('alf.initial_velocities.v1', random_seed, str(molecule_id))).encode())
        seed = hashlib.sha256(material).hexdigest()
        rng = np.random.RandomState(np.frombuffer(bytes.fromhex(seed)[:16], dtype='<u4'))
        momenta = rng.standard_normal((len(atoms), 3)) * np.sqrt(masses[:, None] * units.kB * temperature)
        atoms.set_momenta(momenta - masses[:, None] * momenta.sum(axis=0) / masses.sum())
    return dict(velocity_initialization=mode, velocities_initialized=initialized,
                initial_velocity_seed=seed, initial_velocity_temperature_K=temperature,
                initial_kinetic_temperature_K=held_kinetic_temperature(atoms))
