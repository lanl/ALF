"""GPU4PySCF labeling for flattened state energies and forces (``E#``/``F#``).

``nroots: 0`` is a ground-state calculation: the SCF energy and gradient, labeling ``E0``/``F0`` only.
Store them as ``"E0": ["energy", ...], "F0": ["forces", ...]`` for ground-state trainers and fine-tuning.

The public task in this module labels one nonperiodic ``MoleculesObject`` at a
time on ALF's ordinary QM executor.  It intentionally calls PySCF/GPU4PySCF
directly: the upstream ASE adapter exposes ordinary ground-state properties,
but not ALF's flattened ``E#``/``F#`` multi-state contract.

Optional ``compute_nacr`` additionally labels excited-excited nonadiabatic
coupling vectors so a dataset need not be relabeled for later dynamics work.
NACR is stored but never learned: its property key matches none of the
``E#``/``F#``/``dE#`` patterns that the excited-state trainer derives its heads
from, so the machine-learned potential still targets energies and forces only.
"""

from __future__ import annotations

import importlib
import itertools
import math
import numbers
import os
import time
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from parsl import python_app
from alframework.tools.task_timeline import timed

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
)
from alframework.tools.molecules_class import MoleculesObject


DEFAULT_GPU4PYSCF_CONFIG = {
    "xc": "cam-b3lyp",
    "basis": "6-31g*",
    "charge": 0,
    "multiplicity": 1,
    "density_fit": True,
    "auxbasis": None,
    "grids_level": 3,
    # Include motion/weight response of the DFT quadrature in S0 gradients.
    # Opt-in to retain the force convention of existing excited-state datasets.
    "ground_state_grid_response": False,
    "nroots": 5,
    "scf_conv_tol": 1.0e-10,
    "scf_max_cycle": 100,
    "tda_conv_tol": 1.0e-8,
    "tda_lindep": 1.0e-20,
    "cphf_conv_tol": 1.0e-8,
    "tda_max_cycle": 200,
    "num_threads": 8,
    "max_memory_mb": None,
    "verbosity": 0,
    "energy_offset_eV": 0.0,
    "compute_nacr": False,
    "solvent": None,
}

# ALF stores nonadiabatic coupling vectors under one flattened pair_atomic
# property. The key deliberately matches none of the E#/F#/dE# training
# patterns, so the excited-state trainer never builds a head for it.
NACR_PROPERTY_KEY = "nacr"
NAC_SCOPE = "excited_excited"
NACR_UNITS = "angstrom^-1"

DCM_PCM_CONFIG = {
    "enabled": True, "name": "dichloromethane", "method": "C-PCM",
    "static_dielectric": 9.02, "optical_dielectric": 2.03,
    "lebedev_order": 29, "vdw_scale": 1.2,
    "surface_discretization_method": "SWIG", "pcm_conv_tol": 1e-7,
}
VERTICAL_PROPERTIES = {
    "vertical_excitation_energies", "vertical_transition_dipoles", "vertical_oscillator_strengths",
}


def _validate_solvent(config):
    if config is None:
        return None
    if not isinstance(config, dict) or not config:
        raise ValueError('solvent must be a nonempty PCM configuration dictionary')
    if not isinstance(config.get('enabled', True), bool):
        raise ValueError('solvent.enabled must be Boolean')
    unknown = set(config) - set(DCM_PCM_CONFIG)
    if unknown:
        raise ValueError(f'Unknown PCM solvent settings: {sorted(unknown)}')
    if not config.get('enabled', True):
        return None
    options = {**DCM_PCM_CONFIG, **config}
    if options['method'] != 'C-PCM' or options['surface_discretization_method'] != 'SWIG':
        raise ValueError('This interface supports C-PCM with a SWIG surface')
    if not isinstance(options['name'], str) or not options['name'].strip():
        raise ValueError('solvent.name must be a nonempty descriptive string')
    for key in ('static_dielectric', 'optical_dielectric', 'vdw_scale', 'pcm_conv_tol'):
        options[key] = _positive_float(options[key], name='solvent.'+key)
    if not options['static_dielectric'] >= options['optical_dielectric'] >= 1.:
        raise ValueError('Require static_dielectric >= optical_dielectric >= 1')
    options['lebedev_order'] = _integer(options['lebedev_order'], name='solvent.lebedev_order', minimum=1)
    # PySCF's supported Lebedev orders, not arbitrary sphere point counts.
    if options['lebedev_order'] not in {3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,35,41,47,53,59,65,71,77,83,89,95,101,107,113,119,125,131}:
        raise ValueError('Unsupported solvent.lebedev_order')
    return options


def excited_excited_nac_pairs(nroots: int) -> tuple[tuple[int, int], ...]:
    """Return the ordered excited-excited NAC pairs for ``nroots`` roots."""

    return tuple(itertools.combinations(range(1, int(nroots) + 1), 2))


class GPU4PySCFError(RuntimeError):
    """Base class carrying the failed calculation stage and states."""

    stage = "calculation"

    def __init__(
        self,
        message: str,
        *,
        failed_states: list[int] | tuple[int, ...] = (),
        stage: str | None = None,
    ) -> None:
        self.failed_states = tuple(int(state) for state in failed_states)
        if stage is not None:
            self.stage = str(stage)
        super().__init__(message)


class GPU4PySCFDependencyError(GPU4PySCFError):
    """Raised when the CUDA-specific PySCF stack is unavailable."""

    stage = "dependency"


class GPU4PySCFDeviceError(GPU4PySCFError):
    """Raised when the QM worker cannot select a CUDA device."""

    stage = "device"


class GPU4PySCFSCFError(GPU4PySCFError):
    """Raised when the ground-state RKS calculation fails."""

    stage = "scf"


class GPU4PySCFTDAError(GPU4PySCFError):
    """Raised when one or more TDA roots fail."""

    stage = "tda"


class GPU4PySCFNACError(GPU4PySCFError):
    """Raised when the nonadiabatic coupling solver fails or misreports pairs."""

    stage = "nac"


class GPU4PySCFGradientError(GPU4PySCFError):
    """Raised when a ground- or excited-state gradient fails."""

    stage = "gradient"


class GPU4PySCFOutputError(GPU4PySCFError):
    """Raised when converged backend output is incomplete or malformed."""

    stage = "output"


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite positive number; received {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be finite; received {value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer; received {value!r}.")
    parsed = int(value)
    if minimum is not None and parsed < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}; received {parsed}.")
    return parsed


def validate_gpu4pyscf_config(QM_config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a validated GPU4PySCF configuration with public defaults."""

    if QM_config is not None and not isinstance(QM_config, dict):
        raise TypeError("QM_config must be a dictionary.")
    options = dict(DEFAULT_GPU4PYSCF_CONFIG)
    options.update(dict(QM_config or {}))
    options['solvent'] = _validate_solvent(options.get('solvent'))

    for key in ("xc", "basis"):
        value = options.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string.")
        options[key] = value.strip()

    options["charge"] = _integer(options.get("charge"), name="charge")
    options["multiplicity"] = _integer(
        options.get("multiplicity"),
        name="multiplicity",
        minimum=1,
    )
    if options["multiplicity"] != 1:
        raise ValueError(
            "gpu4pyscf_excited_state_task currently supports only singlet "
            "RKS/TDA calculations (multiplicity=1)."
        )

    if not isinstance(options.get("density_fit"), bool):
        raise ValueError("density_fit must be a Boolean.")
    auxbasis = options.get("auxbasis")
    if auxbasis is not None:
        if not isinstance(auxbasis, str) or not auxbasis.strip():
            raise ValueError("auxbasis must be null or a nonempty string.")
        if not options["density_fit"]:
            raise ValueError("auxbasis requires density_fit=true.")
        options["auxbasis"] = auxbasis.strip()

    options["grids_level"] = _integer(
        options.get("grids_level"), name="grids_level", minimum=0
    )
    if not isinstance(options.get("ground_state_grid_response"), bool):
        raise ValueError("ground_state_grid_response must be a Boolean.")
    # nroots 0 is a ground-state calculation: SCF energy and gradient, with no TDA.
    options["nroots"] = _integer(
        options.get("nroots"), name="nroots", minimum=0
    )
    options["scf_conv_tol"] = _positive_float(
        options.get("scf_conv_tol"), name="scf_conv_tol"
    )
    options["tda_conv_tol"] = _positive_float(
        options.get("tda_conv_tol"), name="tda_conv_tol"
    )
    options["tda_lindep"] = _positive_float(options["tda_lindep"], name="tda_lindep")
    options["cphf_conv_tol"] = _positive_float(options["cphf_conv_tol"], name="cphf_conv_tol")
    for key in ("scf_max_cycle", "tda_max_cycle", "num_threads"):
        options[key] = _integer(options.get(key), name=key, minimum=1)
    options["verbosity"] = _integer(
        options.get("verbosity"), name="verbosity", minimum=0
    )

    memory = options.get("max_memory_mb")
    options["max_memory_mb"] = (
        None
        if memory is None
        else _positive_float(memory, name="max_memory_mb")
    )
    options["energy_offset_eV"] = _finite_float(
        options.get("energy_offset_eV"), name="energy_offset_eV"
    )

    if not isinstance(options.get("compute_nacr"), bool):
        raise ValueError("compute_nacr must be a Boolean.")
    if options["compute_nacr"] and options["nroots"] < 2:
        raise ValueError(
            "compute_nacr requires nroots >= 2; excited-excited coupling needs "
            f"at least two roots but nroots={options['nroots']} yields no pairs."
        )
    return options


def validate_gpu4pyscf_properties(
    properties_list: dict[str, list[Any]],
    *,
    nroots: int,
    compute_nacr: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate ALF's complete energy/force contract for ``nroots``.

    When ``compute_nacr`` is set, ``properties_list`` must also declare the
    ``nacr`` pair_atomic property, and vice versa. Requiring both together
    prevents a configuration that computes coupling with nowhere to store it, or
    that declares storage for data that is never produced.
    """

    state_table = derive_state_property_table(
        properties_list, require_forces=True
    )
    gap_table = derive_gap_property_table(properties_list)
    expected_states = int(nroots) + 1
    if len(state_table) != expected_states:
        raise ValueError(
            "GPU4PySCF property count must equal nroots + 1: "
            f"nroots={nroots} requires states 0 through {nroots}, but "
            f"properties_list defines {len(state_table)} states."
        )

    if int(nroots) == 0:
        excited_only = sorted(
            {str(row["gap_key"]) for row in gap_table}
            | ({"transition_dipoles", "oscillator_strengths", NACR_PROPERTY_KEY} | VERTICAL_PROPERTIES)
            .intersection(properties_list)
        )
        if excited_only:
            raise ValueError(
                "nroots=0 is a ground-state calculation, so it can't give "
                f"excited-state properties: {excited_only}."
            )

    supported_keys: set[str] = set()
    for row in state_table:
        energy_key = str(row["energy_key"])
        force_key = str(row["force_key"])
        energy_schema = properties_list[energy_key]
        force_schema = properties_list[force_key]
        if len(energy_schema) < 2 or str(energy_schema[1]).lower() != "system":
            raise ValueError(
                f"Energy property {energy_key!r} must be a system property."
            )
        if len(force_schema) < 2 or str(force_schema[1]).lower() != "atomic":
            raise ValueError(
                f"Force property {force_key!r} must be an atomic property."
            )
        if not str(energy_schema[0]) or not str(force_schema[0]):
            raise ValueError(
                f"Properties {energy_key!r}/{force_key!r} require nonempty "
                "database names."
            )
        supported_keys.update((energy_key, force_key))
    supported_keys.update(str(row["gap_key"]) for row in gap_table)

    nacr_declared = NACR_PROPERTY_KEY in properties_list
    if bool(compute_nacr) and not nacr_declared:
        raise ValueError(
            f"compute_nacr requires properties_list to declare the "
            f"{NACR_PROPERTY_KEY!r} pair_atomic property so coupling vectors "
            "can be stored."
        )
    if nacr_declared and not bool(compute_nacr):
        raise ValueError(
            f"properties_list declares {NACR_PROPERTY_KEY!r} but "
            "QM_config.compute_nacr is false, so no coupling vectors would be "
            "calculated."
        )
    if nacr_declared:
        nacr_schema = properties_list[NACR_PROPERTY_KEY]
        if len(nacr_schema) < 2 or str(nacr_schema[1]).lower() != "pair_atomic":
            raise ValueError(
                f"Property {NACR_PROPERTY_KEY!r} must use the 'pair_atomic' "
                "scope."
            )
        if not str(nacr_schema[0]):
            raise ValueError(
                f"Property {NACR_PROPERTY_KEY!r} requires a nonempty database "
                "name."
            )
        supported_keys.add(NACR_PROPERTY_KEY)

    optical_keys = {"transition_dipoles", "oscillator_strengths", "absolute_state_energies"} | VERTICAL_PROPERTIES
    for key in optical_keys.intersection(properties_list):
        schema = properties_list[key]
        if len(schema) != 3 or str(schema[1]).lower() != "system" or not str(schema[0]):
            raise ValueError(f"Optical property {key!r} must declare a system dataset and scale.")
        _positive_float(schema[2], name=key+'.scale')
        if key == 'vertical_excitation_energies' and float(schema[2]) != float(properties_list[state_table[0]['energy_key']][2]):
            raise ValueError('Vertical excitation energies and state energies must use consistent scales')
    supported_keys.update(optical_keys)
    unsupported = sorted(set(properties_list) - supported_keys)
    if unsupported:
        raise ValueError(
            "The GPU4PySCF interface supports only flattened E#/F#/dE# "
            f"properties and {NACR_PROPERTY_KEY!r}; unsupported keys: "
            f"{unsupported}."
        )
    return state_table, gap_table


def _energy_offset(
    QM_config: dict[str, Any],
    sampler_config: dict[str, Any],
) -> tuple[float, str]:
    if "energy_offset_eV" in QM_config:
        raw_offset = QM_config["energy_offset_eV"]
        source = "QM_config"
    elif "energy_offset_eV" in sampler_config:
        raw_offset = sampler_config["energy_offset_eV"]
        source = "sampler_config"
    else:
        raw_offset = 0.0
        source = "default"
    offset = _finite_float(raw_offset, name="energy_offset_eV")
    return offset, source


def _import_cupy():
    try:
        return importlib.import_module("cupy")
    except Exception as exc:
        raise GPU4PySCFDependencyError(
            "GPU4PySCF excited-state labeling requires a working CuPy "
            "installation matching the worker's CUDA runtime."
        ) from exc


def select_gpu4pyscf_device(
    gpus_per_node: int | None = None,
) -> dict[str, Any]:
    """Confirm that this QM worker sees exactly one CUDA device, and activate it.

    GPU4PySCF's single-device kernels only work on local device 0, so rank-based
    selection among several visible GPUs can't work. A worker that sees several is
    rejected whatever its rank, so a misconfigured node fails the same way on every
    worker. ``gpus_per_node`` is unused; it remains so that existing calls still work.
    """

    cupy = _import_cupy()
    try:
        visible_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise GPU4PySCFDeviceError(
            "CuPy could not enumerate CUDA devices on the QM worker."
        ) from exc
    if visible_count < 1:
        raise GPU4PySCFDeviceError(
            "gpu4pyscf_excited_state_task requires at least one visible CUDA "
            "device; CPU fallback is intentionally disabled."
        )

    try:
        worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    except ValueError as exc:
        raise GPU4PySCFDeviceError(str(exc)) from exc
    visible_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_count > 1:
        raise GPU4PySCFDeviceError(
            f"GPU4PySCF requires each QM worker to see exactly one CUDA device, but "
            f"worker rank {worker_rank} sees {visible_count} "
            f"(CUDA_VISIBLE_DEVICES={visible_mask}). Bind each QM worker to one "
            "allocated GPU before CUDA imports, for example with Parsl's "
            "available_accelerators."
        )
    device_index = 0
    try:
        cupy.cuda.Device(device_index).use()
    except Exception as exc:
        raise GPU4PySCFDeviceError(
            f"CuPy could not activate CUDA device {device_index}."
        ) from exc
    return {
        "index": device_index,
        "label": f"cuda:{device_index}",
        "visible_count": visible_count,
        "worker_rank": worker_rank,
        "cuda_visible_devices": visible_mask,
    }


def _to_numpy(value: Any, cupy_module: Any | None = None) -> np.ndarray:
    """Explicitly move a possible CuPy value to host memory."""

    if cupy_module is not None:
        array_type = getattr(cupy_module, "ndarray", None)
        if array_type is not None and isinstance(value, array_type):
            return np.asarray(cupy_module.asnumpy(value))
        if type(value).__module__.split(".", 1)[0] == "cupy":
            return np.asarray(cupy_module.asnumpy(value))
    if hasattr(value, "get") and callable(value.get):
        return np.asarray(value.get())
    return np.asarray(value)


def _require_scf_convergence(mf: Any, cupy_module: Any | None = None) -> None:
    raw_flag = getattr(mf, "converged", None)
    if raw_flag is None:
        raise GPU4PySCFSCFError(
            "GPU4PySCF did not expose mf.converged after the RKS solve.",
            failed_states=(0,),
        )
    flag = _to_numpy(raw_flag, cupy_module)
    if flag.shape != () or not np.issubdtype(flag.dtype, np.bool_):
        raise GPU4PySCFSCFError(
            "GPU4PySCF mf.converged must be one Boolean scalar; "
            f"received shape {flag.shape} and dtype {flag.dtype}.",
            failed_states=(0,),
        )
    if not bool(flag):
        raise GPU4PySCFSCFError(
            "GPU4PySCF RKS did not converge.", failed_states=(0,)
        )


def _require_tda_convergence(
    td: Any,
    nroots: int,
    cupy_module: Any | None = None,
) -> None:
    root_states = tuple(range(1, int(nroots) + 1))
    raw_flags = getattr(td, "converged", None)
    if raw_flags is None:
        raise GPU4PySCFTDAError(
            "GPU4PySCF did not expose one TDA convergence flag per root.",
            failed_states=root_states,
        )
    flags = _to_numpy(raw_flags, cupy_module)
    if flags.shape != (int(nroots),) or not np.issubdtype(
        flags.dtype, np.bool_
    ):
        raise GPU4PySCFTDAError(
            "GPU4PySCF td.converged must contain one Boolean per root with "
            f"shape {(int(nroots),)}; received shape {flags.shape} and "
            f"dtype {flags.dtype}.",
            failed_states=root_states,
        )
    failed_states = (np.flatnonzero(~flags) + 1).astype(int).tolist()
    if failed_states:
        raise GPU4PySCFTDAError(
            "GPU4PySCF TDA did not converge for excited states "
            f"{failed_states}.",
            failed_states=failed_states,
        )


def convert_gpu4pyscf_atomic_units(
    energies_hartree: Any,
    gradients_hartree_per_bohr: Any,
    *,
    state_count: int,
    atom_count: int,
    hartree_to_ev: float,
    bohr_to_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw state output and convert it to ALF energy/force units."""

    energies = np.asarray(energies_hartree, dtype=np.float64)
    gradients = np.asarray(
        gradients_hartree_per_bohr, dtype=np.float64
    )
    expected_energies = (int(state_count),)
    expected_gradients = (int(state_count), int(atom_count), 3)
    if energies.shape != expected_energies:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed energies: expected "
            f"{expected_energies}, received {energies.shape}."
        )
    if gradients.shape != expected_gradients:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed gradients: expected "
            f"{expected_gradients}, received {gradients.shape}."
        )
    if not np.all(np.isfinite(energies)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite state energies."
        )
    if not np.all(np.isfinite(gradients)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite state gradients."
        )
    energy_factor = _positive_float(
        hartree_to_ev, name="Hartree-to-eV conversion"
    )
    length_factor = _positive_float(
        bohr_to_angstrom, name="Bohr-to-Angstrom conversion"
    )
    return energies * energy_factor, -gradients * (
        energy_factor / length_factor
    )


def _validated_calculated_outputs(
    energies: Any,
    forces: Any,
    *,
    state_count: int,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    energy_array = np.asarray(energies, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    expected_energies = (int(state_count),)
    expected_forces = (int(state_count), int(atom_count), 3)
    if energy_array.shape != expected_energies:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed converted energies: expected "
            f"{expected_energies}, received {energy_array.shape}."
        )
    if force_array.shape != expected_forces:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed converted forces: expected "
            f"{expected_forces}, received {force_array.shape}."
        )
    if not np.all(np.isfinite(energy_array)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite converted energies."
        )
    if not np.all(np.isfinite(force_array)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite converted forces."
        )
    return energy_array, force_array


def _validated_nac_outputs(
    nacr: Any,
    nac_pairs: Any,
    *,
    nroots: int,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate coupling pair ordering, shape, and finiteness.

    Failures raise so the molecule is marked unconverged and dropped, rather
    than writing malformed coupling into a shard that is expensive to relabel.
    """

    nacr_array = np.asarray(nacr, dtype=np.float64)
    pair_array = np.asarray(nac_pairs, dtype=np.int64)
    expected_pairs = excited_excited_nac_pairs(nroots)
    expected_pair_shape = (len(expected_pairs), 2)
    expected_nacr_shape = (len(expected_pairs), int(atom_count), 3)
    if pair_array.shape != expected_pair_shape:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned a malformed NAC pair index: expected "
            f"{expected_pair_shape}, received {pair_array.shape}."
        )
    if [tuple(int(v) for v in pair) for pair in pair_array] != list(
        expected_pairs
    ):
        raise GPU4PySCFOutputError(
            "GPU4PySCF NAC pair ordering does not match the expected "
            f"excited-excited combinations {list(expected_pairs)}; received "
            f"{[tuple(int(v) for v in pair) for pair in pair_array]}."
        )
    if nacr_array.shape != expected_nacr_shape:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed coupling vectors: expected "
            f"{expected_nacr_shape}, received {nacr_array.shape}."
        )
    if not np.all(np.isfinite(nacr_array)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite coupling vectors."
        )
    return nacr_array, pair_array




def _validated_optical_outputs(dipoles, strengths, *, nroots):
    dipoles = np.asarray(dipoles, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if dipoles.shape != (nroots, 3) or strengths.shape != (nroots,):
        raise GPU4PySCFOutputError("Malformed transition dipoles/oscillator strengths.")
    if not np.all(np.isfinite(dipoles)) or not np.all(np.isfinite(strengths)):
        raise GPU4PySCFOutputError("Nonfinite transition dipoles/oscillator strengths.")
    if np.any(strengths < 0):
        raise GPU4PySCFOutputError("Negative oscillator strength.")
    return dipoles, strengths


def transition_dipoles_staged(td, cupy):
    """Two-stage AO->occupied->virtual contraction used by the CPP benchmark.

    Avoid the large intermediate in a single three-operand contraction.
    Length-gauge moments are in atomic units; retain the solver's phase.
    """
    from pyscf.tdscf.rhf import _charge_center
    mol = td.mol
    with mol.with_common_orig(_charge_center(mol)):
        ao = cupy.asarray(mol.intor_symmetric("int1e_r", comp=3))
    coefficients = cupy.asarray(td._scf.mo_coeff)
    occupations = cupy.asarray(td._scf.mo_occ)
    occupied = coefficients[:, occupations == 2]
    virtual = coefficients[:, occupations == 0]
    temporary = cupy.einsum("xpq,pi->xiq", ao, occupied.conj(), optimize=True)
    ov = cupy.einsum("xiq,qj->xij", temporary, virtual, optimize=True)
    del temporary, ao
    xs = cupy.asarray([x for x, _ in td.xy])
    moments = cupy.einsum("xij,nij->nx", ov, xs, optimize=True) * 2.0
    if isinstance(td.xy[0][1], (np.ndarray, cupy.ndarray)):
        ys = cupy.asarray([y for _, y in td.xy])
        moments += cupy.einsum("xij,nij->nx", ov, ys, optimize=True) * 2.0
    return _to_numpy(moments, cupy)


def _single_device_backend(pyscf):
    """Use upstream's PySCF configuration hook before GPU4PySCF imports.

    GPU4PySCF captures num_devices in many modules at first import. With one
    device upstream kernels index device-local buffers at zero and sometimes
    explicitly enter Device(0). Each QM worker therefore needs one allocated
    GPU exposed as local device zero by its executor. Never rewrite visibility
    masks here or monkeypatch imported module constants.
    """
    import sys
    cached = sys.modules.get("gpu4pyscf.__config__")
    if cached is not None and cached.num_devices != 1:
        raise GPU4PySCFDeviceError(
            "GPU4PySCF was imported in multi-GPU mode; set PySCF num_devices=1 "
            "before its first import in this single-device QM worker.")
    config = getattr(pyscf, "__config__", None)
    missing = object()
    previous = getattr(config, "num_devices", missing)
    if config is not None:
        config.num_devices = 1
    try:
        return (importlib.import_module("gpu4pyscf"),
                importlib.import_module("gpu4pyscf.dft"))
    finally:
        if config is not None:
            if previous is missing:
                del config.num_devices
            else:
                config.num_devices = previous


def _run_tda(mf, options, nroots, root_states, cupy):
    """Run TDA for ``nroots`` excited states, returning the solver and its excitation energies (Hartree)."""
    try:
        td = mf.TDA(equilibrium_solvation=True) if options.get("solvent") else mf.TDA()
        td.nstates = nroots
        td.conv_tol = float(options["tda_conv_tol"])
        td.max_cycle = int(options["tda_max_cycle"])
        td.lindep = float(options["tda_lindep"])
        td.kernel()
    except GPU4PySCFError:
        raise
    except Exception as exc:
        raise GPU4PySCFTDAError(
            f"GPU4PySCF TDA failed: {type(exc).__name__}: {exc}",
            failed_states=root_states,
        ) from exc
    _require_tda_convergence(td, nroots, cupy)
    try:
        excitation_energies = np.asarray(
            _to_numpy(getattr(td, "e", None), cupy),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned TDA roots that could not be converted to "
            "floating-point values.",
            failed_states=root_states,
        ) from exc
    if excitation_energies.shape != (nroots,):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed TDA roots: expected "
            f"{(nroots,)}, received {excitation_energies.shape}.",
            failed_states=root_states,
        )
    return td, excitation_energies


def _run_states_on_cpu(atoms, options):
    """The molecular RKS/TDA path of ``run_gpu4pyscf_states``, with PySCF on the CPU."""
    unsupported = [key for key in ("compute_nacr", "solvent", "_compute_optical", "_compute_vertical")
                   if options.get(key)]
    if unsupported:
        raise GPU4PySCFDependencyError(
            f"{unsupported} need the GPU backend; the CPU backend gives state energies, forces, and gaps.")
    pyscf = importlib.import_module("pyscf")
    dft = importlib.import_module("pyscf.dft")
    nist = importlib.import_module("pyscf.data.nist")
    scf_start = time.time()
    try:
        pyscf.lib.num_threads(int(options["num_threads"]))
        molecule_kwargs = {
            "atom": [(chemical_symbols[int(number)], tuple(float(value) for value in position))
                     for number, position in zip(atoms.get_atomic_numbers(), atoms.get_positions())],
            "basis": options["basis"], "charge": int(options["charge"]), "spin": 0,
            "unit": "Angstrom", "verbose": int(options["verbosity"]),
        }
        if options["max_memory_mb"] is not None:
            molecule_kwargs["max_memory"] = float(options["max_memory_mb"])
        mf = dft.RKS(pyscf.M(**molecule_kwargs), xc=options["xc"])
        if options["density_fit"]:
            mf = mf.density_fit() if options["auxbasis"] is None else mf.density_fit(auxbasis=options["auxbasis"])
        mf.grids.level = int(options["grids_level"])
        mf.conv_tol = float(options["scf_conv_tol"])
        mf.max_cycle = int(options["scf_max_cycle"])
        mf.kernel()
    except GPU4PySCFError:
        raise
    except Exception as exc:
        raise GPU4PySCFSCFError(f"PySCF RKS failed: {type(exc).__name__}: {exc}", failed_states=(0,)) from exc
    _require_scf_convergence(mf)
    scf_seconds = time.time() - scf_start

    nroots = int(options["nroots"])
    root_states = tuple(range(1, nroots + 1))
    tda_start = time.time()
    td, excitation = (None, np.empty(0)) if nroots == 0 else _run_tda(mf, options, nroots, root_states, None)
    tda_seconds = time.time() - tda_start if nroots else 0.0
    gradient_start = time.time()
    gradients = np.empty((nroots + 1, len(atoms), 3))
    try:
        ground_gradient = mf.nuc_grad_method()
        ground_gradient.grid_response = options.get("ground_state_grid_response", False)
        gradients[0] = np.asarray(ground_gradient.kernel(), dtype=np.float64)
    except Exception as exc:
        raise GPU4PySCFGradientError(f"PySCF ground-state gradient failed: {type(exc).__name__}: {exc}",
                                     failed_states=(0,)) from exc
    if nroots:
        try:
            excited_gradient = td.nuc_grad_method()
            excited_gradient.cphf_conv_tol = float(options["cphf_conv_tol"])
        except Exception as exc:
            raise GPU4PySCFGradientError(f"PySCF could not construct the TDA gradient method: "
                                         f"{type(exc).__name__}: {exc}", failed_states=root_states) from exc
        for state in root_states:
            try:
                gradients[state] = np.asarray(excited_gradient.kernel(state=state), dtype=np.float64)
            except Exception as exc:
                raise GPU4PySCFGradientError(f"PySCF gradient failed for state S{state}: {type(exc).__name__}: {exc}",
                                             failed_states=(state,)) from exc
    energies_hartree = float(mf.e_tot) + np.concatenate([[0.0], np.asarray(excitation, dtype=np.float64)])
    energies_ev, forces = convert_gpu4pyscf_atomic_units(
        energies_hartree, gradients, state_count=nroots + 1, atom_count=len(atoms),
        hartree_to_ev=float(nist.HARTREE2EV), bohr_to_angstrom=float(nist.BOHR))
    versions = {
        "ground_state_grid_response": bool(options.get("ground_state_grid_response", False)),
        "qm_solvent": None, "qm_state_response": "gas", "qm_optical_response": "none",
        "qm_vertical_response": "none", "qm_vertical_root_order": "none", "qm_vertical_seconds": 0.0,
        "qm_nac_solver": "none", "_transition_dipoles": np.empty((0, 3)), "_oscillator_strengths": np.empty((0,)),
        "qm_scf_seconds": scf_seconds, "qm_tda_seconds": tda_seconds, "qm_optical_seconds": 0.0,
        "qm_gradient_seconds": time.time() - gradient_start, "gpu4pyscf_num_devices": 0,
        "pyscf_version": str(getattr(pyscf, "__version__", "unknown")), "gpu4pyscf_version": None,
        "cupy_version": None, "s1_gradient_source": "tda_gradient" if nroots else "none",
    }
    return energies_ev, forces, np.empty((0, len(atoms), 3)), np.empty((0, 2), dtype=np.int64), versions


def run_gpu4pyscf_states(
    atoms,
    options: dict[str, Any],
    *,
    device_index: int,
    backend: str = "gpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one GPU4PySCF RKS/TDA energy-and-gradient calculation.

    Returns state energies (eV), forces (eV/Angstrom), NACR (Angstrom^-1, empty
    unless ``compute_nacr``), the NACR pair index, and backend metadata.
    ``backend="cpu"`` runs the same calculation with PySCF on the CPU, for
    ``pyscf_task``; it supports energies, forces and gaps only.
    """

    if backend == "cpu":
        return _run_states_on_cpu(atoms, options)
    if int(device_index) != 0:
        raise GPU4PySCFDeviceError(
            "GPU4PySCF single-device kernels require local CUDA device 0. "
            "Bind each QM worker to one allocated GPU before CUDA imports; "
            "selecting a nonzero CuPy device does not isolate this backend.")
    try:
        pyscf = importlib.import_module("pyscf")
        gpu4pyscf, gpu_dft = _single_device_backend(pyscf)
        cupy = importlib.import_module("cupy")
        nist = importlib.import_module("pyscf.data.nist")
    except Exception as exc:
        raise GPU4PySCFDependencyError(
            "GPU4PySCF excited-state labeling requires compatible PySCF, "
            "GPU4PySCF, and CuPy installations."
        ) from exc

    scf_start = time.time()
    try:
        cupy.cuda.Device(int(device_index)).use()
        pyscf.lib.num_threads(int(options["num_threads"]))
        atom_spec = [
            (
                chemical_symbols[int(number)],
                tuple(float(value) for value in position),
            )
            for number, position in zip(
                atoms.get_atomic_numbers(), atoms.get_positions()
            )
        ]
        molecule_kwargs = {
            "atom": atom_spec,
            "basis": options["basis"],
            "charge": int(options["charge"]),
            "spin": 0,
            "unit": "Angstrom",
            "verbose": int(options["verbosity"]),
        }
        if options["max_memory_mb"] is not None:
            molecule_kwargs["max_memory"] = float(options["max_memory_mb"])
        mol = pyscf.M(**molecule_kwargs)
        mf = gpu_dft.RKS(mol, xc=options["xc"])
        if options["density_fit"]:
            if options["auxbasis"] is None:
                mf = mf.density_fit()
            else:
                mf = mf.density_fit(auxbasis=options["auxbasis"])
        to_gpu = getattr(mf, "to_gpu", None)
        if callable(to_gpu):
            mf = to_gpu()
        mf.grids.level = int(options["grids_level"])
        mf.conv_tol = float(options["scf_conv_tol"])
        mf.max_cycle = int(options["scf_max_cycle"])
        if options.get('solvent'):
            mf = mf.PCM()
            solvent = options['solvent']
            pcm = mf.with_solvent
            pcm.method = solvent['method']
            pcm.eps = solvent['static_dielectric']
            pcm.eps_optical = solvent['optical_dielectric']
            pcm.lebedev_order = solvent['lebedev_order']
            pcm.vdw_scale = solvent['vdw_scale']
            pcm.surface_discretization_method = solvent['surface_discretization_method']
            pcm.conv_tol = solvent['pcm_conv_tol']
        mf.kernel()
    except GPU4PySCFError:
        raise
    except Exception as exc:
        raise GPU4PySCFSCFError(
            f"GPU4PySCF RKS failed: {type(exc).__name__}: {exc}",
            failed_states=(0,),
        ) from exc
    _require_scf_convergence(mf, cupy)
    scf_seconds = time.time() - scf_start
    tda_start = time.time()

    nroots = int(options["nroots"])
    root_states = tuple(range(1, nroots + 1))
    td = None
    excitation_energies = np.empty(0, dtype=np.float64)
    if nroots > 0:
        td, excitation_energies = _run_tda(mf, options, nroots, root_states, cupy)
    tda_seconds = time.time() - tda_start if nroots > 0 else 0.0

    try:
        ground_energy = np.asarray(
            _to_numpy(getattr(mf, "e_tot", None), cupy),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPU4PySCFOutputError(
            "GPU4PySCF mf.e_tot could not be converted to a floating-point "
            "value.",
            failed_states=(0,),
        ) from exc
    if ground_energy.shape != ():
        raise GPU4PySCFOutputError(
            "GPU4PySCF mf.e_tot must be one scalar; received "
            f"shape {ground_energy.shape}.",
            failed_states=(0,),
        )
    energies_hartree = np.empty(nroots + 1, dtype=np.float64)
    energies_hartree[0] = float(ground_energy)
    energies_hartree[1:] = (
        float(ground_energy)
        + np.asarray(excitation_energies, dtype=np.float64)
    )

    optical_start = time.time()
    transition_dipoles = np.empty((0, 3))
    oscillator_strengths = np.empty((0,))
    if options.get("_compute_optical", False):
        try:
            transition_dipoles = transition_dipoles_staged(td, cupy)
            oscillator_strengths = (2.0 / 3.0) * excitation_energies * np.einsum(
                "si,si->s", transition_dipoles, transition_dipoles
            )
            transition_dipoles, oscillator_strengths = _validated_optical_outputs(
                transition_dipoles, oscillator_strengths, nroots=nroots
            )
        except Exception as exc:
            error = GPU4PySCFOutputError(f"Optical properties failed: {type(exc).__name__}: {exc}")
            error.stage = "optical"
            raise error from exc
    optical_seconds = time.time() - optical_start
    gradient_start = time.time()
    atom_count = len(atoms)
    gradients = np.empty((nroots + 1, atom_count, 3), dtype=np.float64)
    try:
        ground_gradient = mf.nuc_grad_method()
        ground_gradient.grid_response = options.get("ground_state_grid_response", False)
        gradients[0] = _to_numpy(
            ground_gradient.kernel(), cupy
        ).astype(np.float64, copy=False)
    except Exception as exc:
        raise GPU4PySCFGradientError(
            f"GPU4PySCF ground-state gradient failed: "
            f"{type(exc).__name__}: {exc}",
            failed_states=(0,),
        ) from exc
    compute_nacr = bool(options.get("compute_nacr", False))
    expected_pairs = (
        excited_excited_nac_pairs(nroots) if compute_nacr else ()
    )
    nac_pairs = np.asarray(expected_pairs, dtype=np.int64).reshape(-1, 2)
    nacr_bohr_inverse = np.empty((0, atom_count, 3), dtype=np.float64)
    nac_elapsed: float | None = None
    # The NAC solver returns the S1 gradient as a byproduct of grad_state=1, so
    # the direct TDA gradient loop starts at S2 when coupling is requested.
    first_direct_state = 1
    if compute_nacr and not options.get("solvent"):
        nac_start = time.time()
        try:
            nac_solver = td.nac_gradient_method()
            nac_solver.states = list(root_states)
            nac_solver.grad_state = 1
            nac_solver.cphf_conv_tol = float(options["cphf_conv_tol"])
            nac_results = nac_solver.kernel()
        except Exception as exc:
            raise GPU4PySCFNACError(
                f"GPU4PySCF nonadiabatic coupling failed: "
                f"{type(exc).__name__}: {exc}",
                failed_states=root_states,
            ) from exc
        nac_elapsed = time.time() - nac_start
        try:
            gradients[1] = np.asarray(
                _to_numpy(nac_solver.grad_result, cupy), dtype=np.float64
            )
        except Exception as exc:
            raise GPU4PySCFNACError(
                "GPU4PySCF NAC solver did not return a usable S1 gradient: "
                f"{type(exc).__name__}: {exc}",
                failed_states=(1,),
            ) from exc
        returned_pairs = sorted(
            tuple(int(state) for state in pair) for pair in nac_results
        )
        if returned_pairs != sorted(expected_pairs):
            raise GPU4PySCFNACError(
                f"GPU4PySCF returned NAC pairs {returned_pairs}; expected "
                f"{sorted(expected_pairs)}.",
                failed_states=root_states,
            )
        try:
            nacr_bohr_inverse = np.stack(
                [
                    np.asarray(
                        _to_numpy(nac_results[pair]["de_etf_scaled"], cupy),
                        dtype=np.float64,
                    )
                    for pair in expected_pairs
                ]
            )
        except Exception as exc:
            raise GPU4PySCFNACError(
                "GPU4PySCF NAC results could not be converted to arrays: "
                f"{type(exc).__name__}: {exc}",
                failed_states=root_states,
            ) from exc
        first_direct_state = 2

    if compute_nacr and options.get('solvent'):
        nac_start = time.time()
        vectors = []
        try:
            for pair in expected_pairs:
                solver = td.nac_method()
                solver.cphf_conv_tol = float(options['cphf_conv_tol'])
                result = solver.kernel(states=pair)
                if len(result) != 4:
                    raise ValueError(f'Expected four NAC arrays for {pair}')
                vectors.append(np.asarray(_to_numpy(result[3], cupy), dtype=np.float64))
            nacr_bohr_inverse = np.stack(vectors)
        except Exception as exc:
            raise GPU4PySCFNACError(f'Solvent-aware NAC failed: {type(exc).__name__}: {exc}', failed_states=root_states) from exc
        nac_elapsed = time.time() - nac_start

    try:
        excited_gradient = td.nuc_grad_method() if nroots > 0 else None
        if excited_gradient is not None:
            excited_gradient.cphf_conv_tol = float(options["cphf_conv_tol"])
    except Exception as exc:
        raise GPU4PySCFGradientError(
            "GPU4PySCF could not construct the TDA gradient method: "
            f"{type(exc).__name__}: {exc}",
            failed_states=root_states,
        ) from exc
    for state in range(first_direct_state, nroots + 1):
        try:
            gradients[state] = _to_numpy(
                excited_gradient.kernel(state=state), cupy
            ).astype(np.float64, copy=False)
        except Exception as exc:
            raise GPU4PySCFGradientError(
                f"GPU4PySCF gradient failed for state S{state}: "
                f"{type(exc).__name__}: {exc}",
                failed_states=(state,),
            ) from exc

    vertical_start = time.time()
    vertical_results = {}
    if options.get('_compute_vertical', False):
        try:
            vertical = mf.TDA(equilibrium_solvation=False) if options.get('solvent') else td
            if vertical is not td:
                vertical.nstates = nroots
                vertical.conv_tol = float(options['tda_conv_tol'])
                vertical.max_cycle = int(options['tda_max_cycle'])
                vertical.lindep = float(options['tda_lindep'])
                vertical.kernel()
            _require_tda_convergence(vertical, nroots, cupy)
            excitation = np.asarray(_to_numpy(vertical.e, cupy), dtype=np.float64)
            if excitation.shape != (nroots,) or not np.isfinite(excitation).all():
                raise ValueError('Invalid vertical excitation energies')
            vertical_results['_vertical_excitation_energies'] = excitation * float(nist.HARTREE2EV)
        except Exception as exc:
            error = GPU4PySCFTDAError(f'Vertical TDA failed: {type(exc).__name__}: {exc}', failed_states=root_states, stage='vertical_tda')
            raise error from exc
        if options.get('_compute_vertical_optical', False):
            try:
                moments = transition_dipoles_staged(vertical, cupy)
                strengths = (2./3.)*excitation*np.einsum('si,si->s', moments, moments)
                moments, strengths = _validated_optical_outputs(moments, strengths, nroots=nroots)
                vertical_results.update(_vertical_transition_dipoles=moments, _vertical_oscillator_strengths=strengths)
            except Exception as exc:
                raise GPU4PySCFOutputError(f'Vertical optics failed: {type(exc).__name__}: {exc}', stage='vertical_optical') from exc
    vertical_seconds = time.time() - vertical_start

    # de_etf_scaled is bohr^-1; ALF stores angstrom^-1 alongside its
    # angstrom-based coordinates and eV/angstrom forces.
    nacr_angstrom_inverse = nacr_bohr_inverse / float(nist.BOHR)

    energies_ev, forces_ev_per_angstrom = convert_gpu4pyscf_atomic_units(
        energies_hartree,
        gradients,
        state_count=nroots + 1,
        atom_count=atom_count,
        hartree_to_ev=float(nist.HARTREE2EV),
        bohr_to_angstrom=float(nist.BOHR),
    )
    solvent = options.get('solvent')
    computed_optical = options.get('_compute_optical', False)
    computed_vertical = options.get('_compute_vertical', False)
    versions = {
        **vertical_results,
        "ground_state_grid_response": bool(options.get("ground_state_grid_response", False)),
        "qm_solvent": solvent,
        "qm_state_response": 'equilibrium_lr_pcm' if solvent else 'gas',
        "qm_optical_response": ('equilibrium_lr_pcm' if solvent else 'gas') if computed_optical else 'none',
        "qm_vertical_response": ('nonequilibrium_lr_pcm' if solvent else 'gas') if computed_vertical else 'none',
        "qm_vertical_root_order": ('independent ascending vertical TDA roots' if solvent
                                   else 'same as state TDA roots') if computed_vertical else 'none',
        "qm_vertical_seconds": vertical_seconds,
        "qm_nac_solver": ('solvent_pairwise' if options.get('solvent') else 'gas_multistate') if compute_nacr else 'none',
        "_transition_dipoles": transition_dipoles,
        "_oscillator_strengths": oscillator_strengths,
        "qm_scf_seconds": scf_seconds,
        "qm_tda_seconds": tda_seconds,
        "qm_optical_seconds": optical_seconds,
        "qm_gradient_seconds": time.time() - gradient_start - (nac_elapsed or 0.0) - vertical_seconds,
        "gpu4pyscf_num_devices": int(getattr(getattr(gpu4pyscf, "__config__", None), "num_devices", 1)),
        "pyscf_version": str(getattr(pyscf, "__version__", "unknown")),
        "gpu4pyscf_version": str(
            getattr(gpu4pyscf, "__version__", "unknown")
        ),
        "cupy_version": str(getattr(cupy, "__version__", "unknown")),
        "s1_gradient_source": (
            "none" if nroots == 0
            else "nac_solver" if compute_nacr and not options.get("solvent") else "tda_gradient"
        ),
    }
    if nac_elapsed is not None:
        versions["qm_nac_seconds"] = float(nac_elapsed)
    return (
        energies_ev,
        forces_ev_per_angstrom,
        nacr_angstrom_inverse,
        nac_pairs,
        versions,
    )


def _convergence_metadata_for_failure(stage: str) -> dict[str, bool]:
    return {
        "qm_scf_converged": stage in {"tda", "optical", "nac", "gradient", "output", "vertical_tda", "vertical_optical"},
        "qm_tda_converged": stage in {"optical", "nac", "gradient", "output", "vertical_tda", "vertical_optical"},
        "qm_nac_converged": stage in {"gradient", "output", "vertical_tda", "vertical_optical"},
    }


def label_gpu4pyscf_excited_state_molecule(
    molecule_object: MoleculesObject,
    *,
    QM_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any] | None = None,
    gpus_per_node: int | None = None,
) -> MoleculesObject:
    """Label one molecule with GPU4PySCF state energies and forces."""

    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")

    qm_options = dict(QM_config or {})
    sampler_options = dict(sampler_config or {})
    start_time = time.time()
    options: dict[str, Any] = {}
    device: dict[str, Any] = {}
    try:
        options = validate_gpu4pyscf_config(qm_options)
        state_table, gap_table = validate_gpu4pyscf_properties(
            properties_list,
            nroots=options["nroots"],
            compute_nacr=bool(options["compute_nacr"]),
        )
        options["_compute_optical"] = bool(
            {"transition_dipoles", "oscillator_strengths"}.intersection(properties_list))
        options['_compute_vertical'] = bool(VERTICAL_PROPERTIES.intersection(properties_list))
        options['_compute_vertical_optical'] = bool({'vertical_transition_dipoles', 'vertical_oscillator_strengths'}.intersection(properties_list))
        atoms = molecule_object.get_atoms()
        if atoms is None or len(atoms) < 1:
            raise ValueError(
                "GPU4PySCF cannot label a molecule without atoms."
            )
        if bool(np.any(atoms.get_pbc())):
            raise NotImplementedError(
                "GPU4PySCF excited-state labeling currently supports only "
                "nonperiodic molecules."
            )
        coordinates = np.asarray(atoms.get_positions(), dtype=np.float64)
        atomic_numbers = np.asarray(
            atoms.get_atomic_numbers(), dtype=np.int64
        )
        if coordinates.shape != (len(atoms), 3) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError(
                "GPU4PySCF coordinates must have shape [atoms, 3] and be "
                "finite."
            )
        if np.any(atomic_numbers <= 0) or np.any(
            atomic_numbers >= len(chemical_symbols)
        ):
            raise ValueError(
                "GPU4PySCF atomic numbers must identify valid elements."
            )
        offset, offset_source = _energy_offset(
            qm_options, sampler_options
        )
        device = select_gpu4pyscf_device(gpus_per_node)
        (
            energies,
            forces,
            nacr,
            nac_pairs,
            backend_metadata,
        ) = run_gpu4pyscf_states(
            atoms,
            options,
            device_index=int(device["index"]),
        )
        energies, forces = _validated_calculated_outputs(
            energies,
            forces,
            state_count=len(state_table),
            atom_count=len(atoms),
        )
        if options["compute_nacr"]:
            nacr, nac_pairs = _validated_nac_outputs(
                nacr,
                nac_pairs,
                nroots=int(options["nroots"]),
                atom_count=len(atoms),
            )
        optical_results = {}
        if {"transition_dipoles", "oscillator_strengths"}.intersection(properties_list):
            try:
                dipoles, strengths = _validated_optical_outputs(
                    backend_metadata.pop("_transition_dipoles"),
                    backend_metadata.pop("_oscillator_strengths"), nroots=options["nroots"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GPU4PySCFOutputError("Missing or invalid optical output.") from exc
            if "transition_dipoles" in properties_list:
                optical_results["transition_dipoles"] = dipoles
            if "oscillator_strengths" in properties_list:
                optical_results["oscillator_strengths"] = strengths
        if "absolute_state_energies" in properties_list:
            optical_results["absolute_state_energies"] = energies.copy()
        for key in VERTICAL_PROPERTIES:
            value = backend_metadata.pop('_'+key, None)
            if key in properties_list:
                expected = (int(options['nroots']), 3) if key=='vertical_transition_dipoles' else (int(options['nroots']),)
                array = np.asarray(value, dtype=np.float64)
                if array.shape != expected or not np.isfinite(array).all():
                    raise GPU4PySCFOutputError('Missing or invalid '+key)
                if key=='vertical_oscillator_strengths' and np.any(array < 0):
                    raise GPU4PySCFOutputError('Negative vertical oscillator strength')
                optical_results[key] = array
        backend_metadata.pop("_transition_dipoles", None)
        backend_metadata.pop("_oscillator_strengths", None)
    except Exception as exc:
        elapsed = time.time() - start_time
        stage = str(getattr(exc, "stage", "configuration"))
        failed_states = [
            int(state) for state in getattr(exc, "failed_states", ())
        ]
        error_metadata: dict[str, Any] = {
            "qm_backend": "gpu4pyscf",
            "qm_solvent": options.get("solvent"),
            "qm_error": str(exc),
            "qm_error_type": type(exc).__name__,
            "qm_failed_stage": stage,
            "qm_elapsed_seconds": float(elapsed),
            **_convergence_metadata_for_failure(stage),
        }
        if failed_states:
            error_metadata["qm_failed_states"] = failed_states
        if device:
            error_metadata.update(
                {
                    "qm_device": str(device["label"]),
                    "qm_device_index": int(device["index"]),
                    "qm_visible_gpu_count": int(device["visible_count"]),
                    "parsl_worker_rank": int(device["worker_rank"]),
                    "cuda_visible_devices": device[
                        "cuda_visible_devices"
                    ],
                }
            )
        for property_key in dict(properties_list or {}):
            molecule_object.get_results().pop(str(property_key), None)
        for key in dict(properties_list or {}):
            molecule_object.get_results().pop(key, None)
        molecule_object.update_metadata(error_metadata)
        molecule_object.set_converged_flag(False)
        return molecule_object

    results: dict[str, Any] = dict(optical_results)
    for row in state_table:
        state = int(row["state"])
        results[str(row["energy_key"])] = float(energies[state] - offset)
        results[str(row["force_key"])] = np.asarray(
            forces[state], dtype=np.float64
        )
    for row in gap_table:
        results[str(row["gap_key"])] = float(
            results[str(row["upper_energy_key"])]
            - results[str(row["lower_energy_key"])]
        )
    if options["compute_nacr"]:
        results[NACR_PROPERTY_KEY] = np.asarray(nacr, dtype=np.float64)

    elapsed = time.time() - start_time
    nac_metadata: dict[str, Any] = {
        "compute_nacr": bool(options["compute_nacr"]),
        "qm_nac_converged": True,
    }
    if options["compute_nacr"]:
        nac_metadata.update(
            {
                # store_pair_index reads nac_pairs to write the shard's pair
                # index dataset, so this key is load-bearing, not just record.
                "nac_pairs": [
                    [int(pair[0]), int(pair[1])] for pair in nac_pairs
                ],
                "nac_scope": NAC_SCOPE,
                "nacr_units": NACR_UNITS,
                "nac_pair_count": int(len(nac_pairs)),
                # NACR carries an arbitrary global sign per pair; nothing is
                # trained on it, so the raw solver sign is stored unaligned.
                "nacr_sign_convention": "raw_solver_arbitrary_global_sign",
            }
        )
    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            **nac_metadata,
            "qm_backend": "gpu4pyscf",
            "qm_device": str(device["label"]),
            "qm_device_index": int(device["index"]),
            "qm_visible_gpu_count": int(device["visible_count"]),
            "parsl_worker_rank": int(device["worker_rank"]),
            "cuda_visible_devices": device["cuda_visible_devices"],
            "qm_elapsed_seconds": float(elapsed),
            "qm_scf_converged": True,
            # A ground-state calculation (nroots 0) runs no TDA.
            "qm_tda_converged": True if int(options["nroots"]) > 0 else None,
            "xc": str(options["xc"]),
            "basis": str(options["basis"]),
            "charge": int(options["charge"]),
            "multiplicity": int(options["multiplicity"]),
            "density_fit": bool(options["density_fit"]),
            "auxbasis": options["auxbasis"],
            "grids_level": int(options["grids_level"]),
            "nroots": int(options["nroots"]),
            "n_states": len(state_table),
            "scf_conv_tol": float(options["scf_conv_tol"]),
            "scf_max_cycle": int(options["scf_max_cycle"]),
            "tda_conv_tol": float(options["tda_conv_tol"]),
            "tda_max_cycle": int(options["tda_max_cycle"]),
            "energy_offset_eV": float(offset),
            "energy_offset_source": offset_source,
            "gap_properties": [
                str(row["gap_key"]) for row in gap_table
            ],
            **dict(backend_metadata),
        }
    )
    molecule_object.set_converged_flag(True)
    return molecule_object


@python_app(executors=["alf_QM_executor"])
@timed('QM')
def gpu4pyscf_excited_state_task(
    molecule_object,
    QM_config,
    properties_list,
    sampler_config=None,
    gpus_per_node=None,
):
    """Label one molecule using GPU4PySCF on ALF's QM executor."""

    return label_gpu4pyscf_excited_state_molecule(
        molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=gpus_per_node,
    )
