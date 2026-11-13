"""Label structures with PySCF (CPU) or GPU4PySCF (GPU): molecules and periodic cells.

``pyscf_task`` labels one structure per task through ``PySCFCalculator``. Its QM configuration holds the
calculator's settings; see ``alframework.qm_interfaces.pyscf.calculator``. Properties use state names:

* ``E0``/``F0``: the ground-state energy and forces, for molecules and periodic cells.
* ``E1``/``F1``, ... and gaps ``dE01``, ...: TDA excited states, for molecules, with ``nroots`` set to the
  number of excited states.
* ``stress``: the periodic stress, in ASE's Voigt order and eV/Å³.

The first element of each property is the shard's dataset name. For ground-state training or fine-tuning,
store ``"E0": ["energy", "system", 1.0], "F0": ["forces", "atomic", 1.0]``. For nonadiabatic couplings,
optical properties, or PCM solvent, use ``gpu4pyscf_excited_state_task``.

The package:

* ``alframework.qm_interfaces.pyscf``: ``pyscf_task``, ALF's labeler.
* ``alframework.qm_interfaces.pyscf.calculator``: ``PySCFCalculator``, an ASE calculator usable outside ALF.
* ``alframework.qm_interfaces.pyscf.gpu4pyscf``: ``gpu4pyscf_excited_state_task``, the GPU excited-state labeler with
  couplings, optical properties, and solvent, and the molecular state code the calculator shares.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from parsl import python_app
from alframework.tools.task_timeline import timed

from alframework.tools.molecules_class import MoleculesObject

STRESS_KEY = "stress"
QM_CONFIG_EXTRAS = ("batching", "energy_offset_eV")


def validate_pyscf_properties(properties_list, *, nroots):
    """Return the state rows, gap rows, and whether stress is requested, after checking them."""
    from alframework.tools.excited_state_tools import derive_gap_property_table, derive_state_property_table

    states = derive_state_property_table(properties_list)
    if len(states) != int(nroots) + 1:
        raise ValueError(f"pyscf_task labels nroots + 1 states: nroots={nroots} needs E0 through E{nroots}, "
                         f"but properties_list has {len(states)}.")
    gaps = derive_gap_property_table(properties_list)
    known = {row[key] for row in states for key in ("energy_key", "force_key") if row[key]}
    known |= {row["gap_key"] for row in gaps}
    if STRESS_KEY in properties_list:
        if str(properties_list[STRESS_KEY][1]).lower() != "system":
            raise ValueError("properties_list['stress'] must be a system property.")
        known.add(STRESS_KEY)
    unknown = sorted(set(properties_list) - known)
    if unknown:
        raise ValueError(f"pyscf_task gives only E#, F#, dE##, and stress, not {unknown}. For couplings or optical "
                         "properties, use gpu4pyscf_excited_state_task.")
    return states, gaps, STRESS_KEY in properties_list


def _settings(QM_config):
    return {key: value for key, value in dict(QM_config or {}).items() if key not in QM_CONFIG_EXTRAS}


def validate_pyscf_config(QM_config, master_config=None):
    """Check a ``pyscf_task`` configuration when ALF starts, so a mistake doesn't fail every task."""
    from alframework.qm_interfaces.pyscf.calculator import validate_pyscf_settings

    options = validate_pyscf_settings(_settings(QM_config))
    properties = (master_config or {}).get("properties_list")
    if properties is not None:
        validate_pyscf_properties(properties, nroots=options["nroots"])
    return options


def _offset(QM_config, sampler_config):
    from alframework.qm_interfaces.pyscf.gpu4pyscf import _energy_offset
    return _energy_offset(dict(QM_config or {}), dict(sampler_config or {}))


def label_pyscf_molecule(molecule_object, QM_config, properties_list, sampler_config=None, gpus_per_node=None):
    """Label one structure, or mark it unconverged with the reason."""
    from alframework.qm_interfaces.pyscf.gpu4pyscf import select_gpu4pyscf_device
    from alframework.qm_interfaces.pyscf.calculator import PySCFCalculator

    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")
    start = time.time()
    device = None
    calculator = None
    try:
        options = validate_pyscf_config(QM_config, {"properties_list": properties_list})
        states, gaps, want_stress = validate_pyscf_properties(properties_list, nroots=options["nroots"])
        offset, offset_source = _offset(QM_config, sampler_config)
        atoms = molecule_object.get_atoms().copy()
        metadata = molecule_object.get_metadata()
        for key in ("charge", "mult"):
            if key in metadata:
                atoms.info[key] = int(metadata[key])
        if options["backend"] == "gpu":
            device = select_gpu4pyscf_device(gpus_per_node)
        calculator = PySCFCalculator(**options)
        wanted = ["energy"] + (["forces"] if any(row["force_key"] for row in states) else [])
        wanted += [STRESS_KEY] if want_stress else []
        calculator.calculate(atoms, wanted)
        results = {}
        for row in states:
            results[row["energy_key"]] = float(calculator.results["state_energies"][row["state"]]) - offset
            if row["force_key"]:
                results[row["force_key"]] = np.asarray(calculator.results["state_forces"][row["state"]], dtype=float)
        for row in gaps:
            results[row["gap_key"]] = results[row["upper_energy_key"]] - results[row["lower_energy_key"]]
        if want_stress:
            results[STRESS_KEY] = np.asarray(calculator.results[STRESS_KEY], dtype=float)
    except Exception as exc:
        molecule_object.set_converged_flag(False)
        molecule_object.update_metadata({
            "qm_backend": "pyscf", "qm_error": f"{type(exc).__name__}: {exc}", "qm_error_type": type(exc).__name__,
            "qm_failed_stage": str(getattr(exc, "stage", "configuration")),
            "qm_elapsed_seconds": time.time() - start,
            **({"qm_failed_states": list(exc.failed_states)} if getattr(exc, "failed_states", None) else {}),
            **({"qm_device": device["label"]} if device else {}),
        })
        return molecule_object

    molecule_object.store_results(results)
    molecule_object.set_converged_flag(True)
    molecule_object.update_metadata({
        **calculator.metadata, "qm_scf_converged": True, "qm_tda_converged": True if options["nroots"] else None,
        "xc": options["xc"], "basis": options["basis"] if calculator.metadata["qm_system"] == "molecule"
        else options["periodic"]["basis"], "nroots": options["nroots"], "n_states": len(states),
        "energy_offset_eV": offset, "energy_offset_source": offset_source, "qm_elapsed_seconds": time.time() - start,
        **({"periodic": options["periodic"]} if calculator.metadata["qm_system"] == "periodic" else {}),
        **({"qm_device": device["label"], "cuda_visible_devices": device["cuda_visible_devices"]} if device else {}),
    })
    return molecule_object


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def pyscf_task(molecule_object, QM_config, properties_list, sampler_config=None, gpus_per_node=None):
    """Label one structure with PySCF or GPU4PySCF on ALF's QM executor. Works with ``"mode": "sequential"``."""
    return label_pyscf_molecule(molecule_object, QM_config, properties_list, sampler_config, gpus_per_node)


# LabelingCoordinator calls this at startup. The docs build replaces python_app with the plain function.
getattr(pyscf_task, "func", pyscf_task).validate_config = validate_pyscf_config
