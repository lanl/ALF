"""An ASE calculator for PySCF (CPU) and GPU4PySCF (GPU), for molecules and periodic cells.

``PySCFCalculator(**settings)`` takes plain settings, the same ones ``pyscf_task`` reads from its QM
configuration, so it can be used anywhere ASE is, such as MD or geometry optimization::

    from alframework.qm_interfaces.pyscf.calculator import PySCFCalculator
    atoms.calc = PySCFCalculator(backend="cpu", xc="pbe", basis="def2-svp")
    atoms.get_forces()

The system type follows ``atoms.pbc``: no periodic direction gives a molecule, and all three give a
periodic cell. Results are in eV, eV/Å, and eV/Å³ (ASE's stress convention, ``(1/V) dE/dε``).

* ``energy`` and ``forces`` are the ground state's.
* ``stress`` is available for periodic cells calculated with ``df: "fftdf"``, or on the GPU.
* With ``nroots > 0``, for molecules, ``state_energies`` ``[nroots + 1]`` and ``state_forces``
  ``[nroots + 1, atoms, 3]`` hold the ground state and the TDA excited states.

With ``periodic.smearing_eV``, Fermi smearing of that width, ``energy`` is the free energy ``E - σS``, which the
SCF minimises and the forces are the gradient of.

``atoms.info["charge"]`` and ``atoms.info["mult"]`` override the ``charge`` and ``multiplicity``
settings; open-shell structures use UKS. Periodic cells must be neutral.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols

from alframework.qm_interfaces.pyscf.gpu4pyscf import (
    GPU4PySCFDependencyError,
    GPU4PySCFError,
    GPU4PySCFGradientError,
    GPU4PySCFOutputError,
    GPU4PySCFSCFError,
    _integer,
    _positive_float,
    _single_device_backend,
    _to_numpy,
    run_gpu4pyscf_states,
    validate_gpu4pyscf_config,
)

BACKENDS = ("cpu", "gpu")
DENSITY_FITTING = ("multigrid", "fftdf", "gdf")
PERIODIC_DEFAULTS = {"basis": "gth-dzvp", "pseudo": "gth-pbe", "kpts": [1, 1, 1], "df": "multigrid",
                     "ke_cutoff": None, "smearing_eV": None}
DEFAULT_SETTINGS = {
    "backend": "cpu", "xc": "pbe", "basis": "def2-svp", "charge": 0, "multiplicity": 1,
    "density_fit": True, "auxbasis": None, "grids_level": 3, "scf_conv_tol": 1.0e-10, "scf_max_cycle": 100,
    "nroots": 0, "tda_conv_tol": 1.0e-8, "tda_lindep": 1.0e-20, "cphf_conv_tol": 1.0e-8, "tda_max_cycle": 200,
    "num_threads": 1, "max_memory_mb": None, "verbosity": 0, "periodic": {},
}


def validate_pyscf_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Return complete, checked settings for ``PySCFCalculator`` and ``pyscf_task``."""
    if settings is not None and not isinstance(settings, dict):
        raise TypeError("PySCF settings must be a dictionary.")
    unknown = sorted(set(settings or {}) - set(DEFAULT_SETTINGS) - {"batching", "energy_offset_eV"})
    if unknown:
        raise ValueError(f"Unknown PySCF settings: {unknown}. Known: {sorted(DEFAULT_SETTINGS)}.")
    options = dict(DEFAULT_SETTINGS)
    options.update({key: value for key, value in (settings or {}).items() if key in DEFAULT_SETTINGS})
    if options["backend"] not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {options['backend']!r}.")
    for key in ("xc", "basis"):
        if not isinstance(options[key], str) or not options[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    options["charge"] = _integer(options["charge"], name="charge")
    options["multiplicity"] = _integer(options["multiplicity"], name="multiplicity", minimum=1)
    if not isinstance(options["density_fit"], bool):
        raise ValueError("density_fit must be true or false.")
    options["grids_level"] = _integer(options["grids_level"], name="grids_level", minimum=0)
    options["nroots"] = _integer(options["nroots"], name="nroots", minimum=0)
    for key in ("scf_conv_tol", "tda_conv_tol", "tda_lindep", "cphf_conv_tol"):
        options[key] = _positive_float(options[key], name=key)
    for key in ("scf_max_cycle", "tda_max_cycle", "num_threads"):
        options[key] = _integer(options[key], name=key, minimum=1)
    options["verbosity"] = _integer(options["verbosity"], name="verbosity", minimum=0)
    if options["max_memory_mb"] is not None:
        options["max_memory_mb"] = _positive_float(options["max_memory_mb"], name="max_memory_mb")
    if options["backend"] == "cpu" and options["nroots"] > 0 and options["density_fit"]:
        raise ValueError("PySCF on the CPU has no excited-state forces with density fitting; "
                         "set density_fit to false, or use backend 'gpu'.")
    periodic = options["periodic"]
    if not isinstance(periodic, dict):
        raise ValueError("periodic must be a dictionary.")
    unknown = sorted(set(periodic) - set(PERIODIC_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown periodic settings: {unknown}. Known: {sorted(PERIODIC_DEFAULTS)}.")
    periodic = dict(PERIODIC_DEFAULTS, **periodic)
    if periodic["df"] not in DENSITY_FITTING:
        raise ValueError(f"periodic.df must be one of {DENSITY_FITTING}, not {periodic['df']!r}.")
    kpts = periodic["kpts"]
    if (not isinstance(kpts, (list, tuple)) or len(kpts) != 3
            or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in kpts)):
        raise ValueError("periodic.kpts must be three positive integers, a k-point mesh such as [1, 1, 1].")
    periodic["kpts"] = [int(k) for k in kpts]
    for key in ("ke_cutoff", "smearing_eV"):
        if periodic[key] is not None:
            periodic[key] = _positive_float(periodic[key], name=f"periodic.{key}")
    if periodic["df"] == "multigrid" and periodic["kpts"] != [1, 1, 1]:
        raise ValueError("periodic.df 'multigrid' gives forces at the Gamma point only; use 'fftdf' for k-points.")
    if periodic["df"] == "gdf" and options["backend"] == "cpu":
        raise ValueError("PySCF has no CPU forces with periodic.df 'gdf'; use 'multigrid' or 'fftdf', "
                         "or backend 'gpu'.")
    options["periodic"] = periodic
    return options


def _to_gpu(mf):
    """Move a PySCF method to GPU4PySCF. PySCF's ``to_gpu`` maps each class to its GPU4PySCF counterpart."""
    return mf.to_gpu()


def _molecule_state(atoms, options):
    """Charge and multiplicity: the structure's own, if it records them, otherwise the settings'."""
    charge = int(atoms.info.get("charge", options["charge"]))
    mult = int(atoms.info.get("mult", options["multiplicity"]))
    electrons = int(sum(atoms.get_atomic_numbers())) - charge
    if mult < 1 or (electrons - (mult - 1)) % 2:
        raise ValueError(f"Multiplicity {mult} is impossible with {electrons} electrons.")
    return charge, mult


def _system_kind(atoms):
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not pbc.any():
        return "molecule"
    if pbc.all():
        return "periodic"
    raise ValueError(f"PySCFCalculator needs all or none of the directions periodic, not pbc={pbc.tolist()}.")


class PySCFCalculator(Calculator):
    """ASE calculator for PySCF and GPU4PySCF DFT. See the module docstring."""

    # With no smearing, free_energy equals energy; ASE's optimizers and finite-difference helpers ask for it.
    implemented_properties = ["energy", "free_energy", "forces", "stress", "state_energies", "state_forces"]

    def __init__(self, restart=None, label=None, atoms=None, directory=".", **settings):
        self.options = validate_pyscf_settings(settings)
        Calculator.__init__(self, restart=restart, label=label, atoms=atoms, directory=directory)
        self.metadata: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        kind = _system_kind(self.atoms)
        if kind == "periodic" and self.options["nroots"] > 0:
            raise ValueError("Excited states (nroots > 0) are available for molecules only.")
        if self.options["nroots"] > 0:
            self._excited_states()
        else:
            self._ground_state(kind, want_forces="forces" in properties or "state_forces" in properties,
                               want_stress="stress" in properties)

    # The ground state, for molecules and periodic cells, on either backend.

    def _modules(self):
        pyscf = importlib.import_module("pyscf")
        nist = importlib.import_module("pyscf.data.nist")
        cupy = None
        if self.options["backend"] == "gpu":
            try:
                _single_device_backend(pyscf)
                cupy = importlib.import_module("cupy")
            except GPU4PySCFError:
                raise
            except Exception as exc:
                raise GPU4PySCFDependencyError(
                    "backend 'gpu' needs GPU4PySCF and a CuPy build matching the worker's CUDA runtime.") from exc
        return pyscf, nist, cupy

    def _build(self, kind, charge, mult, pyscf):
        options, periodic = self.options, self.options["periodic"]
        atom = [(chemical_symbols[int(z)], tuple(float(x) for x in position))
                for z, position in zip(self.atoms.get_atomic_numbers(), self.atoms.get_positions())]
        common = dict(atom=atom, charge=charge, spin=mult - 1, unit="Angstrom", verbose=options["verbosity"])
        if options["max_memory_mb"] is not None:
            common["max_memory"] = float(options["max_memory_mb"])
        if kind == "molecule":
            dft = importlib.import_module("pyscf.dft")
            mol = pyscf.M(basis=options["basis"], **common)
            mf = (dft.RKS if mult == 1 else dft.UKS)(mol, xc=options["xc"])
            if options["density_fit"]:
                mf = mf.density_fit() if options["auxbasis"] is None else mf.density_fit(auxbasis=options["auxbasis"])
        else:
            if charge != 0:
                raise ValueError(f"Periodic cells must be neutral, but this one has charge {charge}.")
            gto = importlib.import_module("pyscf.pbc.gto")
            dft = importlib.import_module("pyscf.pbc.dft")
            cell = gto.Cell()
            for key, value in common.items():
                setattr(cell, key, value)
            cell.a = np.asarray(self.atoms.get_cell(), dtype=float)
            cell.basis, cell.pseudo = periodic["basis"], periodic["pseudo"]
            if periodic["ke_cutoff"] is not None:
                cell.ke_cutoff = periodic["ke_cutoff"]
            cell.build()
            if periodic["df"] == "multigrid":
                mf = (dft.RKS if mult == 1 else dft.UKS)(cell, xc=options["xc"]).multigrid_numint()
            else:
                kpts = cell.make_kpts(periodic["kpts"])
                mf = (dft.KRKS if mult == 1 else dft.KUKS)(cell, kpts=kpts, xc=options["xc"])
                if periodic["df"] == "gdf":
                    mf = mf.density_fit()
            if periodic["smearing_eV"] is not None:
                nist = importlib.import_module("pyscf.data.nist")
                smearing = importlib.import_module("pyscf.pbc.scf.addons")
                mf = smearing.smearing_(mf, sigma=periodic["smearing_eV"] / float(nist.HARTREE2EV), method="fermi")
        mf.conv_tol = options["scf_conv_tol"]
        mf.max_cycle = options["scf_max_cycle"]
        if hasattr(mf, "grids") and getattr(mf.grids, "level", None) is not None:
            mf.grids.level = options["grids_level"]
        if self.options["backend"] == "gpu":
            mf = _to_gpu(mf)
        return mf

    def _ground_state(self, kind, want_forces, want_stress):
        if want_stress and kind != "periodic":
            raise ValueError("stress is defined for periodic cells only.")
        if want_stress and self.options["backend"] == "cpu" and self.options["periodic"]["df"] != "fftdf":
            raise ValueError("On the CPU, PySCF gives stress only with periodic.df 'fftdf'.")
        pyscf, nist, cupy = self._modules()
        charge, mult = _molecule_state(self.atoms, self.options)
        pyscf.lib.num_threads(self.options["num_threads"])
        start = time.time()
        try:
            mf = self._build(kind, charge, mult, pyscf)
            mf.kernel()
        except (GPU4PySCFError, ValueError):
            raise
        except Exception as exc:
            raise GPU4PySCFSCFError(f"PySCF SCF failed: {type(exc).__name__}: {exc}", failed_states=(0,)) from exc
        if not bool(_to_numpy(getattr(mf, "converged", False), cupy)):
            raise GPU4PySCFSCFError(f"The {type(mf).__name__} SCF did not converge in {self.options['scf_max_cycle']} "
                                    "cycles.", failed_states=(0,))
        scf_seconds = time.time() - start
        # With smearing, the SCF minimises the free energy E - σS, and the forces are its gradient, so it's the
        # energy reported.
        smeared = getattr(mf, "e_free", None) is not None
        energy = float(_to_numpy(mf.e_free if smeared else mf.e_tot, cupy)) * float(nist.HARTREE2EV)
        if not np.isfinite(energy):
            raise GPU4PySCFOutputError("PySCF returned a non-finite energy.", failed_states=(0,))
        self.results["energy"] = self.results["free_energy"] = energy
        self.results["state_energies"] = np.array([energy])
        start = time.time()
        if want_forces or want_stress:
            try:
                gradients = mf.Gradients()
                if want_forces:
                    gradient = np.asarray(_to_numpy(gradients.kernel(), cupy), dtype=float)
                    forces = -gradient * float(nist.HARTREE2EV) / float(nist.BOHR)
                    if forces.shape != (len(self.atoms), 3) or not np.isfinite(forces).all():
                        raise GPU4PySCFOutputError("PySCF returned malformed or non-finite forces.")
                    self.results["forces"] = forces
                    self.results["state_forces"] = forces[None]
                if want_stress:
                    sigma = np.asarray(_to_numpy(gradients.get_stress(), cupy), dtype=float)
                    sigma = 0.5 * (sigma + sigma.T) * float(nist.HARTREE2EV) / float(nist.BOHR) ** 3
                    self.results["stress"] = sigma[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
            except GPU4PySCFError:
                raise
            except Exception as exc:
                raise GPU4PySCFGradientError(f"PySCF gradient failed: {type(exc).__name__}: {exc}",
                                             failed_states=(0,)) from exc
        self.metadata = {
            "qm_backend": "pyscf" if self.options["backend"] == "cpu" else "gpu4pyscf",
            "qm_system": kind, "qm_method": type(mf).__name__, "charge": charge, "mult": mult,
            "qm_scf_seconds": scf_seconds, "qm_gradient_seconds": time.time() - start,
            **({"smearing_eV": self.options["periodic"]["smearing_eV"], "smearing_entropy": float(mf.entropy),
                "energy_is": "free energy, E - sigma*S"} if smeared else {}),
            "pyscf_version": str(getattr(pyscf, "__version__", "unknown")),
        }

    # Excited states, for molecules: the shared RKS/TDA path.

    def _excited_states(self):
        charge, mult = _molecule_state(self.atoms, self.options)
        if mult != 1:
            raise ValueError("Excited states are available for singlets only (RKS/TDA).")
        keys = ("xc", "basis", "density_fit", "auxbasis", "grids_level", "nroots", "scf_conv_tol", "tda_conv_tol",
                "tda_lindep", "cphf_conv_tol", "scf_max_cycle", "tda_max_cycle", "num_threads", "max_memory_mb",
                "verbosity")
        options = validate_gpu4pyscf_config({key: self.options[key] for key in keys} | {"charge": charge})
        energies, forces, _, _, versions = run_gpu4pyscf_states(
            self.atoms, options, device_index=0, backend=self.options["backend"])
        self.results.update(energy=float(energies[0]), free_energy=float(energies[0]), forces=np.asarray(forces[0]),
                            state_energies=np.asarray(energies), state_forces=np.asarray(forces))
        self.metadata = {"qm_backend": "pyscf" if self.options["backend"] == "cpu" else "gpu4pyscf",
                         "qm_system": "molecule", "charge": charge, "mult": mult,
                         **{k: v for k, v in versions.items() if not k.startswith("_")}}


__all__ = ["PySCFCalculator", "validate_pyscf_settings", "DEFAULT_SETTINGS", "PERIODIC_DEFAULTS"]
