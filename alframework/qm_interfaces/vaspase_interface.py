"""Deprecated legacy VASP ASE interface.

This module is kept for compatibility with older imports. New ALF workflows
should use ``alframework.qm_interfaces.ase_calculator_interface`` instead,
especially ``VASP_ase_calculator_task`` for VASP calculations through ASE.
"""

import os
import pickle as pkl
import shutil
import warnings
from contextlib import contextmanager

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp


class SCFConvergenceFailure(Exception):
    """Raised when a VASP calculation finishes without SCF convergence."""


@contextmanager
def _temporary_environment(updates):
    old_values = {}
    for key, value in updates.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _atoms_from_input(molecule):
    if isinstance(molecule, Atoms):
        return molecule.copy()
    if hasattr(molecule, "get_atoms"):
        return molecule.get_atoms().copy()
    if hasattr(molecule, "S") and hasattr(molecule, "X"):
        cell = getattr(molecule, "C", None)
        pbc = cell is not None
        return Atoms(molecule.S, positions=molecule.X, cell=cell, pbc=pbc)
    raise TypeError("molecule must be an ase.Atoms, MoleculesObject, or legacy object with S/X attributes")


def _molecule_id(molecule):
    if hasattr(molecule, "get_moleculeid"):
        return molecule.get_moleculeid()
    return getattr(molecule, "ids", "vasp")


class VASPGenerator:
    """Deprecated VASP single-point helper.

    Parameters are intentionally site-neutral. Provide the VASP launch command,
    pseudopotential path, and any module/environment setup from the caller or
    Parsl worker initialization. This class no longer constructs LANL-specific
    commands or relies on MPI/GPU globals.
    """

    def __init__(
        self,
        vasp_options,
        vasp_command=None,
        scratch="./",
        output_store=None,
        rm_scratch=False,
        vasp_pp_path=None,
        environment=None,
        omp_num_threads=None,
    ):
        warnings.warn(
            "VASPGenerator is deprecated. Use "
            "alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task "
            "or ase_calculator_task for new ALF workflows.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.vasp_options = vasp_options.copy()
        self.vasp_command = vasp_command or self.vasp_options.pop("command", None)
        self.working_dir = scratch
        self.output_store = output_store
        self.rm_scratch = rm_scratch
        self.environment = {} if environment is None else environment.copy()
        if self.vasp_command is not None:
            self.environment["VASP_COMMAND"] = self.vasp_command
        if vasp_pp_path is not None:
            self.environment["VASP_PP_PATH"] = vasp_pp_path
        if omp_num_threads is not None:
            self.environment["OMP_NUM_THREADS"] = omp_num_threads

        os.makedirs(self.working_dir, exist_ok=True)
        if self.output_store is not None:
            os.makedirs(self.output_store, exist_ok=True)

        self.settings = {
            "xc": "pbe",
            "prec": "Accurate",
            "ncore": self.vasp_options.get("ncore", 1),
            "lreal": "Auto",
            "nelm": self.vasp_options.get("nelm", 120),
            "ivdw": self.vasp_options.get("ivdw", 0),
        }
        for key, val in self.vasp_options.items():
            if key == "kpoints":
                self.settings["kpts"] = val
            elif key != "command":
                self.settings[key] = val

    def get_magmom(self, atomic_no):
        """Return configured magnetic moments or a simple atomic-number default."""
        if "magmom" in self.vasp_options:
            return self.vasp_options["magmom"]
        return np.array(atomic_no).copy()

    def single_point(self, molecule, force_calculation=False, output_file=None):
        """Run one VASP calculation through ASE.

        Args:
            molecule: ``ase.Atoms``, ``MoleculesObject``, or legacy object with
                ``S`` and ``X`` attributes.
            force_calculation (bool): If True, include forces in the returned
                properties.
            output_file: Kept for legacy call compatibility. It is not used.

        Returns:
            tuple: ``(atoms, properties)`` where ``atoms`` is the final
            ``ase.Atoms`` object and ``properties`` contains ASE-unit results.
        """
        del output_file

        atoms = _atoms_from_input(molecule)
        molecule_id = _molecule_id(molecule)
        run_dir = os.path.join(self.working_dir, str(molecule_id))
        os.makedirs(run_dir, exist_ok=True)

        calc = Vasp(directory=run_dir, command=self.vasp_command, **self.settings)
        atoms.calc = calc

        properties = {}
        with _temporary_environment(self.environment):
            properties["energy"] = atoms.get_potential_energy()
            properties["stress"] = atoms.get_stress(voigt=False)
            if force_calculation:
                properties["forces"] = atoms.get_forces()

        if not getattr(calc, "converged", False):
            raise SCFConvergenceFailure("VASP calculation did not converge")

        if self.output_store is not None:
            self._copy_outputs(run_dir, molecule_id)
            pkl.dump(
                {"atoms": atoms.copy(), "props": properties},
                open(os.path.join(self.output_store, "data-" + str(molecule_id) + ".p"), "wb"),
            )

        atoms.calc = None
        if self.rm_scratch:
            shutil.rmtree(run_dir, ignore_errors=True)
        return atoms, properties

    def _copy_outputs(self, run_dir, molecule_id):
        for filename in ("OUTCAR", "POSCAR", "CONTCAR"):
            src = os.path.join(run_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(self.output_store, "data-" + str(molecule_id) + "." + filename)
                shutil.copyfile(src, dst)
