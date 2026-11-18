"""Stand-ins for QM programs, so ASE's own ORCA and Q-Chem writers and parsers run without licensed codes.

Each stand-in is a small Python script that checks what ASE wrote and prints output in the program's format.
"""

import inspect
import shlex
import sys

from alframework.qm_interfaces.ase import generic

# The generic ASE task's function, called directly in the worker's process.
label_with_ase = inspect.unwrap(generic.ase_calculator_task.func)

EMT_CONFIG = {"ASE_calculator": "ase.calculators.emt.EMT"}


def executable(tmp_path, body):
    """A command that runs the Python ``body``, from a path with a space in it."""
    script = tmp_path / "controlled chemistry.py"
    script.write_text(body)
    return shlex.quote(sys.executable) + " " + shlex.quote(str(script))


def qchem_output(natoms=2):
    """Q-Chem output with an energy of -2.5 Hartree and every gradient component 0.1 Hartree/Bohr."""
    lines = [" Total energy in the final basis set = -2.5000000000", " Gradient of SCF Energy"]
    for offset in range(0, natoms, 6):
        count = min(6, natoms - offset)
        lines.append(" atom headers")
        for axis in range(3):
            lines.append(f"    {axis + 1}" + "".join(f"{0.1:12.8f}" for _ in range(count)))
        if offset + count == natoms:
            lines.append(" Max gradient component = .1")
    return "\n".join(lines) + "\n"


def qchem_command(tmp_path, output=None, exit_code=0):
    """ASE's Q-Chem command form, ``<program> PREFIX.inp PREFIX.out``, writing ``output``."""
    output = qchem_output() if output is None else output
    body = ("import sys\nfrom pathlib import Path\n"
            f"Path(sys.argv[2]).write_text({output!r})\nsys.exit({exit_code})\n")
    return executable(tmp_path, body) + " PREFIX.inp PREFIX.out"


def orca_output(normal=True, energy="-2.5", dipole=True):
    """ORCA output for H2: an energy in Hartree, a dipole in atomic units, and normal termination."""
    return ("Number of atoms .... 2\nCARTESIAN COORDINATES (ANGSTROEM)\n-------------------\n"
            "H 0 0 0\nH 0 0 .74\nSum of atomic charges 0.0\n"
            + ("Total Dipole Moment 1.0 2.0 3.0\n" if dipole else "")
            + f"FINAL SINGLE POINT ENERGY {energy}\n"
            + ("ORCA TERMINATED NORMALLY\n" if normal else "interrupted\n"))


def orca_command(tmp_path, *, normal=True, energy="-2.5", dipole=True, gradient=True, exit_code=0):
    """An ORCA stand-in that checks it was given ``orca.inp`` and writes its output and gradient."""
    text = orca_output(normal, energy, dipole)
    gradient_text = "# The current gradient\n#\n" + "\n".join(["0.1"] * 6) + "\n# The atomic numbers\n"
    return executable(tmp_path,
                      "import sys\nfrom pathlib import Path\n"
                      'assert Path(sys.argv[1]).name == "orca.inp"\n'
                      "assert Path(sys.argv[1]).exists()\n"
                      + (f'Path("orca.engrad").write_text({gradient_text!r})\n' if gradient else "")
                      + f"print({text!r})\nsys.exit({exit_code})\n")
