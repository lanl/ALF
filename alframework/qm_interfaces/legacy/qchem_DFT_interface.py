# The builder code goes here
import glob
import random
import os
import numpy as np
from parsl import python_app, bash_app
from alframework.tools.task_timeline import timed
import json
from pathlib import Path

import ase
from ase import Atoms
from ase import neighborlist
from ase.geometry.cell import complete_cell
from ase.io import cfg
from ase.io import read, write
from ase.data import chemical_symbols
from ase import units

from alframework.tools.tools import random_rotation_matrix
from alframework.tools.tools import build_input_dict
from alframework.tools.tools import system_checker
import random
from copy import deepcopy

import subprocess
import shlex

class qchemGenerator():
    """
    QChem 6.0
    """

    def __init__(self, scratch_path="/tmp/", store_path=None, nproc=36,\
                 unit={'energy': 'hartree', 'length': 'bohr'},\
                 qchem_env_file=None, qchem_command="", qcheminput="", qchemblocks=""):
        """
        unit used for input coordincates: Angstrom
        default units for output: atomic unit
        output parsing is for qchem 6
        """
        self.scratch_path = scratch_path
        self.store_path = store_path
        self.nproc = nproc
        self.qchem_env_fn = qchem_env_file
        self.qchem_command = qchem_command
        self.qcheminput = qcheminput if qcheminput.endswith('\n') else qcheminput+'\n'
        self.qchemblocks = qchemblocks if qchemblocks.endswith('\n') else qchemblocks+'\n'
        self.unit = unit
        # default a.u., hartree for energy, bohr for length as used for qchem
        if self.unit['energy'] == 'hartree':
            self.E_unit = 1.0
        elif self.unit['energy'] == 'ev':
            self.E_unit = 27.2113834
        else:
            raise KeyError('energy unit not implemented')
        if self.unit['length'] =='bohr':
            self.L_unit = 1.0
        elif self.unit['length'] == 'angstrom':
            self.L_unit = 0.5291772083
        else:
            raise KeyError('length unit not implemented')
        self.datacounter = 0

    def write_qchem_input(self, molecule, charge, mult, filename="qchem.in"):
        numbers = molecule.get_atomic_numbers()
        positions = molecule.get_positions()
        with open(filename, "w") as f:
            f.write(f"$molecule\n{charge} {mult}\n")
            for ix, ixyz in zip(numbers, positions):
                f.write("{} {} {} {}\n".format(ix, *ixyz))
            f.write("$end\n\n$rem\n")
            f.write(self.qcheminput)
            f.write("$end\n")
            f.write(self.qchemblocks)

    def single_point(self, molecule, charge=0, mult=1, prefix="qchem", properties=['energy','forces']):
        """
        mol : ase.atoms.Atoms object, will get chemical symbols and positions
        prefix : all the input and output file will start with this prefix, eg. qchem.in, qchem.out
        """

        scratch = str(Path(self.scratch_path).resolve())
        job_path = os.path.join(scratch, prefix)
        print(f"job_path:, {job_path}")


        filename = os.path.join(job_path, f"{prefix}.in")
        os.makedirs(job_path, exist_ok=True)
        self.write_qchem_input(molecule, charge, mult, filename=filename)

        # Preserve the configured-wrapper contract: it receives the scratch root as one argument,
        # runs beside prefix.in, and prints Q-Chem output to stdout.
        runcmd = 'set -e\n'
        if self.qchem_env_fn is not None:
            runcmd += 'source ' + shlex.quote(str(Path(self.qchem_env_fn).resolve())) + '\n'
        if not self.qchem_command.strip():
            return {'converged': False, 'error': 'Q-Chem command is empty'}
        runcmd += self.qchem_command + ' ' + shlex.quote(scratch)
        environment = dict(os.environ)
        environment.setdefault('QCSCRATCH', scratch)
        proc = subprocess.run(['bash', '-c', runcmd], cwd=job_path, env=environment,
                              capture_output=True, text=True)

        if self.store_path is not None:
            store_dir = os.path.join(self.store_path, prefix)
            outfile = os.path.join(store_dir, f'{prefix}.out')
            #os.makedirs(store_dir, exist_ok=True)
            Path(store_dir).mkdir(exist_ok=True,parents=True)
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
            Path(store_dir, f'{prefix}.stderr').write_text(proc.stderr)
        if proc.returncode:
            return {'converged': False, 'error': f'Q-Chem command exited {proc.returncode}: {proc.stderr[-1000:]}'}
        try:
            return self.parse_output(proc.stdout, len(molecule), properties)
        except (ValueError, IndexError, StopIteration) as error:
            return {'converged': False, 'error': f'Invalid Q-Chem output: {error}'}

    def parse_output(self, text, natoms, properties):
        energy = None
        for line in text.split("\n"):
            if "Total energy in the final basis set = " in line:
                energy = float(line.split()[-1].replace('D','E'))

        nblocks = natoms // 6
        if natoms%6 > 0: nblocks += 1
        grad = None
        Acharge = dipole = quadrupole = None
        out = iter(text.split("\n"))
        for line in out:
            if "Gradient of SCF Energy" in line:
                grad = np.zeros((natoms, 3))
                for block in range(nblocks):
                    next(out)
                    for i in range(3):
                        line = next(out)
                        grad_i = np.array([float(value.replace('D','E')) for value in line.split()[1:]])
                        if len(grad_i) != min(6,natoms-6*block):
                            raise ValueError('Incomplete SCF gradient block')
                        grad[6*block:6*block+len(grad_i), i] = grad_i
            elif "Ground-State Mulliken Net Atomic Charges" in line:
                Acharge = np.zeros(natoms)
                next(out)
                next(out)
                next(out)
                for curI in range(natoms):
                    line = next(out)
                    Acharge[curI] = float(line.split()[2])
            elif "Cartesian Multipole Moments" in line:
                next(out)
                next(out)
                next(out)
                next(out)
                line=next(out)
                Sline = line.split()
                dipole = np.array([float(Sline[1]),float(Sline[3]),float(Sline[5])])
                next(out)
                next(out)
                line1=next(out)
                line2=next(out)
                Sline1=line1.split()
                Sline2=line2.split()
                quadrupole = np.array([[float(Sline1[1]),float(Sline1[3]),float(Sline2[1])],
                [float(Sline1[3]),float(Sline1[5]),float(Sline2[3])],
                [float(Sline2[1]),float(Sline2[3]),float(Sline2[5])]])

        # Unit conversion happens once, here; a requested property that is missing or nonfinite fails the label.
        available = dict(energy=None if energy is None else energy*self.E_unit,
                         forces=None if grad is None else -grad*self.E_unit/self.L_unit,
                         dipole=dipole, quadrupole=quadrupole, mulliken_charge=Acharge)
        if any(available.get(key) is None or not np.isfinite(available[key]).all() for key in properties):
            return {'converged': False, 'error': 'Missing or nonfinite requested Q-Chem properties'}

        propertiesout = {'converged': True}
        propertiesout.update({key: available[key] for key in properties})

        self.datacounter += 1

        return propertiesout


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def qchem_dft_calculator_task(molecule_object,ncpu,qchem_env_file,QM_run_command,rem,qchemblocks,QM_scratch_dir,properties_list):
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object.get_moleculeid() + '/'
    Path(directory).mkdir(parents=True,exist_ok=True)

    atoms = molecule_object.get_atoms().copy()

    calc = qchemGenerator(scratch_path=directory, store_path=directory+'/', nproc=ncpu,qchem_env_file=qchem_env_file,qchem_command=QM_run_command,qcheminput=rem,qchemblocks=qchemblocks,unit={'energy': 'ev', 'length': 'angstrom'})

    metadata = molecule_object.get_metadata()
    out_properties = calc.single_point(molecule=atoms, charge=metadata.get('charge',0),
                                       mult=metadata.get('mult',1), prefix='qchem', properties=properties)

    molecule_object.store_results(out_properties)
    molecule_object.set_converged_flag(bool(out_properties.get('converged',False)))
    if not molecule_object.check_convergence():
        molecule_object.update_metadata({'qm_error': out_properties.get('error','Q-Chem failed')})


    return(molecule_object)
