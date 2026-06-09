# The builder code goes here
import glob
import random
import os
import numpy as np
from parsl import python_app, bash_app
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
        if not os.environ.get('QCSCRATCH'):
            os.environ['QCSCRATCH'] = scratch_path   
 
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

        job_path = os.path.join(self.scratch_path, prefix)
        print(f"job_path:, {job_path}")


        filename = os.path.join(job_path, f"{prefix}.in")
        os.makedirs(job_path, exist_ok=True)
        self.write_qchem_input(molecule, charge, mult, filename=filename)
        
        if self.qchem_env_fn != None:
            runcmd = 'source ' + self.qchem_env_fn + '; '
        else:
            runcmd = ''

        runcmd = runcmd + ' ' + self.qchem_command + ' ' + self.scratch_path

        if self.store_path is not None:
            store_dir = os.path.join(self.store_path, prefix)
            outfile = os.path.join(store_dir, f'{prefix}.out')
            #os.makedirs(store_dir, exist_ok=True)
            Path(store_dir).mkdir(exist_ok=True,parents=True)
            with open(outfile, 'w') as f:
                f.write(proc.stdout)

        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set = " in line:
                energy = float(line.split()[-1])

        natoms = len(molecule)
        nblocks = natoms // 6
        if natoms%6 > 0: nblocks += 1
        grad = None
        Acharge = np.zeros(natoms)
        out = iter(proc.stdout.split("\n"))
        for line in out:
            if "Gradient of SCF Energy" in line:
                grad = np.zeros((natoms, 3))
                for block in range(nblocks):
                    next(out)
                    for i in range(3):
                        line = next(out)[5:].rstrip()
                        grad_i = np.array([float(line[j:j+12]) for j in range(0, len(line), 12)])
                        grad[6*block:6*block+len(grad_i), i] = grad_i
            elif "Ground-State Mulliken Net Atomic Charges" in line:
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

        if energy is None or grad is None:
            return {'converged': False}

        propertiesout = {'converged': True}
        if 'forces' in properties:
            propertiesout['forces'] = -grad*self.E_unit/self.L_unit
        if 'energy' in properties:
            propertiesout['energy'] = energy*self.E_unit
        if 'dipole' in properties:
            propertiesout['dipole'] = dipole
        if 'quadrupole' in properties:
            propertiesout['quadrupole'] = quadrupole
        if 'mulliken_charge' in properties:
            propertiesout['mulliken_charge'] = Acharge

        self.datacounter += 1

        return propertiesout
 
        
@python_app(executors=['alf_QM_executor'])
def qchem_dft_calculator_task(molecule_object,ncpu,qchem_env_file,QM_run_command,rem,qchemblocks,QM_scratch_dir,properties_list):
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object.get_moleculeid() + '/'
    Path(directory).mkdir(parents=True,exist_ok=True)
    
    molecule_id = molecule_object.get_moleculeid
    atoms = molecule_object.get_atoms().copy()

    calc = qchemGenerator(scratch_path=directory, store_path=directory+'/', nproc=ncpu,qchem_env_file=qchem_env_file,qchem_command=QM_run_command,qcheminput=rem,qchemblocks=qchemblocks,unit={'energy': 'ev', 'length': 'angstrom'})
    
    out_properties = calc.single_point(molecule=atoms, charge=0, mult=1, prefix='qchem', properties=properties)

    molecule_object.store_results(out_properties)
    if 'forces' in out_properties:
        molecule_object.set_converged_flag(True)
    else: 
        molecule_object.set_converged_flag(False)
        
    
    return(molecule_object)


