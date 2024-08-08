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
from alframework.samplers.builders import condensed_phase_builder
import random
from copy import deepcopy

import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem

#def ASEfromSMILES(SMILES):
#    ps = AllChem.ETKDGv2()
#    ps.useRandomCoords = True 
#    m = Chem.MolFromSmiles(SMILES)
#    m = Chem.AddHs(m)
#    specString=''
#    for a in m.GetAtoms():
#        specString = specString + chemical_symbols[a.GetAtomicNum()]
#    AllChem.EmbedMolecule(m,ps)
#    positions = m.GetConformer().GetPositions()
#    aseAtoms = Atoms(specString,positions=positions)
#    return(aseAtoms)

def ASEfromSMILES(SMILES,elongate=False,maxD=300,dD=10):
    #elongate can either be False, True, or number indicies
    m = Chem.MolFromSmiles(SMILES)
    m = Chem.AddHs(m)
    mLen = len(m.GetAtoms())
    specString=''
    for a in m.GetAtoms():
        specString = specString + chemical_symbols[a.GetAtomicNum()]
    AllChem.EmbedMolecule(m,useRandomCoords=True,ETversion=2,maxAttempts=5)
    positions = m.GetConformer().GetPositions()
    if elongate!=False:
        try: 
            for curD in np.arange(dD,maxD,dD):
                coordMap={0:Geometry.Point3D(0,0,0),mLen-1:Geometry.Point3D(float(curD),0,0)}
                AllChem.EmbedMolecule(m,ETversion=2,coordMap=coordMap)
                positions = m.GetConformer().GetPositions()
        except:
            print("Failed at {:.1f} distance".format(float(curD)))
    aseAtoms = Atoms(specString,positions=positions)
    return(aseAtoms)

@python_app(executors=['alf_sampler_executor'])
def rdkit_condensed_phase_builder_task(moleculeid,builder_config,cell_range,Rrange,solute_molecule_options,solvent_molecules):
    """
    Elements in  builder parameters
        molecule_library_path: path to library of molecular fragments to read in
        solute_molecule_options: listof lists detailing sets of solutes
        solvent_molecules: list or dictionary of solvent molecules. If dictionary, corresponding value is relative weight of solvent
        cell_range: 3X2 list with x, y, and z ranges for cell size 
        Rrange: density range
        min_dist: minimum contact distance between fragments
        max_patience: How many attempts to  make before giving up on build
        center_first_molecule: Boolian,  if true first solute is centered in box and not rotated (useful for large molecules)
        shake: Distance to displace initial configurations
        print_attempt: Boolian,controls printing (set to False)
    """
    
    cell_shape = [np.random.uniform(dim[0],dim[1]) for dim in cell_range]
    
    empty_system = [{'moleculeid':moleculeid},Atoms(cell=cell_shape,pbc=True),{}]
        
    solute_molecules = random.choice(solute_molecule_options)
    
    molecule_library = {}
    for curSMILES in (solute_molecules + solvent_molecules):
        molecule_library[curSMILES] = ASEfromSMILES(curSMILES)

    feed_parameters = {}
    
    feed_parameters['solute_molecules'] = solute_molecules
    feed_parameters['density'] = np.random.uniform(Rrange[0],Rrange[1])
    
    input_parameters = build_input_dict(condensed_phase_builder,[{"start_system":empty_system,"molecule_library":molecule_library,"solute_molecules":solute_molecules,"density":np.random.uniform(Rrange[0],Rrange[1])},builder_config])
    system = condensed_phase_builder(**input_parameters)
    system_checker(system)
    return(system)
    
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
        filename = os.path.join(job_path, f"{prefix}.in")
        os.makedirs(job_path, exist_ok=True)
        self.write_qchem_input(molecule, charge, mult, filename=filename)
        
        if self.qchem_env_fn != None:
            runcmd = 'source ' + self.qchem_env_fn + '; '
        else:
            runcmd = ''
        runcmd = runcmd + self.qchem_command + ' -nt ' + str(self.nproc) + ' ' + filename
        print(runcmd)
        print(os.environ['QCSCRATCH'])
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, shell=True)
        print(proc.stdout)
        print(proc.stderr)

        if self.store_path is not None:
            store_dir = os.path.join(self.store_path, prefix)
            outfile = os.path.join(store_dir, f'{prefix}.out')
            os.makedirs(store_dir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(proc.stdout)

        energy = None
        for line in proc.stdout.split("\n"):
            if "RIMP2         total energy =" in line:
                energy = float(line.split()[-2])

        natoms = len(molecule)
        nblocks = natoms // 5
        if natoms%5 > 0: nblocks += 1
        grad = None
        out = iter(proc.stdout.split("\n"))
        for line in out:
            if "Gradient of MP2 Energy" in line:
                grad = np.zeros((natoms, 3))
                for block in range(nblocks):
                    next(out)
                    for i in range(3):
                        line = next(out)[5:].rstrip()
                        grad_i = np.array([float(line[j:j+14]) for j in range(0, len(line), 14)])
                        grad[5*block:5*block+len(grad_i), i] = grad_i
                break

        if energy is None or grad is None:
            return {'converged': False}

        propertiesout = {'converged': True}
        if 'forces' in properties:
            propertiesout['forces'] = -grad*self.E_unit/self.L_unit
        if 'energy' in properties:
            propertiesout['energy'] = energy*self.E_unit

        self.datacounter += 1

        return propertiesout
        
@python_app(executors=['alf_QM_executor'])
def qchem_mp2_calculator_task(molecule_object,QM_store_dir,ncpu,qchem_env_file,QM_run_command,rem,qchemblocks,QM_scratch_dir,properties_list):
    system_checker(molecule_object)
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid'] + '/'
    Path(directory).mkdir(parents=True,exist_ok=True)
    
    molecule_id = molecule_object[0]['moleculeid']
    atoms = molecule_object[1]
    
    calc = qchemGenerator(scratch_path=directory, store_path=QM_store_dir, nproc=ncpu,qchem_env_file=qchem_env_file,qchem_command=QM_run_command,qcheminput=rem,qchemblocks=qchemblocks,unit={'energy': 'ev', 'length': 'angstrom'})
    
    out_properties = calc.single_point(molecule=atoms, charge=0, mult=1, prefix=molecule_id, properties=properties)
        
    return_system = [molecule_object[0],molecule_object[1],out_properties]
    system_checker(return_system)
    
    return(return_system)