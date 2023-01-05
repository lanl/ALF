#TODO: the whole file needs refactoring and cleanup.

# -*- coding: utf-8 -*-
"""
@author: Ziyan
"""

from ase import Atoms, Atom
from ase.calculators.vasp import Vasp

from ase import units

#VASP should handle MPI things
#from mpi4py import MPI

import os
import sys
import re
import shutil
import pickle as pkl
import numpy as np

class SCFConvergenceFailure(Exception):
    pass

class VASPGenerator:

    # Constructor
    # This will need to be more complicated for ad hoc cluster
    def __init__(self, vasp_options, vasp_command, scratch='./', output_store=None,rm_scratch=False):
        # Store variables
        self.vasp_options = vasp_options ###RAM: Ziyan had commented this out. But I think it is useful
        self.vasp_command = vasp_command #Command sequence to run VASP
        self.working_dir = scratch
        self.output_store = output_store

        # Check for existing calculations
        #self.existing_pkls = np.array([f for f in os.listdir(output_store) if f[-2:] == '.p'])

        # Create rank working dir
        if not os.path.isdir(self.working_dir):
            os.mkdir(self.working_dir)

        # Change current dir
        try:
            os.chdir(self.working_dir)
        except NotADirectoryError:
            print('Error: working dir not found.')
            exit(1)
        except PermissionError:
            print('Error: permission error.')
            exit(1)

        # Prepare and save vasp command
        if 'mpirun' not in VASP_COMMAND and 'two-step' not in VASP_COMMAND: ###Accepts two different forms of command
            prepared_vasp_command = "module purge; source /usr/projects/ml4chem/Programs/vasp.6.2.1/vaspExports.bash; /projects/darwin-nv/centos8/x86_64/packages/nvhpc/Linux_x86_64/21.11/comm_libs/mpi/bin/mpirun -n 1 -host " + MPI.Get_processor_name() + ' ' + VASP_COMMAND ###RAM: 1/11
        elif 'two-step' in VASP_COMMAND:
            ###RAM: VASP calculation designed to fail immediately by removing the POTCAR
            ###RAM: Changed to the new VASP compile 11/04/22
            prepared_vasp_command = "module purge; source /usr/projects/t1vasp/vasp.6.3.2/vaspExports.bash; rm POTCAR; sed -i s/'ISPIN = .*'/'ISPIN = 1'/ INCAR; sed -i s/'NELM = .*'/'NELM = 1'/ INCAR; /projects/darwin-nv/rhel8/x86_64/packages/nvhpc/Linux_x86_64/22.9/comm_libs/mpi/bin/mpirun -n 1 -host " + MPI.Get_processor_name() + ' ' + '/usr/projects/t1vasp/vasp.6.3.2/bin/vasp_std'
        elif 'mpirun' == VASP_COMMAND[:6]:
            prepared_vasp_command = "mpirun -host " + MPI.Get_processor_name() + ' ' + VASP_COMMAND[6:] ###RAM: Remove the mpirun portion
        print('Prepared VASP Command:',prepared_vasp_command)
        os.environ['VASP_COMMAND'] = prepared_vasp_command
        os.environ['VASP_PP_PATH'] = VASP_PP_PATH

        # Set cuda device and number of OMP threads
        if gpuid: os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid) ###RAM: Only set if using GPU
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
       
        # Define VASP settings
        self.settings = dict(xc='pbe', prec='Accurate',
        #self.settings = dict(prec='Accurate',
                             ncore=vasp_options['ncore'] if 'ncore' in vasp_options else 1,
                             lreal='Auto',
                             nelm=vasp_options['nelm'] if 'nelm' in vasp_options else 120,
                             ivdw=vasp_options['ivdw'] if 'ivdw' in vasp_options else 0,
                             ispin=vasp_options['ispin']
                             )
        for key, val in vasp_options.items(): 
            if key == 'kpoints':
                self.settings['kpts'] = val
            else:
                self.settings[key] = val
      

   
    def get_magmom(self,atomic_no):

        ###RAM: Allows the user to directly set magmom
        if 'magmom' in self.vasp_options: return self.vasp_options['magmom']

        magmom = atomic_no.copy() 

        return magmom

    #
    # Function for obtaining a single point calculation
    #
    def single_point(self, molecule, force_calculation=False, output_file='output.opt'):

        calc = Vasp(**self.settings) 

        if self.existing_pkls.size > 0:
            compute = np.where(self.existing_pkls == 'data-'+molecule.ids+'.p')[0].size == 0
        else: 
            compute = True

        properties = {}

        if compute:
            # Define ASE Atoms object and set the calculator
            atoms = Atoms(molecule.S, positions=molecule.X, cell=molecule.C, pbc=(True,True,True))
            #if vasp_options['ispin'] == 2:
            atomic_no = np.array(atoms.get_atomic_numbers())

            atoms.set_calculator(calc)

            try:
                # Attempt to compute energies
                properties['energy'] = (1.0/units.Hartree)*atoms.get_potential_energy()
                
                # Attempt to compute stress tensor: (xx, yy, zz, yz, xz, xy) or a 3x3 matrix
                properties['stress'] = atoms.get_stress(voigt=False)

                # Attempt to compute magnetic moment
                #properties['magnetic_moment'] = atoms.get_magnetic_moment()

                # Attempt to compute forces
                if force_calculation:
                    properties['forces'] = (1.0/units.Hartree)*atoms.get_forces()

                #print('SCF CHECK:',atoms.calc.converged)
                if not atoms.calc.converged:
                    raise SCFConvergenceFailure()
        
                # Define new system
                molecule = Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids,C=molecule.C)
                
                # Move the success OUTCAR
                output_data = open(self.working_dir+"/"+"OUTCAR", 'r').read()
                output_file = open(self.output_store+"data-"+molecule.ids+'.OUTCAR',"w")
                output_file.write(output_data)
                output_file.close()
                
                output_data = open(self.working_dir+"/"+"POSCAR", 'r').read()
                output_file = open(self.output_store+"data-"+molecule.ids+'.POSCAR',"w")
                output_file.write(output_data)
                output_file.close()
                
                output_data = open(self.working_dir+"/"+"CONTCAR", 'r').read()
                output_file = open(self.output_store+"data-"+molecule.ids+'.CONTCAR',"w")
                output_file.write(output_data)
                output_file.close()
                
                # Store the calculation in pkl format
                pkl.dump( {"molec":molecule,"props":properties}, open( self.output_store+"/data-"+molecule.ids+'.p', "wb" ) )

                # Remove working files
                for f in os.listdir(self.working_dir):
                    os.remove(self.working_dir+'/'+f)
                
                # Return data
                return molecule, properties
            except:
                # Obtain and print error out
                e = sys.exc_info()[0]
                print('!!ERROR!!:','data-'+molecule.ids+'.failed-OUTCAR\n',e)

                # Move the failed OUTCAR
                output_data = open(self.working_dir+"/"+"OUTCAR", 'r').read()    
                output_file = open(self.output_store+"data-"+molecule.ids+'.failed-OUTCAR',"w")
                output_file.write(output_data)
                output_file.close()
                 
                output_data = open(self.working_dir+"/"+"POSCAR", 'r').read()
                output_file = open(self.output_store+"data-"+molecule.ids+'.failed-POSCAR',"w")
                output_file.write(output_data)
                output_file.close()
                
                # Remove working files
                for f in os.listdir(self.working_dir):
                    os.remove(self.working_dir+'/'+f)

                # Return failed data
                return Molecule(np.array(atoms.get_positions()),molecule.S,molecule.Q,molecule.M,molecule.ids,C=molecule.C,failed=True), properties
        else:
            loaded_data = pkl.load( open( self.output_store+"/data-"+molecule.ids+'.p', "rb" ) )
            print("DATA LOADED FROM FILE:",molecule.ids)
            return loaded_data["molec"], loaded_data["props"]
            
        
