from parsl import python_app, bash_app
import os
import numpy as np
import subprocess
from ase.io import read, write
from alframework.tools.tools import system_checker

class xtbGenerator():
    """
    xtb 6.6.0
    """

    def __init__(self, scratch_path="/tmp/", store_path=None,\
                 unit={'energy': 'hartree', 'length': 'bohr'},\
                 xtb_command="", xtb_input=""):
        """
        unit used for input coordincates: Angstrom
        default units for output: atomic unit
        output parsing is for xtb 6.6.0
        """
        self.scratch_path = scratch_path
        self.store_path = store_path
        self.xtb_command = xtb_command
        self.xtb_input = xtb_input.strip()
        self.unit = unit
        # default a.u., hartree for energy, bohr for length as used for xtb
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
        os.makedirs(self.scratch_path, exist_ok=True)
 
    def write_xtb_input(self, molecule, filename="xtb.xyz"):
        write(filename, molecule, format='xyz')

    def single_point(self, molecule, prefix="xtb", properties=['energy','forces']):
        """
        mol : ase.atoms.Atoms object, will get chemical symbols and positions
        prefix : all the input and output file will start with this prefix, eg. xtb.log, xtb.inp, xtb.engrad
        """
        os.chdir(self.scratch_path)
        filename = os.path.join(self.scratch_path, f"{prefix}.xyz")
        self.write_xtb_input(molecule, filename)
        runcmd = [self.xtb_command, filename, *self.xtb_input.split()]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)

        if self.store_path is not None:
            store_dir = os.path.join(self.store_path, prefix)
            outfile = os.path.join(store_dir, f'{prefix}.out')
            os.makedirs(store_dir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
  
        try:
            with open('gradient', 'r') as f:
                data = f.readlines()
            n_atoms = len(molecule)
            energy = float(data[1].split()[6])
            grad = np.array([[float(i) for i in l.split()] for l in data[n_atoms+2:-1]]).reshape(-1)
        except:
            return {"converged": False}

        outproperties = {"converged": True}
        if 'forces' in properties:
            outproperties['forces'] = -1 * grad * self.E_unit/self.L_unit 
        if 'energy' in properties:
            outproperties['energy'] = energy * self.E_unit
        self.datacounter += 1

        return outproperties

@python_app(executors=['alf_QM_executor'])
def simple_xtb_task(molecule_object, QM_config, QM_scratch_dir, properties_list=['energy','forces']):

    system_checker(molecule_object)

    molecule = molecule_object[1]
    prefix = molecule_object[0]['moleculeid']
    properties = list(properties_list)

    xtb = xtbGenerator(scratch_path=QM_scratch_dir, store_path=QM_config['QM_store_dir'],
                        xtb_command=QM_config['xtb_command'],
                        xtb_input=QM_config['xtb_input'])
    out_properties = xtb.single_point(molecule=molecule, prefix=prefix, properties=properties)
    return_system = [molecule_object[0], molecule_object[1], out_properties]

    system_checker(return_system)
    
    return(return_system)
