from parsl import python_app, bash_app
import os
import numpy as np
import subprocess

from alframework.tools.tools import system_checker

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

        runcmd = [self.qchem_command, '-nt', str(self.nproc), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)

        if self.store_path is not None:
            store_dir = os.path.join(self.store_path, prefix)
            outfile = os.path.join(store_dir, f'{prefix}.out')
            os.makedirs(store_dir, exist_ok=True)
            with open(outfile, 'w') as f:
                f.write(proc.stdout)

        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set" in line:
                energy = float(line.split()[-1])

        natoms = len(molecule)
        nblocks = natoms // 6
        if natoms%6 > 0: nblocks += 1
        grad = None
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
def simple_qchem_task(molecule_object, QM_config, QM_scratch_dir, properties_list):

    system_checker(molecule_object)

    molecule = molecule_object[1]
    charge = molecule_object[0].get('charge', 0)
    mult = molecule_object[0].get('multiplicity', 1)
    prefix = molecule_object[0]['moleculeid']
    properties = list(properties_list)

    calc = qchemGenerator(scratch_path=QM_scratch_dir, store_path=QM_config['QM_store_dir'], nproc=QM_config['ncpu'],
                          qchem_env_file=None, qchem_command=QM_config['QM_run_command'], 
                          qcheminput=QM_config['rem'], qchemblocks=QM_config['qchemblocks'])
    out_properties = calc.single_point(molecule=molecule, charge=charge, mult=mult, prefix=prefix, properties=properties)
    return_system = [molecule_object[0],molecule_object[1],out_properties]

    system_checker(return_system)

    return(return_system)
