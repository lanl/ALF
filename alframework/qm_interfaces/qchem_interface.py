import numpy as np
import subprocess
from tempfile import NamedTemporaryFile
from ase.units import Bohr


class QChemRunner(object):

    def __init__(self, atoms, rem, charge=0, mult=1, nt=12):

        self.atoms = atoms
        self.numbers = atoms.get_atomic_numbers()
        tf = NamedTemporaryFile(mode='w', suffix='.qcin', delete=False)
        self.filename = tf.name
        tf.close()
        
        self.rem = rem
        self.charge = charge
        self.mult = mult
        self.nt = nt

    def write(self, positions):

        positions = positions.reshape(-1, 3)

        with open(self.filename, 'w') as f:
            f.write(f"$molecule\n{self.charge} {self.mult}\n")
            [f.write("{} {} {} {}\n".format(ix, *ixyz)) for ix, ixyz in zip(self.numbers, positions)]
            f.write("$end\n\n$rem\n")
            [f.write(f"{k} {v}\n") for k,v in self.rem.items()]
            f.write("$end\n")

        return self.filename
  
    def energy(self, positions):

        filename = self.write(positions)
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        #print(proc.stdout)
        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set" in line:
                energy = float(line.split()[-1])
        return energy

    def grad(self, positions):

        filename = self.write(positions)
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        #print(proc.stdout)
        energy = None
        for line in proc.stdout.split("\n"):
            if "Total energy in the final basis set" in line:
                energy = float(line.split()[-1])

        natoms = len(self.atoms)
        nblocks = natoms // 6
        if nblocks%6 > 0: nblocks += 1
        grad = np.zeros((natoms, 3))
        out = iter(proc.stdout.split("\n"))
        for line in out:
            if "Gradient of SCF Energy" in line:
                for block in range(nblocks):
                    next(out)
                    for i in range(3):
                        line = next(out)[5:].rstrip()
                        grad_i = np.array([float(line[j:j+12]) for j in range(0, len(line), 12)])
                        grad[6*block:6*block+len(grad_i), i] = grad_i
                grad = (grad/Bohr).reshape(-1)
                break

        return energy, grad
    
    def hess(self, positions):
        self.rem['vibman_print'] = '4'
        filename = self.write(positions)
        runcmd = ['qchem', '-nt', str(self.nt), filename]
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        data = proc.stdout.split("\n")
        out = iter(data)

        natoms = len(self.atoms)
        nblocks = 3*natoms // 6
        if (3*natoms)%6 > 0: nblocks += 1
        hess = np.zeros((3*natoms, 3*natoms))
        for line in out:
            if "Mass-Weighted Hessian Matrix:" in line:
                for block in range(nblocks):
                    next(out)
                    next(out)
                    for i in range(3*natoms):
                        hline = next(out)[4:].rstrip()
                        hess_i = np.array([float(hline[j:j+12]) for j in range(0, len(hline), 12)])
                        hess[i, 6*block:6*block+len(hess_i)] = hess_i
                break
        return hess
    
    def __call__(self, positions):
        return self.grad(positions)

@python_app(executors=['alf_QM_executor'])
def simple_qchem_task(molecule_object,QM_config,QM_scratch_dir,properties_list):
    system_checker(molecule_object)
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    molecule_id = molecule_object[0]['moleculeid']
    atoms = molecule_object[1]
    charge = molecule_object[0].get('charge',0)
    mult = molecule_object[0].get('multiplicity',1)
    
    calc = QChemRunner(atoms,QM_config['rem'],charge=charge,mult=mult,nt=QM_config.get('nt',1))
    
    if 'forces' in properties:
        energy, grad = QChemRunner.grad(atoms.get_positions())
        forces = -1*grad
        return_system = [molecule_object[0],molecule_object[1],{'energy':energy,'forces':forces}]
    else:
        energy = QChemRunner.energy(atoms.get_positions())
        return_system = [molecule_object[0],molecule_object[1],{'energy':energy}]
        	
    system_checker(return_system)
    
    return(return_system)
    