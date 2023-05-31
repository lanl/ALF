from parsl import python_app, bash_app
import os
import numpy as np
import json
from ase.atoms import Atoms
import ase.io
import shutil
import re

from alframework.tools.tools import system_checker

class orcaGenerator():
    """
    ORCA 5.0.1
    """

    def __init__(self,scratch_path="/tmp/", store_path=None, nproc=36,\
                 unit={'energy': 'hartree', 'length': 'bohr'},\
                 orca_env_file=None,orca_command="",orcainput="",orcablocks=""):
        """
        unit used for input coordincates: Angstrom
        default units for output: atomic unit
        output parsing is for orca 5
        """
        self.scratch_path = scratch_path
        self.store_path = store_path
        self.nproc = nproc
        self.orca_env_fn = orca_env_file
        self.orca_command = orca_command
        self.orcainput = orcainput.strip()
        self.orcablocks = orcablocks if orcablocks.endswith('\n') else orcablocks+'\n'
        self.unit = unit
        #self.E_unit = 1.0  # default a.u., hartree for energy, bohr for length as used for orca
        #self.L_unit = 1.0
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
    
    def write_orca_input(self, ase_atoms, charge, multiplicity, job_path, filename="orca.inp"):

        with open(job_path+filename,"w") as f:
            f.write("! "+ self.orcainput + "\n")
            if not(self.nproc == None) and not(self.nproc == 1):
                f.write("%pal nproc " + str(self.nproc) + " end\n")
            f.write(self.orcablocks)
            f.write("* xyzfile %d %d input.xyz\n" % (charge, multiplicity))
        ase.io.write(job_path+"input.xyz", ase_atoms,format='xyz')
    
    def single_point(self, molecule, prefix="orca",properties=['energy','forces']):
        """
        mol : ase.atoms.Atoms object, will get chemical symbols and positions
        prefix : all the input and output file will start with this prefix, eg. orca.log, orca.inp, orca.engrad
        unlike other generators, the choice of calculations will be performed is setup in __init__: orcainput
        """
        #logF = open('orca_run.log','w')
        #assert molecule.periodic() == False, 'ORCA does not support periodic boundary condition'
        job_path = self.scratch_path 
        os.makedirs(job_path,exist_ok=True)
            
        #logF.write('directory made\n')
        # prepare orca input
        self.write_orca_input(molecule, 0, 1, job_path, filename=prefix+".inp")
        
        #logF.write('input printed\n')

        # run job
        if self.orca_env_fn is None:
            #logF.write("cd " + job_path +" ; " + self.orca_command + " " + prefix + ".inp > " + prefix + ".log\n")
            os.system("cd " + job_path +" ; " + self.orca_command + " " + prefix + ".inp > " + prefix + ".log")
        else:
            #logF.write("source " + self.orca_env_fn + " ; cd " + job_path +" ; " + self.orca_command + " " + prefix + ".inp > " + prefix + ".log\n")
            os.system("source " + self.orca_env_fn + " ; cd " + job_path +" ; " + self.orca_command + " " + prefix + ".inp > " + prefix + ".log")
        #logF.close()
        # parse output, force and energy
        if self.check_normal_termination( job_path + prefix + ".log"):
            n_atom = len(molecule)
            propertiesout = self.parse_output(job_path, prefix, n_atom,properties)
            propertiesout['converged'] = True
        else:
            propertiesout = {"converged":False}
            
        # save files and clean up
        if not(self.store_path == None):
            os.system("cd " + self.scratch_path + " && tar -cvf " + self.store_path + "/orca_job_" + str(molecule.ids) + ".tar orca_job_" + str(molecule.ids))
            shutil.rmtree(job_path)
        
        self.datacounter += 1
        return propertiesout

    def check_normal_termination(self, logfile):
        d = open(logfile).read()
        return "TOTAL RUN TIME: " in d  and "SCF CONVERGED AFTER" in d
    
    def parse_output(self,job_path, prefix, natom, properties, sign=-1):
        # sign: -1: force, +1, gradient
        outengrad = open(job_path + prefix + '.engrad','r').read()
        outlog = open(job_path + prefix + '.log','r').read()
        outproperty = open(job_path + prefix + '_property.txt','r').read()
        
        reTOTALenergy = re.compile("The current total energy in Eh\n#\n([\s\S]+?)#")
        reSCFenergy = re.compile("SCF Energy:([\s\S]+?)\n")
        reCORRenergy = re.compile("Correlation Energy:([\s\S]+?)\n")
        reEnergy = re.compile("Total[\s\S]+?Energy:([\s\S]+?)\n")
        reDipole = re.compile("Electric_Properties\n[\s\S]+?Total Dipole moment:\n([\s\S]+?)---------------------")
        reQuad = re.compile("Electric_Properties\n[\s\S]+?Total quadrupole moment\n([\s\S]+?)# --------------")
        reGrad = re.compile('The current gradient in Eh\/bohr\n#\n([\s\S]+?)#')
        reHirshTrue = re.compile("HIRSHFELD ANALYSIS")
        reHirsh = re.compile("HIRSHFELD ANALYSIS\n[\s\S]+?SPIN  \n([\s\S]+?)TOTAL")
        
        #Always pull in energies and forces from engrad
        outproperties = {}
        if 'forces' in properties:
            outproperties['forces'] = np.array([float(i) for i in reGrad.findall(outengrad)[-1].split("\n")[:natom*3]]).reshape([natom,3]) * sign * self.E_unit/self.L_unit
        if 'energy' in properties:
            outproperties['energy'] = float(reTOTALenergy.findall(outengrad)[-1]) * self.E_unit
        if 'SCF_energy' in properties:
            outproperties['SCF_energy'] = float(reEnergy.findall(outproperty)[-1]) * self.E_unit
        if 'CORR_energy' in properties:
            outproperties['CORR_energy'] = float(reEnergy.findall(outproperty)[-1]) * self.E_unit
        if 'dipole' in properties:
            outproperties['dipole'] = np.array([float(i.split()[1]) for i in reDipole.findall(outproperty)[-1].split("\n")[1:4]])
        if 'quadrupole' in properties:
            outproperties['quadrupole'] = np.array([[float(j) for j in i.split()[1:4]] for i in reQuad.findall(outproperty)[-1].split("\n")[1:4]])
        if 'hirshfeld' in properties:
            outproperties['hirshfeld'] = np.array([float(i.split()[2]) for i in reHirsh.findall(outlog)[-1].split("\n")[:natom]])
        if 'hirshfeld_spin' in properties:
            outproperties['hirshfeld_spin'] = np.array([float(i.split()[3]) for i in reHirsh.findall(outlog)[-1].split("\n")[:natom]])
        return outproperties
    
#    def parse_output(self, job_path, prefix, natom, sign=-1):
#        # sign: -1: force, +1, gradient
#        fn = job_path + prefix + ".engrad"
#        energy = float(os.popen("grep energy %s -A 2 | tail -n 1" % fn).read().strip().split()[0])*self.E_unit
#        forces = sign*np.asarray([float(x.strip().split()[0]) for x in os.popen("grep gradient %s -A %d | tail -n %d" \
#            % (fn, natom*3+1, natom*3)).read().split("\n")[:-1]]).reshape(-1,3)*self.E_unit/self.L_unit
#        fn = job_path + prefix + ".log"
#        mulliken = np.asarray( [ float(x.strip().split()[-1]) 
#                       for x in os.popen("grep 'MULLIKEN ATOMIC CHARGES' %s -A %d | tail -n %d" 
#                       % (fn, natom+1, natom)).read().split("\n")[:-1]])
#        fn = job_path + prefix + "_property.txt"
#        dipole = np.asarray( [ float(x.strip().split()[-1]) for x in \
#            os.popen("grep 'Total Dipole moment:' %s -A 4 | tail -n 3" % fn).read().strip().split('\n')])*self.L_unit
#            
#        
#            
#        return_dictionary = {"energy":energy,"forces":forces,"dipole":dipole,"mulliken":mulliken}
#        return return_dictionary


@python_app(executors=['alf_QM_executor'])
def orca_calculator_task(molecule_object,QM_config,QM_scratch_dir,properties_list):
    system_checker(molecule_object)
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    molecule_id = molecule_object[0]['moleculeid']
    atoms = molecule_object[1]
    
    orca = orcaGenerator(scratch_path=directory,nproc=QM_config['ncpu'],orca_env_file=QM_config['orca_env_file'],orca_command=QM_config['QM_run_command'],orcainput=QM_config['orcasimpleinput'],orcablocks=QM_config['orcablocks'])
    
    out_properties = orca.single_point(molecule=atoms,properties=properties)
    
    return_system = [molecule_object[0],molecule_object[1],out_properties]
    system_checker(return_system)
    
    return(return_system)


@python_app(executors=['alf_QM_executor'])
def orca_double_calculator_task(molecule_object,QM_config,QM_scratch_dir,properties_list):
    system_checker(molecule_object)
    properties = list(properties_list)
    directory = QM_scratch_dir + '/' + molecule_object[0]['moleculeid']
    molecule_id = molecule_object[0]['moleculeid']
    atoms = molecule_object[1]
    directory1 = directory + '/1/'
    directory2 = directory + '/2/'
    
    orca1 = orcaGenerator(scratch_path=directory1,nproc=QM_config['ncpu'],orca_env_file=QM_config['orca_env_file'],orca_command=QM_config['QM_run_command'],orcainput=QM_config['orcasimpleinput'],orcablocks=QM_config['orcablocks'])
    
    properties1 = orca1.single_point(molecule=atoms,properties=properties)
    
    orca2 = orcaGenerator(scratch_path=directory2,nproc=QM_config['ncpu'],orca_env_file=QM_config['orca_env_file'],orca_command=QM_config['QM_run_command'],orcainput=QM_config['orcasimpleinput'],orcablocks=QM_config['orcablocks'])
    
    properties2 =  orca2.single_point(molecule=atoms,properties=properties)
    
    maxEdev = np.abs(properties1['energy'] - properties2['energy'])
    maxFdev = np.max(np.abs(properties1['forces']-properties2['forces']))
    
    properties = {}
    for key in list(properties1):
        properties[key] = np.mean([properties1[key],properties2[key]],axis=0)
    
    if (maxEdev < configuration['Ediff']) and (maxFdev < QM_config['Fdiff']) and properties1['converged'] and properties2['converged']:
        properties['converged'] = True
    else:
        properties['converged'] = False
    
    return_system = [molecule_object[0],molecule_object[1],properties]
    system_checker(return_system)
    
    return(return_system)
    