import os
import glob
import numpy as np
import json
from ase.geometry import complete_cell
from ase import Atoms
from alframework.tools import pyanitools as pyt

def annealing_schedule(t, tmax, amp, per, srt, end):
    linear = t / tmax
    linearT = (1 - linear) * srt + linear * end
    return (amp) * np.power(np.sin(np.pi * t / per), 2) + linearT
    
#This function takes a directory and configures a dictionary for passing into ase_interface import ensemblemolecule
def build_ANI_info(directory):
    #TODO: Check that directory is a valid directory
    ani_dict={}
    cnst_files = glob.glob(directory + '/*.params')
    assert len(cnst_files) == 1,"Too many or too few params files detected: " + str(cnst_files)
    ani_dict['cnstfile'] = cnst_files[0]
    ani_dict['saefile'] = directory + '/sae_linfit.dat'
    ani_dict['nnfprefix'] = directory + '/train'
    model_dirs = glob.glob(ani_dict['nnfprefix'] + '*')
    ani_dict['Nnet'] = len(model_dirs)
    return(ani_dict)

def compute_empirical_formula(S):
    uniques = np.unique(S,return_counts=True)
    arg_sort = np.argsort(uniques[0])
    return "_".join([i+str(j).zfill(2) for i,j in zip(uniques[0][arg_sort],uniques[1][arg_sort])])

def random_rotation_matrix(deflection=1.0, randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def store_current_data(h5path, system_data, properties):
#    system data is a list of [mol-id(string), atoms, properties dictionary]
    data_dict = {}
    print("Saving h5 file: " + h5path)
    totalNumber = len(system_data)
    savedNumber = 0
    nanNumber = 0
    convergedNumber = 0
    for system in system_data:
        # We can append the system data to an existing set of system data
        cur_moliculeid = system[0]['moleculeid']
        cur_atoms = system[1]
        cur_properties = system[2]
        molkey = compute_empirical_formula(cur_atoms.get_chemical_symbols())
        #Ensure no nan data before saving
        
        #Ensure system converged before saving 
        if cur_properties['converged'] and system_checker(system,kill_on_fail=False):
            savedNumber = savedNumber + 1
            #The 
            atom_index = np.argsort(cur_atoms.get_atomic_numbers())
            #If there is already a molecule with the same formula, append
            if molkey in data_dict:
                data_dict[molkey]["_id"].append(cur_moliculeid)
                data_dict[molkey]["coordinates"].append(cur_atoms.get_positions()[atom_index])
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"].append(complete_cell(cur_atoms.get_cell()))
                for  prop in properties:
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]].append(cur_properties[prop]*properties[prop][2])
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]].append(np.array(cur_properties[prop])[atom_index]*properties[prop][2])
                    else:
                        raise RuntimeError('Unknown property format')
            #if there is not already a molecule with this empirical formula, make a new one
            else:
                data_dict[molkey] = {}
                data_dict[molkey]["species"] = np.array(cur_atoms.get_chemical_symbols())[atom_index]
                data_dict[molkey]["_id"]=[cur_moliculeid]
                data_dict[molkey]["coordinates"]=[cur_atoms.get_positions()[atom_index]]
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"]=[complete_cell(cur_atoms.get_cell())]
                for  prop in properties:
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]]=[cur_properties[prop]*properties[prop][2]]
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]]=[np.array(cur_properties[prop])[atom_index]*properties[prop][2]]
                    else:
                        raise RuntimeError('Unknown property format')
        elif not(system_checker(system,kill_on_fail=False)):
            nanNumber = nanNumber + 1
        elif not(cur_properties['converged']):
            convergedNumber = convergedNumber + 1
        else:
            print("Warning: molecule not saved for unknown reason")
    print("Total Systems: " + str(totalNumber))
    print("Saved Systems: " + str(savedNumber))
    print("NAN Systems: " + str(nanNumber))
    print("Unconverged Systems: " + str(convergedNumber))

    for isokey in data_dict:
        #print('isokeys:',isokey)
        for propkey in data_dict[isokey]:
            if propkey.lower() in ['species','_id']:
                data_dict[isokey][propkey] = [el.encode('utf-8') for el in list(data_dict[isokey][propkey])]
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
            else:
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
                #print("encoding species")
#                    if type(data_dict[isokey][propkey]) is 'numpy.ndarray':
#                        data_dict[isokey][propkey] = np.stack(data_dict[isokey][propkey])
#                    else:
#                        data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
#                    print('propkey:', propkey,data_dict[isokey][propkey].shape)
#
    dpack = pyt.datapacker(h5path)
    for key in data_dict:
        dpack.store_data(key,**data_dict[key])
    dpack.cleanup()



#Recommend creation of parsl queue object
class parsl_task_queue():
    def __init__(self):
        #Create a list 
        self.task_list = []
    
    def add_task(self,task):
        self.task_list.append(task)
        #self.task_list[-1].start()
        
    def get_completed_number(self):
        task_status = [task.done() for task in self.task_list]
        return(int(np.sum(task_status)))
        
    def get_running_number(self):
        task_status = [task.running() for task in self.task_list]
        return(int(np.sum(task_status)))
        
    def get_number(self):
        return(len(self.task_list))
    
    def get_queued_number(self):
        return(int(self.get_number()-self.get_running_number()-self.get_completed_number()))
            
    def get_task_results(self):
        results_list = []
        failed_number = 0
        for taski,task in enumerate(self.task_list):
            task_status = task.task_status()
            if task_status == 'exec_done' and task.done:
                results_list.append(task.result())
                del self.task_list[taski]
            elif task_status == 'failed':
                failed_number=failed_number+1
                del self.task_list[taski]
        return(results_list,failed_number)
    
    def get_task_status(self):
        status_list = []
        for task in self.task_list:
            status_list.append(task.task_status())
        return(status_list)
        
    def print_status(self):
       print('Total Tasks: {:d}'.format(self.get_number()))
       #print('Queued Tasks: {:d}'.format(self.get_queued_number()))
       #print('Running Tasks: {:d}'.format(self.get_running_number()))
       print('Finished Tasks: {:d}'.format(self.get_completed_number()))
       
#used to find current version of directories to re-start with
def find_empty_directory(pattern,limit=9999):
    curI = 0
    while os.path.exists(pattern.format(curI)):
        curI=curI+1
    return(curI)
    
#Throughout this code individual systems are passed around as three element lists
#element 1: metadata: this is required to include moleculeid,  but may also include sampling and other metadata
#element 2: an ASE atoms object. 
#element 3: Evaluated QM properties
def system_checker(system,kill_on_fail=True):
    try: 
        assert isinstance(system,list) or isinstance(system,tuple)
        assert len(system) == 3
        assert isinstance(system[0],dict)
        assert isinstance(system[0]['moleculeid'],str)
        assert isinstance(system[1],Atoms)
        assert isinstance(system[2],dict)
        
        no_nan = True
        if np.sum(np.isnan(system[1].get_positions())) > 0:
            no_nan = False
        if any(system[1].get_pbc()):
            if np.sum(np.isnan(system[1].get_cell())) > 0:
                no_nan = False
        for prop in system[2]:
            if isinstance(system[2][prop],np.ndarray):
                if np.sum(np.isnan(system[2][prop]))>0:
                    no_nan = False
        if not(no_nan):
            raise RuntimeError('NAN in system')
        
        return(True)
        
    except Exception as e:
        print(e)
        if kill_on_fail:
            raise RuntimeError('Atomic system failed to meet requirements.')
        return(False)
	          
def load_config_file(path,master_directory=None):
    with open(path,'r') as input_file:
        config = json.load(input_file)
    if master_directory is None and "master_directory" in config:
        if config["master_directory"] == 'pwd':
            master_directory = os.getcwd() + '/'
        else: 
            master_directory = config["master_directory"] + '/' 
        config["master_directory"] = master_directory
    if isinstance(config, dict):
        dir_dict = {}
        for entry in config:
            if (entry[-3:].lower() == 'dir') and config[entry][0] != '/':
                config[entry] = master_directory + config[entry]
            elif (entry[-4:].lower() == 'path'):
                if config[entry][0] != '/':
                    config[entry] = master_directory + config[entry]
                #For every 'path' entry, make a corresponding 'dir' entry that holds files in the path
                dir_dict[entry[:-4]+'dir'] = '/'.join(config[entry].split('/')[:-1]) + '/'
        config.update(dir_dict)
    return(config)
    