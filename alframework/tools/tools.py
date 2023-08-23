import glob
import inspect
import json
import os
from importlib import import_module

import numpy as np
from ase import Atoms
from ase.geometry import complete_cell

from alframework.tools import pyanitools as pyt
from alframework.tools.database import Database as ZarrDatabase
from alframework.tools.molecules_class import MoleculesObject


def annealing_schedule(t, tmax, amp, per, srt, end):
    """Defines the overall temperature profile in the molecular dynamics simulation.

    Args:
        t (float): Current simulation time.
        tmax (float): Total simulation time.
        amp (float): Amplitude of the temperature oscillation.
        per (float): Period of the temperature oscillation.
        srt (float): Initial temperature.
        end (float): Final temperature.

    Returns:
        (float): Temperature at step 't' given the specified annealing schedule.

    """
    linear = t / tmax
    linear_T = (1 - linear) * srt + linear * end

    return amp * np.sin(np.pi * t / per) ** 2 + linear_T


def build_ANI_info(directory):
    """Scrapes directory and create a dictionary containing the information that can be passed to 'ensemblemolecule'.

    Args:
        directory (str): Directory to look for the information needed to fill the dict.

    Returns:
        (dict): Dictionary that can be passed into 'ensemblemolecule' in ase_interface module.

    """
    # TODO: Check that directory is a valid directory
    ani_dict = {}
    cnst_files = glob.glob(directory + '/*.params')
    assert len(cnst_files) == 1, "Too many or too few params files detected: " + str(cnst_files)
    ani_dict['cnstfile'] = cnst_files[0]
    ani_dict['saefile'] = directory + '/sae_linfit.dat'
    ani_dict['nnfprefix'] = directory + '/train'
    model_dirs = glob.glob(ani_dict['nnfprefix'] + '*')
    ani_dict['Nnet'] = len(model_dirs)

    return ani_dict


def compute_empirical_formula(S):
    """Computes an empirical formula to describe a chemical system.

    Args:
        S (list): List of strings where each strings represents an atom in the system.

    Returns:
        (str): An empircal formula representing the system.

    """
    uniques = np.unique(S, return_counts=True)
    arg_sort = np.argsort(uniques[0])

    return "_".join([i + str(j).zfill(2) for i, j in zip(uniques[0][arg_sort], uniques[1][arg_sort])])


def random_rotation_matrix(deflection=1.0, randnums=None):
    """Returns a random rotation matrix

    Args:
        deflection (float): Pole deflection.
        randnums (ndarray): A (3,) numpy array containing the parameters theta, phi, and z.

    Returns:
        (ndarray): Random 3x3 rotation matrix.

    """
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    r = np.sqrt(z)

    V = [np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)]

    st = np.sin(theta)
    ct = np.cos(theta)

    R_z = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    M = (np.outer(V, V) - np.eye(3)) @ R_z

    return M


def store_to_zarr(zarr_path, system_data, properties):
    """Stores the key results of the QM calculations in the database.

    Args:
        zarr_path (str): Path to zarr database.
        system_data (list): A list of MoleculesObjects objects.
        properties (dict): Dictionary defined in master.json whose keys are the properties that we want to retrieve
                           from the QM calculations and store in the database.

    Returns:
        (None)

    """
    print("Saving zarr database: " + zarr_path)
    print(f"Total Systems: {len(system_data)}")

    data_list = [system.to_dict(qm_keys=properties) for system in system_data if system.check_convergence()]

    print(f"Saved Systems: {len(data_list)}")

    print(data_list)


    if os.path.exists(zarr_path):
        database = ZarrDatabase.load_from_zarr(zarr_path)
    else:
        database = ZarrDatabase(zarr_path)
    database.add_instance(data_list)


def store_current_data(h5path, system_data, properties):
    """Stores the key results of the QM calculations in the database.

    Args:
        h5path (str): Path to store the .h5 files.
        system_data (list): A list of MoleculesObjects objects.
        properties (dict): Dictionary defined in master.json whose keys are the properties that we want to retrieve
                           from the QM calculations and store in the database.

    Returns:
        (None)

    """
    # system data is a list of [mol-id(string), atoms, properties dictionary]
    data_dict = {}
    print("Saving h5 file: " + h5path)
    total_number = len(system_data)
    saved_number = 0
    nan_number = 0
    unconverged_number = 0
    for system in system_data:
        assert isinstance(system, MoleculesObject), 'system must be an instance of MoleculesObject'

        cur_moliculeid = system.get_moleculeid()
        cur_atoms = system.get_atoms()
        cur_properties = system.get_results()
        molkey = compute_empirical_formula(cur_atoms.get_chemical_symbols())

        # Ensure system converged before saving
        if system.check_convergence():
            saved_number += 1
            atom_index = np.argsort(cur_atoms.get_atomic_numbers())
            # If there is already a molecule with the same formula, append
            if molkey in data_dict:
                data_dict[molkey]["_id"].append(cur_moliculeid)
                data_dict[molkey]["coordinates"].append(cur_atoms.get_positions()[atom_index])
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"].append(complete_cell(cur_atoms.get_cell()))
                for prop in properties:
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]].append(cur_properties[prop] * properties[prop][2])
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]].append(
                            np.array(cur_properties[prop])[atom_index] * properties[prop][2])
                    else:
                        raise RuntimeError('Unknown property format')
            # If there is not already a molecule with this empirical formula, make a new one
            else:
                data_dict[molkey] = {}
                data_dict[molkey]["species"] = np.array(cur_atoms.get_chemical_symbols())[atom_index]
                data_dict[molkey]["_id"] = [cur_moliculeid]
                data_dict[molkey]["coordinates"] = [cur_atoms.get_positions()[atom_index]]
                if any(cur_atoms.get_pbc()):
                    data_dict[molkey]["cell"] = [complete_cell(cur_atoms.get_cell())]
                for prop in properties.keys():
                    if properties[prop][1].lower() == "system":
                        data_dict[molkey][properties[prop][0]] = [cur_properties[prop] * properties[prop][2]]
                    elif properties[prop][1].lower() == "atomic":
                        data_dict[molkey][properties[prop][0]] = [
                            np.array(cur_properties[prop])[atom_index] * properties[prop][2]]
                    else:
                        raise RuntimeError('Unknown property format')
        elif not isinstance(system,
                            MoleculesObject):  # code never enter in this line, but leaving for now to avoid problems
            nan_number += 1
        elif not system.check_convergence():
            unconverged_number = unconverged_number + 1
        else:
            print("Warning: molecule not saved for unknown reason")
    print("Total Systems: " + str(total_number))
    print("Saved Systems: " + str(saved_number))
    print("NAN Systems: " + str(nan_number))
    print("Unconverged Systems: " + str(unconverged_number))

    for isokey in data_dict:
        # print('isokeys:',isokey)
        for propkey in data_dict[isokey]:
            if propkey.lower() in ['species', '_id']:
                data_dict[isokey][propkey] = [el.encode('utf-8') for el in list(data_dict[isokey][propkey])]
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
            else:
                data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
                # print("encoding species")
    #                    if type(data_dict[isokey][propkey]) is 'numpy.ndarray':
    #                        data_dict[isokey][propkey] = np.stack(data_dict[isokey][propkey])
    #                    else:
    #                        data_dict[isokey][propkey] = np.array(data_dict[isokey][propkey])
    #                    print('propkey:', propkey,data_dict[isokey][propkey].shape)
    #
    dpack = pyt.datapacker(h5path)
    for key in data_dict:
        dpack.store_data(key, **data_dict[key])
    dpack.cleanup()


# Recommend creation of parsl queue object
class parsl_task_queue():
    def __init__(self):
        # Create a list
        self.task_list = []

    def add_task(self, task):
        """Add a task to the task list
        """
        self.task_list.append(task)
        # self.task_list[-1].start()

    def get_completed_number(self):
        """Get the number of completed tasks.
        """
        task_status = [task.done() for task in self.task_list]
        return int(np.sum(task_status))

    def get_running_number(self):
        """Get the number of running tasks.
        """
        task_status = [task.running() for task in self.task_list]
        return int(np.sum(task_status))

    def get_number(self):
        """Get the the number of tasks in the task list.
        """
        return len(self.task_list)

    def get_queued_number(self):
        """Get the number of queued tasks.
        """
        return int(self.get_number() - self.get_running_number() - self.get_completed_number())

    def get_task_results(self):
        """Get the reults of the tasks

        Returns:
            (tuple): A tuple containing, respectively, a list that stores the task results and an int that tells
                     the number of failed tasks.
        """
        results_list = []
        failed_number = 0
        for taski, task in enumerate(self.task_list):
            task_status = task.task_status()
            if task_status == 'exec_done' and task.done:
                results_list.append(task.result())
                del self.task_list[taski]
            elif task_status == 'failed':
                failed_number += failed_number
                del self.task_list[taski]

        return results_list, failed_number

    def get_task_status(self):
        """Get the status of the tasks.

            Returns:
                (list): List containing the status of the tasks.
        """
        status_list = []
        for task in self.task_list:
            status_list.append(task.task_status())

        return status_list

    def print_status(self):
        """Prints the status of the tasks.
        """
        print('Total Tasks: {:d}'.format(self.get_number()))
        # print('Queued Tasks: {:d}'.format(self.get_queued_number()))
        # print('Running Tasks: {:d}'.format(self.get_running_number()))
        print('Finished Tasks: {:d}'.format(self.get_completed_number()))


# Used to find current version of directories to re-start with
def find_empty_directory(pattern):
    """Find an empty directory to restart.

    Args:
        (str): Common prefix of the directories to check.

    Returns:
        (int): Number of the empty directory.

    """
    curI = 0
    while os.path.exists(pattern.format(curI)):
        curI += 1

    return curI


# Throughout this code individual systems are passed around as three element lists
# element 1: metadata: this is required to include moleculeid,  but may also include sampling and other metadata
# element 2: an ASE atoms object.
# element 3: Evaluated QM properties
def system_checker(system, kill_on_fail=True, print_error=True):
    """Checks if the system returned by the builder meets all requeriments.

    Args:
        system (list): A list containing three elements. The first is a dict containing metadata of the system,
                       and one of its keys must be 'moleculeid' whose value is a unique identifier of the system.
                       The second element is an ASE Atoms object. The third element is a dict that stores the
                       desired properties from the QM calculation (e.g. forces and energies).
        kill_on_fail (bool): Kills the process if something goes wrong.
        print_error (bool): If True prints the error message if something goes wrong.

    Returns:
        (bool): True if 'system' meets all requirements and False otherwise.

    """
    try:
        assert isinstance(system, list) or isinstance(system, tuple)
        assert len(system) == 3
        assert isinstance(system[0], dict)
        assert isinstance(system[0]['moleculeid'], str)
        assert isinstance(system[1], Atoms)
        assert isinstance(system[2], dict)

        no_nan = True
        if np.sum(np.isnan(system[1].get_positions())) > 0:
            no_nan = False
        if any(system[1].get_pbc()):
            if np.sum(np.isnan(system[1].get_cell())) > 0:
                no_nan = False
        for prop in system[2]:
            if isinstance(system[2][prop], np.ndarray):
                if np.sum(np.isnan(system[2][prop])) > 0:
                    no_nan = False
        if not no_nan:
            raise RuntimeError('NAN in system')

        return True

    except Exception as e:
        if print_error:
            print(e)
        if kill_on_fail:
            raise RuntimeError('Atomic system failed to meet requirements.')

        return False


def load_config_file(path, master_directory=None):
    """Extracts the main parameters to run ALF from the master json file.

    Args:
        path (str): Path of the master json file.
        master_directory (str): Master directory to run ALF.

    Returns:
        (dict): Dictionary containing the main parameters needed to configure ALF.
    """
    with open(path, 'r') as input_file:
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
            elif entry[-4:].lower() == 'path':
                if config[entry][0] != '/':
                    config[entry] = master_directory + config[entry]
                # For every 'path' entry, make a corresponding 'dir' entry that holds files in the path
                dir_dict[entry[:-4] + 'dir'] = '/'.join(config[entry].split('/')[:-1]) + '/'
        config.update(dir_dict)

    return config


def load_module_from_config(config, module_field):
    """Loads a module from a config json file.

    Args:
        config (dict): Dictionary representing the config json file.
        module_field (str): Module to load.

    Returns:
        (type): Type of the class that was loaded from the config file.

    """
    module_string = '.'.join(config[module_field].split('.')[:-1])
    class_string = config[module_field].split('.')[-1]

    return getattr(import_module(module_string), class_string)


def load_module_from_string(module_field):
    """Loads a python module from a string.

    Args:
        module_field (str): Strings that contains the modules to be loaded.

    Returns:
        (type): Type of the class that was loaded from module_field.

    """
    module_string = '.'.join(module_field.split('.')[:-1])
    class_string = module_field.split('.')[-1]

    return getattr(import_module(module_string), class_string)


def build_input_dict(function, dictionary_list, use_local_space=False, raise_on_fail=False):
    """Builds an input dict by matching the parameters of a function with the keys of dictionaries in a list.

    Args:
        function (function): Callable object whose parameters going to be extracted.
        dictionary_list (list): List of dictionaries to search.
        use_local_space (bool): If True also looks for the parameters in the local symbol table.
        raise_on_fail (bool): If True raises a value error if this function fails.

    Returns:
        (dict): Dictionary containing whose keys are the parameters of 'function'.

    """
    if use_local_space:
        local_space_dict = locals()
    return_dictionary = {}
    sig = inspect.signature(function)
    input_params = list(sig.parameters)
    for parameter in input_params:
        for cur_dict in dictionary_list:
            if parameter in cur_dict.keys():
                return_dictionary[parameter] = cur_dict[parameter]
                break
        if not (parameter in return_dictionary):  # Still have not found the variable
            if use_local_space and (parameter in local_space_dict):
                return_dictionary[parameter] = local_space_dict[parameter]
            elif sig.parameters[parameter].default != inspect._empty:
                # There is a default value for this entry
                pass
            elif parameter == 'self':
                pass
            elif raise_on_fail:
                raise ValueError("Required input parameter {:s} of {:s} not defined in any space.".format(parameter,
                                                                                                          function.__name__))

    return return_dictionary
