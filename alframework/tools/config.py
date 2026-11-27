"""Load workflow settings, resolve configured names, and bind task arguments."""

import inspect
from dataclasses import dataclass
import json
import os
from importlib import import_module
from pathlib import Path

import numpy as np


def load_config_file(path, master_directory=None, *, working_directory=None):
    """Extracts the main parameters to run ALF from the master json file.

    Args:
        path (str): Path of the master json file.
        master_directory (str): Master directory to run ALF.
        working_directory (str): The directory a ``"master_directory": "pwd"`` means; by default the current one,
            which is the run directory when the coordinator runs in it.

    Returns:
        (dict): Dictionary containing the main parameters needed to configure ALF.
    """
    with open(path, 'r') as input_file:
        config = json.load(input_file)
    if master_directory is None and "master_directory" in config:
        if config["master_directory"] == 'pwd':
            master_directory = str(working_directory or os.getcwd()) + '/'
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
                dir_dict[entry[:-4]+'dir'] = '/'.join(config[entry].split('/')[:-1]) + '/'
        config.update(dir_dict)
        topology = config.get('topology_check')
        if isinstance(topology, dict) and topology.get('enabled'):
            root = Path(master_directory or Path(path).resolve().parent)
            if topology.get('reference_conformer_path'):
                topology['reference_conformer_path'] = str(
                    (root / Path(topology['reference_conformer_path']).expanduser()).resolve())
            if isinstance(topology.get('references'), dict):
                topology['references'] = {
                    identity: str((root / Path(reference).expanduser()).resolve())
                    for identity, reference in topology['references'].items()
                }

    return config


def _resolve_dotted_name(name):
    components = name.split('.')
    return getattr(import_module('.'.join(components[:-1])), components[-1])


def load_module_from_string(module_field):
    """Resolve a Python attribute from its fully qualified name.

    Args:
        module_field (str): Dotted path to the attribute.

    Returns:
        object: The resolved attribute, typically a workflow task or loader.

    """
    return _resolve_dotted_name(module_field)


def load_module_from_config(config, module_field):
    """Resolve the dotted Python name stored under a configuration key.

    Args:
        config (dict): Workflow configuration.
        module_field (str): Configuration key containing a dotted name.

    Returns:
        object: The resolved attribute, typically a workflow task or loader.

    """
    return _resolve_dotted_name(config[module_field])


# The task configurations a master configuration names, by the key that holds each one's path.
WORKFLOW_CONFIG_PATHS = (('builder', 'builder_config_path'), ('sampler', 'sampler_config_path'),
                         ('QM', 'QM_config_path'), ('ML', 'ML_config_path'))


@dataclass
class WorkflowConfigs:
    """A run's five configurations as loaded from their files, before the ``scaling`` and ``sizing`` blocks apply."""
    master: dict
    builder: dict
    sampler: dict
    QM: dict
    ML: dict

    def as_list(self):
        """The configurations in the order tasks' arguments are looked up in: master, builder, sampler, QM, ML."""
        return [self.master, self.builder, self.sampler, self.QM, self.ML]


def load_workflow_configs(master_path, base=None, *, required=True):
    """Load the master configuration at ``master_path`` and the four it names, as the coordinator does.

    Relative configuration paths are read from the working directory, or from ``base`` when given (such as a
    run directory). With ``required`` false, a configuration the master doesn't name, or that doesn't exist, is
    an empty dictionary instead of an error.
    """
    with open(master_path) as handle:
        names_directory = 'master_directory' in json.load(handle)
    # Without master_directory, relative paths are read from base or the working directory, as open() reads them.
    master = load_config_file(master_path, None if names_directory else str(base or os.getcwd()) + '/',
                              working_directory=base)
    master.setdefault('master_directory', None)
    loaded = {}
    for name, key in WORKFLOW_CONFIG_PATHS:
        path = master.get(key)
        if path is not None and base is not None and not os.path.isabs(path):
            path = os.path.join(base, path)
        if path is None or not os.path.exists(path):
            if required:
                raise FileNotFoundError(f'The master configuration\'s {key} {master.get(key)!r} does not name a file.')
            loaded[name] = {}
            continue
        loaded[name] = load_config_file(path, master['master_directory'])
    return WorkflowConfigs(master=master, **loaded)


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
        if not (parameter in return_dictionary): # Still have not found the variable
            if use_local_space and (parameter in local_space_dict):
                return_dictionary[parameter] = local_space_dict[parameter]
            elif sig.parameters[parameter].default != inspect._empty:
                # There is a default value for this entry
                pass
            elif parameter == 'self':
                pass
            elif raise_on_fail:
                raise ValueError("Required input parameter {:s} of {:s} not defined in any space.".format(parameter, function.__name__))

    return return_dictionary


def coordinator_poll_interval(master_config):
    """Maximum seconds between coordinator checks; partial-batch deadlines may shorten it."""
    interval = master_config.get('poll_interval', 60)
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not np.isfinite(interval) or interval <= 0):
        raise ValueError(f'poll_interval must be a positive finite number of seconds; received {interval!r}.')
    return float(interval)


def stored_label_budget(master_config):
    """Return the validated ``maximum_stored_labels`` setting, or ``None`` when it isn't set."""
    budget = master_config.get('maximum_stored_labels')
    if budget is None:
        return None
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f'maximum_stored_labels must be a positive integer; received {budget!r}.')
    return budget
