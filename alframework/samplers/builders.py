import glob
import random

from ase.io import cfg

def cfg_loader(model_string,start_molecule_index,cfg_directory,number=1):
    #TODO: Add assertion that cfg_directory exists and has cfg files
    #TODO: Add assertion that number is a number
    cfg_list = glob.glob(cfg_directory+'/*.cfg')
    atom_list = [['mol-{:s}-{:010d}'.format(model_string,start_molecule_index+i),cfg.read_cfg(random.choice(cfg_list))] for i in range(number)]
    return(atom_list)