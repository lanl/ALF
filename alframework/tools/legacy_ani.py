"""Discover parameter files and ensemble members for legacy ANI models."""

import glob


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
