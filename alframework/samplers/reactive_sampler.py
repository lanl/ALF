import os
import numpy as np
import ase
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory
import time
from ase import Atoms
import pickle as pkl
from parsl import python_app, bash_app

from alframework.tools.tools import annealing_schedule
from alframework.tools.tools import system_checker
from alframework.tools.tools import load_module_from_config
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.tools import build_input_dict
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

import sys

from ase import io
from ase.mep import NEB
from ase.optimize import BFGS

@python_app(executors=['alf_sampler_executor'])
def reactive_sampling(molecule_object, sampler_config,model_path,current_model_id,gpus_per_node, ase_calculator, N_neb, max_iter, neb_steps, reactive_sampling_method, Escut, Fscut, Fmmult=3.0) :
    # Performs reactive sampling with NEB/dimer methods based on input based in the config file
    
    failed=False

    # NEB runs first and attempts to find structure above uncertainity
    if 'neb' in reactive_sampling_method.lower(): # NEB sampling
        neb_images = neb_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps)
    
        for i in range(0,max_iter):

            # Check if NEB image already above threshold for uncertainty
            for neb_im in neb_images[1:]:
                neb_im.calc.calculate(neb_im,properties=['energy_stdev','forces_stdev_mean','forces_stdev_max'])
                Es = neb_im.calc.results['energy_stdev']
                Fs = neb_im.calc.results['forces_stdev_mean']
                Fsmax = neb_im.calc.results['forces_stdev_max']

                Ecrit = Es > Escut
                Fcrit = Fs > Fscut
                Fmcrit = Fsmax > Fmmult*Fscut

                if Ecrit or Fcrit or Fmcrit: # If above threshold then breaks loop
                    print('Above critical')
                    last_bad = Atoms(neb_im.get_chemical_symbols(),positions=neb_im.get_positions(wrap=True),cell=neb_im.get_cell(),pbc=neb_im.get_pbc())
                    failed = True
                    break

            if failed:
                break
        
            # Run neb again
            neb_images = neb_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps)

    # Dimer sampling only occurs if NEB structure not identified or only dimer method passed in config file
    if failed == False and 'dimer' in reactive_sampling_method.lower():
        
        dimer_struct = dimer_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps)

        for i in range(0,max_iter):
            dimer_struct.calc.calculate(dimer_struct,properties=['energy_stdev','forces_stdev_mean','forces_stdev_max'])
            Es = dimer_struct.calc.results['energy_stdev']
            Fs = dimer_struct.calc.results['forces_stdev_mean']
            Fsmax = dimer_struct.calc.results['forces_stdev_max']
            
            Ecrit = Es > Escut
            Fcrit = Fs > Fscut
            Fmcrit = Fsmax > Fmmult*Fscut

            if Ecrit or Fcrit or Fmcrit: 
                last_bad = Atoms(dimer_struct.get_chemical_symbols(),positions=dimer_struct.get_positions(wrap=True),cell=dimer_struct.get_cell(),pbc=dimer_struct.get_pbc())
                failed = True
                break

            dimer_struct = dimer_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps)


    if failed:
        ase_atoms = last_bad
        
        meta_dict = { "chemical_symbols" : ase_atoms.get_chemical_symbols(),
                      "positions" : ase_atoms.get_positions(wrap=True),
                      "cell" : ase_atoms.get_cell(),
                      "reactant":molecule_object.metadata['reactant'],
                      "product": molecule_object.metadata['product'],
                      "ts":molecule_object.metadata['ts']
                     }

        meta_dict.update(molecule_object.get_metadata())
    
        ase_atoms.calc = None  # replace calculator for return

        molecule_object.update_metadata(meta_dict)

        molecule_object.update_atoms(ase_atoms)
        return molecule_object
    else:
        # Else no update
        molecule_object.update_atoms(None)
        return molecule_object
    


def neb_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps):
    # ASE's neb sampler
    
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    
    gpu = '0' #os.environ.get('PARSL_WORKER_RANK') now using  cuda visible devices
    calculator_list = calc_class(model_path.format(current_model_id) + '/',device='cuda:'+gpu)
    
    # Read initial and final states:
    initial = molecule_object.metadata['reactant']
    final = molecule_object.metadata['product']

    print(initial.positions)
    print(final.positions)
    
    # Make a band consisting of N_neb images:
    images = [initial]
    images += [initial.copy() for i in range(N_neb)]
    images += [final]
    neb = NEB(images)

    # Interpolate linearly the potisions of the three middle images:
    neb.interpolate()
    
    # Set calculators:
    for image in neb.images:
        image.calc = MLMD_calculator(calculator_list)
        
    # Optimize:    
    optimizer = BFGS(neb)
    optimizer.run(fmax=0.04, steps=neb_steps)

    print('Finished NEB sampler')
    
    return neb.images

def dimer_sampler(molecule_object, sampler_config, model_path, current_model_id, N_neb, neb_steps):
    # ASE's dimer sampler
    
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    
    gpu = '0' #os.environ.get('PARSL_WORKER_RANK') now using  cuda visible devices
    calculator_list = calc_class(model_path.format(current_model_id) + '/',device='cuda:'+gpu)

    ts = molecule_object.metadata['ts']
    
    ts.calc = MLMD_calculator(calculator_list)
    
    # Set up the dimer                                                                                                                                                                           
    with DimerControl(
        initial_eigenmode_method="gauss",
        displacement_method="vector",
        logfile=None,
    ) as d_control:
        d_atoms = MinModeAtoms(ts, d_control)

	# Converge to a saddle point                                                                                                                                             
        with MinModeTranslate(
            d_atoms, trajectory="dimer_method.traj", logfile='dimer.log'
        ) as dim_rlx:
            dim_rlx.run(fmax=0.01, steps=50)
            
    ts.get_forces()
    
    return ts
