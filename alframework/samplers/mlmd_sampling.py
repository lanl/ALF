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
    

#For now I will take a dictionary with all sample parameters.
#We may want to make this explicit. Ying Wai, what do you think? 
def mlmd_sampling(molecule_object, ase_calculator,dt,maxt,Escut,Fscut,Ncheck,Tamp,Tper,Tsrt,Tend,Ramp,Rper,Rend,meta_dir=None,use_potential_specific_code=None,trajectory_interval=None):
    system_checker(molecule_object)
    ase_atoms = molecule_object[1]
    T = annealing_schedule(0.0, maxt, Tamp, Tper, Tsrt, Tend)

    # Setup Rho
    if Rend == None:
        str_Rvalue=None
    else:
        str_Rvalue = (1.66054e-24/1.0e-24)*(np.sum(ase_atoms.get_masses())/ase_atoms.get_volume())

    # Set the ASE Calculator
    ase_atoms.set_calculator(ase_calculator)

    # Compute energies
    #ase_atoms.calc.get_potential_energy()

    # Define thermostat
    dyn = Langevin(ase_atoms, dt * units.fs, friction=0.02, temperature_K=T, communicator=None)
    if trajectory_interval != None:
        trajob = Trajectory( meta_dir+"/metadata-"+molecule_object[0]['moleculeid']+'.traj',mode='w',atoms=ase_atoms)
        dyn.attach(trajob.write,interval=trajectory_interval)

    # Iterate MD
    failed = False

    set_temp = T
    set_dens = str_Rvalue

    sim_start_time = time.time()

    temps = []
    denss = []
    
    dyn.run(1)

    for i in range(int(np.ceil((1000*maxt)/(dt*Ncheck)))):

        # Check if MD should be ran
        if use_potential_specific_code==None:
            ase_atoms.calc.calculate(ase_atoms,properties=['energy_stdev','forces_stdev_mean','forces_stdev_max'])
            Es = ase_atoms.calc.results['energy_stdev']
            Fs = ase_atoms.calc.results['forces_stdev_mean']
            Fsmax = ase_atoms.calc.results['forces_stdev_max']
        	
        elif use_potential_specific_code.lower() == 'neurochem':
            Es = ase_atoms.calc.Estddev*1000.0
            Fs, Fsmax = ase_atoms.calc.get_Fstddev()

        Ecrit = Es > Escut
        Fcrit = Fs > Fscut
        Fmcrit = Fsmax > 3*Fscut
    
        if Ecrit or Fcrit or Fmcrit:
            #print('MD FAIL (',model_id,',',self.counter,',',"{0:.4f}".format(time.time()-self.str_time),') -- Uncertainty:', "{0:.4f}".format(Es), "{0:.4f}".format(Fs), "{0:.4f}".format(Fsmax), (i*Ncheck*dt)/1000)
            last_bad = Atoms(ase_atoms.get_chemical_symbols(),positions=ase_atoms.get_positions(wrap=True),cell=ase_atoms.get_cell(),pbc=ase_atoms.get_pbc())
            failed = True

            break

        # Set the temperature
        set_temp = annealing_schedule((i * Ncheck * dt) / 1000, maxt, Tamp, Tper, Tsrt, Tend)
        dyn.set_temperature(temperature_K = set_temp)

        # Set the density
        if str_Rvalue ==None:
            pass
        else:
            set_dens = (1.0e-24 / 1.66054e-24) * annealing_schedule((i * Ncheck * dt) / 1000, maxt, Ramp, Rper, str_Rvalue, Rend)
            scalar = np.power(np.sum(ase_atoms.get_masses()) / (ase_atoms.get_volume() * set_dens), 1. / 3.)
            ase_atoms.set_cell(scalar * ase_atoms.get_cell(), scale_atoms=True)
            denss.append(set_dens)

        temps.append(set_temp)

        # Run MD
        dyn.run(Ncheck)

    meta_dict = {"realtime_simulation" : time.time()-sim_start_time,
                    "Es"    : Es,
                    "Fs"    : Fs,
                    "Fsmax" : Fsmax,
                    "simulationtime" : (i*Ncheck*dt)/1000,
                    "temps" : temps,
                    "denss" : denss,
                    "system_comp": np.unique(ase_atoms.get_chemical_symbols(),return_counts=True),
                    "Tamp" : Tamp,
                    "Tper" : Tper,
                    "Tsrt" : Tsrt,
                    "Tend" : Tend,
                    "Ramp" : Ramp,
                    "Rper" : Rper,
                    "Rsrt" : str_Rvalue,
                    "Rend" : Rend,
                    "Ecrit" : Ecrit,
                    "Fcrit" : Fcrit,
                    "Fmcrit" : Fmcrit,
                    "chemical_symbols" : ase_atoms.get_chemical_symbols(),
                    "positions" : ase_atoms.get_positions(wrap=True),
                    "cell" : ase_atoms.get_cell()
                }
    meta_dict.update(molecule_object[0])
    if meta_dir is not None:
        pkl.dump( meta_dict, open( meta_dir+"/metadata-"+molecule_object[0]['moleculeid']+'.p', "wb" ) )
    
    ase_atoms.calc = None  #replace calculator for return
    
    if failed:
        return [meta_dict, ase_atoms,{}]
    else:
        #print('MD SUCCESS',self.counter)
        return [meta_dict, None,{}]

@python_app(executors=['alf_sampler_executor'])
def simple_mlmd_sampling_task(molecule_object,sampler_config,model_path,current_model_id,gpus_per_node):
    """
    A simple implementation of uncertanty based MD sampling
    parameters:
    molecules_object: an object that conforms the the standard inposed by 'system_checker'
    sample_params: a dictionary containing the following quantities
      dt: MD time step in fs
      maxt: maximum number of MD time steps in ps
      Escut: Energy standard deviation threshold for capturing frame
      Fscut: Force standard deviation threshold for capturing frame
      Ncheck: Every X steps perform uncertanty checking
      srt_temp: [min,max] for starting temperature
      end_temp: [min,max] for ending temperature
      amp_temp: [min,max] for temperature fluctuations
      per_temp: [min,max] for temperature fluctuation period in ps
      end_dens: [min,max] ending pressure range
      amp_dens: [min,max] pressure fluctuation range
      per_dens: [min,max] pressure amplitude range
      meta_dir: path to store sampling statistics. 
    model_path (Set by master): Path to current ML model
    
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(os.environ.get('PARSL_WORKER_RANK'))%gpus_per_node)
    system_checker(molecule_object)
    feed_parameters = {}

    # Setup T
    feed_parameters['Tamp'] = np.random.uniform(sampler_config['amp_temp'][0], sampler_config['amp_temp'][1])
    feed_parameters['Tper'] = np.random.uniform(sampler_config['per_temp'][0], sampler_config['per_temp'][1])
    feed_parameters['Tsrt'] = np.random.uniform(sampler_config['srt_temp'][0], sampler_config['srt_temp'][1])
    feed_parameters['Tend'] = np.random.uniform(sampler_config['end_temp'][0], sampler_config['end_temp'][1])
    
    if sampler_config['amp_dens'] == None:
        feed_parameters['Ramp'] = None
    else: 
        feed_parameters['Ramp'] = np.random.uniform(sampler_config['amp_dens'][0], sampler_config['amp_dens'][1])
    if sampler_config['per_dens'] == None:
        feed_parameters['Rper'] = None
    else:
        feed_parameters['Rper'] = np.random.uniform(sampler_config['per_dens'][0], sampler_config['per_dens'][1])
    if sampler_config['end_dens'] == None:
        feed_parameters['Rend'] = None
    else:
        feed_parameters['Rend'] = np.random.uniform(sampler_config['end_dens'][0], sampler_config['end_dens'][1])
        
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    
    if sampler_config.get("use_potential_specific_code",None) == 'neurochem':
        model_info = {}
        model_info['model_path'] = model_path + '/'
        model_info['Nn'] = 8
        model_info['gpu'] = '0' #os.environ.get('PARSL_WORKER_RANK') now using cuda visible devices
        ase_calculator = calc_class(model_info)
            
    else:
        gpu = '0' #os.environ.get('PARSL_WORKER_RANK') now using  cuda visible devices
        calculator_list = calc_class(model_path + '/',device='cuda:'+gpu)
        ase_calculator = MLMD_calculator(calculator_list,**sampler_config['MLMD_calculator_options'])
        
    if sampler_config.get('translate_to_center',False):
        molecule_object[1].set_positions(molecule_object[1].get_positions()-molecule_object[1].get_center_of_mass())
    
    #build_input_dict will check for trajectory_frequency from feed_parameters first.
    if np.random.rand() < sampler_config.get('trajectory_frquency',0):
        feed_parameters['trajectory_interval'] = sampler_config.get('trajectory_interval',20)
    else:
        feed_parameters['trajectory_interval'] = sampler_config.get('trajectory_interval',None)     
    
    feed_parameters['molecule_object'] = molecule_object
    feed_parameters['ase_calculator'] = ase_calculator
    function_input = build_input_dict(mlmd_sampling,[feed_parameters,sampler_config])
    
    molecule_output = mlmd_sampling(**function_input)
    system_checker(molecule_output)
    return(molecule_output)
