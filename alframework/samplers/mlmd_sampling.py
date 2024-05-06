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
from alframework.tools.tools import load_module_from_config
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.tools.tools import build_input_dict
from alframework.tools.molecules_class import MoleculesObject
    

#For now I will take a dictionary with all sample parameters.
#We may want to make this explicit. Ying Wai, what do you think? 
def mlmd_sampling(molecule_object, ase_calculator, dt, maxt, Escut, Fscut, Ncheck, Tamp, Tper, Tsrt, Tend, Ramp, Rper,
                  Rend, meta_dir=None, use_potential_specific_code=None, trajectory_interval=None):
    """Uncertainty based MD sampling.

    The idea is that if the MD fails because the NNPs didn't agree on the energies and forces, then this function
    returns an ase.Atoms object containing the failed configuration in order to properly evaluate its energy
    and forces using the QM code. The disagreement between the NNPs are quantified by the standard deviation of
    the energy and forces output by all NNs in the ensemble.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        ase_calculator: An ase calculator.
        dt (float): Time step [fs].
        maxt (float): Total simulation time [ps].
        Escut (float): Maximum tolerated energy standard deviation before calling QM evaluation.
        Fscut (float): Maximum tolareted mean standard deviation of the forces before calling QM evaluation.
        Ncheck (int): Uncertainty evaluation is performed at each 'Ncheck' MD steps.
        Tamp (float): Amplitude of the temperature fluctuation.
        Tper (float): Period of the temperature fluctuation.
        Tsrt (float): Initial temperature.
        Tend (float): Final temperature.
        Ramp (float): Amplitude fo the density fluctuation.
        Rper (float): Period of the density fluctuation.
        Rend (float): Final density.
        meta_dir (str): Path of the directory containing the metadata.
        use_potential_specific_code (str): Determines whether to use a specific interatomic potential code.
        trajectory_interval (int): Write the trajectory at every 'trajectory_interval' steps.

    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations. If the std deviation of
                           the energy or forces are above the established threshold, molecule_object.atoms will store
                           the failed configuration, otherwise it will store None.
    """
    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

    ase_atoms = molecule_object.get_atoms()
    T = annealing_schedule(0.0, maxt, Tamp, Tper, Tsrt, Tend)

    # Setup Rho
    if Rend is None:
        str_Rvalue = None
    else:
        str_Rvalue = (1.66054e-24/1.0e-24) * (np.sum(ase_atoms.get_masses())/ase_atoms.get_volume())

    # Set the ASE Calculator
    ase_atoms.set_calculator(ase_calculator)

    # Compute energies
    # ase_atoms.calc.get_potential_energy()

    # Define thermostat
    dyn = Langevin(ase_atoms, dt * units.fs, friction=0.02, temperature_K=T, communicator=None)
    if trajectory_interval is not None:
        trajob = Trajectory(meta_dir + "/metadata-" + molecule_object.get_moleculeid() + '.traj', mode='w', atoms=ase_atoms)
        dyn.attach(trajob.write, interval=trajectory_interval)

    # Iterate MD
    failed = False

    sim_start_time = time.time()

    temps = []
    denss = []
    
    dyn.run(1)

    for i in range(int(np.ceil((1000*maxt)/(dt*Ncheck)))):

        # Check if MD should be ran
        if use_potential_specific_code is None:
            ase_atoms.calc.calculate(ase_atoms, properties=['energy_stdev', 'forces_stdev_mean', 'forces_stdev_max'])
            Es = ase_atoms.calc.results['energy_stdev']
            Fs = ase_atoms.calc.results['forces_stdev_mean']
            Fsmax = ase_atoms.calc.results['forces_stdev_max']
        	
        elif use_potential_specific_code.lower() == 'neurochem':
            Es = ase_atoms.calc.Estddev * 1000
            Fs, Fsmax = ase_atoms.calc.get_Fstddev()

        Ecrit = Es > Escut
        Fcrit = Fs > Fscut
        Fmcrit = Fsmax > Fmmult*Fscut
    
        if Ecrit or Fcrit or Fmcrit:
            # print('MD FAIL (',model_id,',',self.counter,',',"{0:.4f}".format(time.time()-self.str_time),') -- Uncertainty:', "{0:.4f}".format(Es), "{0:.4f}".format(Fs), "{0:.4f}".format(Fsmax), (i*Ncheck*dt)/1000)
            # last_bad = Atoms(ase_atoms.get_chemical_symbols(), positions=ase_atoms.get_positions(wrap=True),
            #                  cell=ase_atoms.get_cell(), pbc=ase_atoms.get_pbc())
            failed = True
            break

        # Set the temperature
        set_temp = annealing_schedule((i * Ncheck * dt) / 1000, maxt, Tamp, Tper, Tsrt, Tend)
        dyn.set_temperature(temperature_K=set_temp)

        # Set the density
        if str_Rvalue is None:
            pass
        else:
            set_dens = (1.0e-24 / 1.66054e-24) * annealing_schedule((i * Ncheck * dt) / 1000, maxt, Ramp, Rper, str_Rvalue, Rend)
            scalar = np.power(np.sum(ase_atoms.get_masses()) / (ase_atoms.get_volume() * set_dens), 1. / 3.)
            ase_atoms.set_cell(scalar * ase_atoms.get_cell(), scale_atoms=True)
            denss.append(set_dens)

        temps.append(set_temp)

        # Run MD
        dyn.run(Ncheck)

    meta_dict = {"realtime_simulation" : time.time() - sim_start_time,
                    "Es"    : Es,
                    "Fs"    : Fs,
                    "Fsmax" : Fsmax,
                    "simulationtime" : (i * Ncheck * dt)/1000,
                    "temps" : temps,
                    "denss" : denss,
                    "system_comp": np.unique(ase_atoms.get_chemical_symbols(), return_counts=True),
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

    meta_dict.update(molecule_object.get_metadata())

    if meta_dir is not None:
        pkl.dump(meta_dict, open(meta_dir + "/metadata-" + molecule_object.get_moleculeid() + '.p', "wb"))
    
    ase_atoms.calc = None  # replace calculator for return

    molecule_object.update_metadata(meta_dict)

    if failed:
        molecule_object.update_atoms(ase_atoms)
        return molecule_object
    else:
        # print('MD SUCCESS', self.counter)
        molecule_object.update_atoms(None)
        return molecule_object


@python_app(executors=['alf_sampler_executor'])
def simple_mlmd_sampling_task(molecule_object, sampler_config, model_path, current_model_id, gpus_per_node):
    """ A simple implementation of uncertanty based MD sampling.

    Args:
        molecule_object (MoleculesObject): An object from the class MoleculesObject.
        sampler_config (dict): A dictionary containing the following quantities specified in sampler_config.json:
                               dt (float): MD time step in [fs]
                               maxt (int): Total MD simulation time in [ps]
                               Escut (float): Energy standard deviation threshold for capturing frame
                               Fscut (float): Force standard deviation threshold for capturing frame
                               Ncheck (int): Every 'N' steps perform uncertanty checking
                               srt_temp (list): [min, max] for starting temperature
                               end_temp (list): [min, max] for ending temperature
                               amp_temp (list): [min, max] for temperature fluctuations
                               per_temp (list): [min, max] for temperature fluctuation period in [ps]
                               end_dens (list): [min, max] ending pressure range
                               amp_dens (list): [min, max] pressure fluctuation range
                               per_dens (list): [min, max] pressure amplitude range
                               meta_dir (list): path to store sampling statistics.
                               trajectory_frequency (float): Frequency of creating trajectory files of the MD simulation.
                               trajectory_interval (int): Writes trajectory file each 'trajectory_interval' steps.
        model_path (str): Path to current ML model as specified in the master_config.json file.
        current_model_id (int): An integer that identifies the current ML model in 'model_path'.
        gpus_per_node (int): Number of GPUs per node set by the master_config.json file.

    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations. If the std deviation of
                           the energy or forces are above the established threshold, molecule_object.atoms will store
                           the failed configuration, otherwise it will store None.
    
    """
    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(os.environ.get('PARSL_WORKER_RANK')) % gpus_per_node)
    feed_parameters = {}

    # Setup T
    feed_parameters['Tamp'] = np.random.uniform(sampler_config['amp_temp'][0], sampler_config['amp_temp'][1])
    feed_parameters['Tper'] = np.random.uniform(sampler_config['per_temp'][0], sampler_config['per_temp'][1])
    feed_parameters['Tsrt'] = np.random.uniform(sampler_config['srt_temp'][0], sampler_config['srt_temp'][1])
    feed_parameters['Tend'] = np.random.uniform(sampler_config['end_temp'][0], sampler_config['end_temp'][1])
    
    if sampler_config['amp_dens'] is None:
        feed_parameters['Ramp'] = None
    else: 
        feed_parameters['Ramp'] = np.random.uniform(sampler_config['amp_dens'][0], sampler_config['amp_dens'][1])

    if sampler_config['per_dens'] is None:
        feed_parameters['Rper'] = None
    else:
        feed_parameters['Rper'] = np.random.uniform(sampler_config['per_dens'][0], sampler_config['per_dens'][1])

    if sampler_config['end_dens'] is None:
        feed_parameters['Rend'] = None
    else:
        feed_parameters['Rend'] = np.random.uniform(sampler_config['end_dens'][0], sampler_config['end_dens'][1])
        
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    
    if sampler_config.get("use_potential_specific_code", None) == 'neurochem':
        model_info = {'model_path': model_path.format(current_model_id) + '/',
                      'Nn': 8,
                      'gpu': '0'}

        # os.environ.get('PARSL_WORKER_RANK') now using cuda visible devices
        ase_calculator = calc_class(model_info)
            
    else:
        gpu = '0' # os.environ.get('PARSL_WORKER_RANK') now using cuda visible devices
        calculator_list = calc_class(model_path.format(current_model_id) + '/', device='cuda:' + gpu)
        ase_calculator = MLMD_calculator(calculator_list, **sampler_config['MLMD_calculator_options'])
        
    if sampler_config.get('translate_to_center', False):
        molecule_object.atoms.set_positions(molecule_object.atoms.get_positions() - molecule_object.atoms.get_center_of_mass())
    
    # build_input_dict will check for trajectory_frequency from feed_parameters first.
    if np.random.rand() > sampler_config.get('trajectory_frequency', 0):
        sampler_config['trajectory_interval'] = None  
    
    feed_parameters['molecule_object'] = molecule_object
    feed_parameters['ase_calculator'] = ase_calculator
    function_input = build_input_dict(mlmd_sampling, [feed_parameters, sampler_config])
    
    molecule_output = mlmd_sampling(**function_input)

    return molecule_output
