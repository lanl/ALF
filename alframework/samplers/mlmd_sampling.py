import numpy as np
from ase.constraints import FixCom
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory
import time
import pickle as pkl

from alframework.tools.annealing import annealing_schedule
from alframework.tools.config import load_module_from_config
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator
from alframework.samplers.engines.ase import has_fixed_atoms
from alframework.ml_interfaces.ensemble_inference import atomic_uncertainty_arrays
from alframework.tools.geometry import minimum_distance
from alframework.tools.config import build_input_dict
from alframework.tools.molecules import MoleculesObject
from alframework.tools.gpu import sampling_device, neurochem_device
from alframework.tools.tasks import alf_task
from alframework.tools.velocities import initialize_velocities, kinetic_temperature, velocity_initialization_settings
    

#For now I will take a dictionary with all sample parameters.
#We may want to make this explicit. Ying Wai, what do you think? 
def mlmd_sampling(molecule_object, ase_calculator, dt, maxt, Ncheck, Tamp, Tper, Tsrt, Tend, Ramp, Rper,
                  Rend, meta_dir=None, use_potential_specific_code=None, trajectory_interval=None, distcut=1.2, min_time=0,
                  velocity_initialization='preserve', random_seed=None, initial_velocity_temperature_K=None,
                  Escut=None, Fscut=None, acquisition=None):
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
        distcut (float): Minimum allowed distance between two atoms in [Angstrom]
        min_time (float): Minimum simulation time before checking for uncertanties in [ps]
        velocity_initialization (str): Preserve momenta (default), or draw thermal momenta only when missing.
        random_seed: Optional seed for the independent initial-velocity draw; scalar Langevin keeps its legacy RNG.
        initial_velocity_temperature_K (float): Optional temperature for drawing momenta. Defaults to the
                                               initial thermostat target.

    Returns:
        (MoleculesObject): A MoleculesObject containing the results of the QM calculations. If the std deviation of
                           the energy or forces are above the established threshold, molecule_object.atoms will store
                           the failed configuration, otherwise it will store None.
    """
    assert isinstance(molecule_object, MoleculesObject), 'molecule_object must be an instance of MoleculesObject'
    # The same frame decision as the multi-trajectory sampler: the ensemble's spread against Escut and Fscut, and any
    # terms of the run's own on the uncertainty its calculators report (acquisition.terms).
    from alframework.samplers.acquisition import (acquisition_settings, assess_frame, custom_terms,
                                                  resolve_force_atom_indices, selected_force_diagnostics)
    criteria = {key: value for key, value in dict(Escut=Escut, Fscut=Fscut, distcut=distcut, min_time=min_time,
                                                  acquisition=acquisition).items() if value is not None}
    settings = acquisition_settings(criteria)
    if settings['selection']['mode'] == 'final_frame':
        raise ValueError("acquisition.selection.mode 'final_frame' requires the batched sampler "
                         "(mlmd_trajectories_task).")
    own_uncertainty = sorted({term['uncertainty'] for term in custom_terms(criteria).values()})
    max_uncertainty = {}

    ase_atoms = molecule_object.get_atoms()
    if Rend is not None and ('surface_builder' in molecule_object.get_metadata() or has_fixed_atoms(ase_atoms)):
        raise ValueError('Sampling slabs or structures with FixAtoms requires a fixed cell. Set end_dens to null.')
    selected_sources = settings.get("selected_force_signals", ())
    force_atom_indices = resolve_force_atom_indices(settings.get("force_atoms"), ase_atoms,
                                                    molecule_object.get_moleculeid())
    force_diagnostics = {}
    T = annealing_schedule(0.0, maxt, Tamp, Tper, Tsrt, Tend)
    molecule_object.update_metadata(initialize_velocities(ase_atoms, T, velocity_initialization,
                                                        molecule_id=molecule_object.get_moleculeid(),
                                                        random_seed=random_seed,
                                                        initial_velocity_temperature_K=initial_velocity_temperature_K))

    # Setup Rho
    if Rend is None:
        str_Rvalue = None
    else:
        str_Rvalue = (1.66054e-24/1.0e-24) * (np.sum(ase_atoms.get_masses())/ase_atoms.get_volume())

    # Set the ASE Calculator
    ase_atoms.calc = ase_calculator

    # Compute energies
    # ase_atoms.calc.get_potential_energy()

    # FixAtoms already anchors the substrate. Adding FixCom would translate the fixed positions.
    constraints = list(ase_atoms.constraints)
    if not has_fixed_atoms(ase_atoms) and not any(isinstance(constraint, FixCom) for constraint in constraints):
        ase_atoms.set_constraint(constraints + [FixCom()])
    dyn = Langevin(ase_atoms, dt * units.fs, friction=0.02, temperature_K=T, fixcm=False)
    if trajectory_interval is not None:
        trajob = Trajectory(meta_dir + "/metadata-" + molecule_object.get_moleculeid() + '.traj', mode='w', atoms=ase_atoms)
        dyn.attach(trajob.write, interval=trajectory_interval)

    # Iterate MD
    failed = False

    sim_start_time = time.time()

    temps = []
    denss = []
    
    dyn.run(1)

    for i in range( int( np.ceil((1000*maxt)/(dt*Ncheck)) ) ):

        # Check if MD should be ran
        if use_potential_specific_code is None:
            ase_atoms.calc.calculate(ase_atoms, properties=['energy_stdev', 'forces_stdev_mean', 'forces_stdev_max'])
            Es = ase_atoms.calc.results['energy_stdev']
            Fs = ase_atoms.calc.results['forces_stdev_mean']
            Fsmax = ase_atoms.calc.results['forces_stdev_max']
        	
        elif use_potential_specific_code.lower() == 'neurochem':
            Es = ase_atoms.calc.Estddev * 1000
            Fs, Fsmax = ase_atoms.calc.get_Fstddev()

        results = getattr(ase_atoms.calc, 'results', {})
        own_scores = results.get('alf_uncertainty') or {}
        signals = {'Es': Es, 'Fs': Fs, 'Fsmax': Fsmax, **own_scores}
        if selected_sources:
            member_forces = results.get('forces_contributions')
            if member_forces is not None:
                member_forces = member_forces[:getattr(ase_atoms.calc, 'N_models', None)]
            force_diagnostics = selected_force_diagnostics(
                signals, selected_sources, force_atom_indices, member_forces=member_forces,
                atomic=results.get('alf_atomic_uncertainty'))
        for name in own_uncertainty:
            if name in signals:
                max_uncertainty[name] = max(signals[name], max_uncertainty.get(name, signals[name]))
        dist_min = minimum_distance(ase_atoms)
        # Ecrit, Fcrit and Fmcrit: the energy, mean-force and max-force spreads above Escut, Fscut and 3 Fscut.
        decision = assess_frame(signals, dist_min, i * Ncheck * dt / 1000, criteria, settings,
                                trajectory_id=molecule_object.get_moleculeid(), step=1 + i * Ncheck)
        Ecrit, Fcrit, Fmcrit, distcrit = (decision[key] for key in ('Ecrit', 'Fcrit', 'Fmcrit', 'distcrit'))
        if decision['outcome'] != 'continue':  # rejected for distcut, or a frame to label, once min_time has passed
            # (with acquisition.distcut_frames "label", a distcut violation is a frame to label)
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
                    "distmin": dist_min,
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
                    "distcrit": distcrit,
                    "chemical_symbols" : ase_atoms.get_chemical_symbols(),
                    "positions" : ase_atoms.get_positions(wrap=True),
                    "cell" : ase_atoms.get_cell()
                }
    if max_uncertainty:
        meta_dict['max_uncertainty'] = max_uncertainty

    meta_dict.update(molecule_object.get_metadata())
    meta_dict.update(kinetic_temperature_K=kinetic_temperature(ase_atoms),
                     thermostat_temperature_K=temps[-1] if temps else T)
    if failed and decision['outcome'] == 'eligible':
        atomic = atomic_uncertainty_arrays(results.get('alf_atomic_uncertainty', {}), len(ase_atoms))
        if atomic:
            meta_dict['atomic_uncertainty'] = {name: values.copy() for name, values in atomic.items()}
        if atomic or own_scores or force_diagnostics:
            meta_dict.update(force_diagnostics)
            meta_dict['uncertainty'] = {**own_scores, **{name: signals[name] for name in selected_sources}}
            for key in ('acquisition_trigger_flags', 'acquisition_cutoffs', 'acquisition_score_terms'):
                meta_dict[key] = decision[key]

    if meta_dir is not None:
        pkl.dump(meta_dict, open(meta_dir + "/metadata-" + molecule_object.get_moleculeid() + '.p', "wb"))
    
    ase_atoms.calc = None  # replace calculator for return
    # A FixCom left on would remove the net force from any labels read through these Atoms.
    ase_atoms.set_constraint(constraints)

    molecule_object.update_metadata(meta_dict)

    if failed and decision['outcome'] == 'eligible':
        molecule_object.update_atoms(ase_atoms)
        return molecule_object
    else:
        # print('MD SUCCESS', self.counter)
        molecule_object.update_atoms(None)
        return molecule_object


@alf_task('sampler')
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

    velocity_initialization_settings(sampler_config)
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
        device = neurochem_device(sampler_config, gpus_per_node)
        model_info = {'model_path': model_path.format(current_model_id) + '/',
                      'Nn': 8,
                      'gpu': device.split(':')[1]}

        # os.environ.get('PARSL_WORKER_RANK') now using cuda visible devices
        ase_calculator = calc_class(model_info)
            
    else:
        # Preserve this task's GPU default; explicit CPU execution is opt-in.
        device = sampling_device(dict(sampler_config, device=sampler_config.get('device') or 'cuda'),
                                 gpus_per_node)
        calculator_list = calc_class(model_path.format(current_model_id) + '/', device=device)
        from alframework.samplers.acquisition import require_members
        require_members(calculator_list, sampler_config)
        ase_calculator = MLMD_calculator(calculator_list, **sampler_config['MLMD_calculator_options'])
        
    if sampler_config.get('translate_to_center', False):
        if has_fixed_atoms(molecule_object.atoms):
            raise ValueError('Sampling with FixAtoms requires translate_to_center=false.')
        molecule_object.atoms.set_positions(molecule_object.atoms.get_positions() - molecule_object.atoms.get_center_of_mass())
    
    # build_input_dict will check for trajectory_frequency from feed_parameters first.
    if np.random.rand() > sampler_config.get('trajectory_frequency', 0):
        sampler_config['trajectory_interval'] = None  
    
    feed_parameters['molecule_object'] = molecule_object
    feed_parameters['ase_calculator'] = ase_calculator
    function_input = build_input_dict(mlmd_sampling, [feed_parameters, sampler_config])
    
    molecule_output = mlmd_sampling(**function_input)

    return molecule_output
