from parsl import python_app, bash_app
import sys
import multiprocessing
import re
import numpy as np
import glob
import os
from ase import units
def train_HIPNN_model(model_dir,h5_train_dir,energy_key,coordinates_key,species_key,network_params,
                      cell_key=None,
                      force_key=None,
                      hipnn_order = 'scalar',
                      energy_mae_loss_weight = 1e2,
                      energy_rmse_loss_weight = 1e2,
                      force_mae_loss_weight = 1.0,
                      force_rmse_loss_weight = 1.0,
                      rbar_loss_weight = 1.0,
                      l2_reg_loss_weight = 1e-6,
                      pairwise_inference_parameters = None,
                      valid_size = .1,
                      test_size = .1,
                      plot_every = 50,
                      h5_test_dir = None,
                      learning_rate = 1e-3,
                      scheduler_options = {"max_batch_size":128,"patience":25,"factor":0.5},
                      controller_options = {"batch_size":64,"eval_batch_size":128,"max_epochs":1000,"termination_patience":50},
                      device_string = '0',
                      from_multiprocessing_nGPU = None,
                      remove_high_energy_cut = None, 
                      remove_high_energy_std = None,
                      remove_high_forces_cut = None, 
                      remove_high_forces_std = None
                      ):
    """
    directory: directory where HIPPYNN model will reside
    h5_directory: directory where h5 files will reside
        energy: string, energy key inside inside h5 dataset. Must match db name in 'properties_list'.
        force: string, force key insdie h5 dataset. Must match db name in 'properties_list'. Do not define to remove force training
        coordinates': string, coordinates key inside h5 dataset. Must match db name in 'properties_list'.
        species': string, species key inside h5 dataset. Must match db name in 'properties_list'.
        cell': string, cell key inside h5 dataset. Must match db name in 'properties_list'. Do not define if data is not periodic.
        hipnn_order': string 'scalar','vector','quadradic'
        energy_mae_loss_weight': MAE energy loss weight.
        energy_rmse_loss_weight': RMSE energy loss weight.
        force_mae_loss_weight': MAE force loss weight
        force_rmse_loss_weight': 2 element list of scalars. [MAE weight, RMSE weight]
        rbar_loss_weight': Weight of rbar in loss
        l2_reg_loss_weight': Weight of l2_reg in loss. typically 1e-6
        network_params': Dictonary that defines the model.  See examples for details
        pairwise_inference_parameters': Dictionary that defines any p
        h5_directory': 
        external_test_set': None or directory
        plot_every': integer, every plot_every epochs, generate plots
        remove_high_energy_cut: float, or None: If none, all data is kept. If float, data with energies per atom that many standard deviations or larger are removed. 
        remove_high_energy_std: float, or None: If none, all data is kept. If float, data with energies per atom that many standard deviations or larger are removed. 
        remove_high_forces_cut: float, or None: If none, all data is kept. If float, data with forces that many standard deviations or larger are removed. 
        remove_high_forces_std: float, or None: If none, all data is kept. If float, data with forces that many standard deviations or larger are removed. 
    """
    
    #Import Torch and configure GPU. 
    #This must occur after the subprocess fork!!!
    import torch
    if device_string.lower() == "from_multiprocessing":
        process = multiprocessing.current_process()
        os.environ["CUDA_VISIBLE_DEVICES"] = str((process._identity[-1]-1)%from_multiprocessing_nGPU)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_string
    torch.cuda.set_device('cuda:0')
    import hippynn
    from hippynn import plotting
    from hippynn.graphs import inputs, networks, targets, physics, loss
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    from hippynn.experiment.assembly import assemble_for_training
    from hippynn.pretraining import set_e0_values
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
    from hippynn.experiment import SetupParams, setup_and_train
    from hippynn.databases.database import  prettyprint_arrays
    
    with hippynn.tools.active_directory(model_dir):
        with hippynn.tools.log_terminal("training_log.txt","wt"):
        	
            print("CUDA_VISIBLE_DEVICES: " + os.environ["CUDA_VISIBLE_DEVICES"])
            
            species = inputs.SpeciesNode(db_name=species_key)
            positions = inputs.PositionsNode(db_name=coordinates_key)
            if not(cell_key is None):
                cell = inputs.CellNode(db_name=cell_key)
                if hipnn_order.lower() == 'scalar':
                    network = networks.Hipnn("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                elif hipnn_order.lower() == 'vector':
                    network = networks.HipnnVec("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                elif hipnn_order.lower() == 'quadradic':
                    network = networks.HipnnQuad("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)
                else:
                    raise RuntimeError('Unrecognized hipnn_order parameter')
            else:
                if hipnn_order.lower() == 'scalar':
                    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                elif hipnn_order.lower() == 'vector':
                    network = networks.HipnnVec("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                elif hipnn_order.lower() == 'quadradic':
                    network = networks.HipnnQuad("HIPNN", (species, positions), module_kwargs=network_params, periodic=False)
                else:
                    raise RuntimeError('Unrecognized hipnn_order parameter')
            
            henergy = targets.HEnergyNode("HEnergy",network)
            sys_energy = henergy.mol_energy
            sys_energy.db_name = energy_key
            
            en_peratom = physics.PerAtom("TperAtom", sys_energy)
            en_peratom.db_name = energy_key+"peratom"
            
            hierarchicality = henergy.hierarchicality
            hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
            
            if not(force_key is None):
                force = physics.GradientNode("forces", (sys_energy, positions), sign=-1)
                force.db_name = force_key
            
            mae_energy = loss.MAELoss.of_node(sys_energy)
            rmse_energy = loss.MSELoss.of_node(sys_energy) ** (1 / 2)
            
            mae_energyperatom = loss.MAELoss.of_node(en_peratom)
            rmse_energyperatom = loss.MSELoss.of_node(en_peratom) ** (1 / 2)
            
            rbar = loss.Mean.of_node(hierarchicality)
            
            l2_reg = loss.l2reg(network)
            
            loss_energy = energy_mae_loss_weight * mae_energyperatom + \
                energy_rmse_loss_weight * rmse_energyperatom
            
            loss_regularization = rbar_loss_weight * rbar + \
                l2_reg_loss_weight * l2_reg
                        
            if not(force_key is None):
                mae_force = loss.MAELoss.of_node(force)
                rmse_force = loss.MSELoss.of_node(force) ** (1 / 2)
                loss_force = force_mae_loss_weight * mae_force + \
                    force_rmse_loss_weight * rmse_force
                
                loss_error = loss_energy + loss_force
            else:
                loss_error = loss_energy
                
            loss_train = loss_error + loss_regularization
            
            if not(force_key is None):
                validation_losses = {
                    "T-MAE": mae_energy,
                    "T-RMSE": rmse_energy,
                    "TperAtom-MAE": mae_energyperatom,
                    "TperAtom-RMSE": rmse_energyperatom,
                    "ForceMAE": mae_force,
                    "ForceRMSE": rmse_force,
                    "T-Hier": rbar,
                    "L2Reg": l2_reg,
                    "Loss-Error": loss_error,
                    "Loss-Regularization": loss_regularization,
                    "Loss": loss_train
                 }
            else: 
                validation_losses = {
                    "T-MAE": mae_energy,
                    "T-RMSE": rmse_energy,
                    "TperAtom-MAE": mae_energyperatom,
                    "TperAtom-RMSE": rmse_energyperatom,
                    "T-Hier": rbar,
                    "L2Reg": l2_reg,
                    "Loss-Error": loss_error,
                    "Loss-Regularization": loss_regularization,
                    "Loss": loss_train
                 }
            early_stopping_key = "Loss-Error"
            
            if force_key == None:
                plot_maker = plotting.PlotMaker(
                    plotting.Hist2D.compare(sys_energy, saved=True),
                    plotting.Hist2D.compare(en_peratom, saved=True),
                    plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),
                    plot_every=plot_every,
                )
            else: 
                plot_maker = plotting.PlotMaker(
                    plotting.Hist2D.compare(sys_energy, saved=True),
                    plotting.Hist2D.compare(en_peratom, saved=True),
                    plotting.Hist2D.compare(force, saved=True),
                    plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="sense_0.pdf"),
                    plot_every=plot_every,
                )
            
            training_modules, db_info = assemble_for_training(loss_train, validation_losses, plot_maker=plot_maker)
            
            model, loss_module, model_evaluator= training_modules
            
            database = PyAniDirectoryDB(directory=h5_train_dir,allow_unfound=True,quiet=False,seed=np.random.randint(1e9),inputs=None,targets=None)
            
            #Ensure that species array is int64 for indexing
            arrays = database.arr_dict
            arrays[species_key] = arrays[species_key].astype('int64')
            n_atoms = arrays[species_key].astype(bool).astype(int).sum(axis=1)
            if not( energy_key+"peratom" in arrays):
                arrays[energy_key+"peratom"] =  arrays["energy"] / n_atoms
            
            #Ensure that float arrays are float32
            for k, v in arrays.items():
                if v.dtype == np.dtype('float64'):
                    arrays[k] = v.astype(np.float32)
            
            database.inputs = db_info["inputs"]
            database.targets = db_info["targets"]
            
            for key in list(arrays):
                if not(key in db_info['inputs']) and not(key in db_info['targets']) and not(key=='indices'):
                    del arrays[key]
            
            #Remove High energies and forces
            database.remove_high_property(force_key,True,species_key=species_key,cut=remove_high_forces_cut,std_factor=remove_high_forces_std)
            database.remove_high_property(energy_key,False,species_key=species_key,cut=remove_high_energy_cut,std_factor=remove_high_energy_std)
            
            print("Array Shapes After Cleaning")
            prettyprint_arrays(database.arr_dict)
            print(np.max(database.arr_dict['forces']))
            print(np.min(database.arr_dict['forces']))
            print(np.max(database.arr_dict['energyperatom']))
            print(np.min(database.arr_dict['energyperatom']))
            
            database.make_random_split("valid",valid_size)
            if not(h5_test_dir is None):
                database_test = PyAniDirectoryDB(directory=h5_test_dir,allow_unfound=True,quiet=False,seed=np.random.randint(1e9),**db_info)
                #Ensure that species array is int64 for indexing
                arrays_test = database.arr_dict
                arrays_test[species_key] = arrays_test[species_key].astype('int64')
            
                #Ensure that float arrays are float32
                for k, v in arrays_test.items():
                    if v.dtype == np.dtype('float64'):
                        arrays[k] = v.astype(np.float32)
                database_test.split_the_rest("test")
                database.split_the_rest("train")
                database.splits['test'] = database_test.splits['test']
            else:
                database.make_random_split("test",test_size)
                database.split_the_rest("train")
            
            #Set E0 values
            training_modules.model.to(database.splits['train'][sys_energy.db_name].device)
            
            set_e0_values(henergy, database, peratom=False, energy_name=energy_key, decay_factor=1e-2)
            
            optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=learning_rate)
            
            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                **scheduler_options
                )

            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key=early_stopping_key,
                **controller_options
                )
            experiment_params = SetupParams(controller=controller)

            print("Experiment Params:")
            print(experiment_params)

            # Parameters describing the training procedure.

            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )
            
def train_HIPNN_model_wrapper(arg_dict):
    return(train_HIPNN_model(**arg_dict))

@python_app(executors=['alf_ML_executor'])
def train_HIPPYNN_ensemble_task(configuration,h5_dir,model_path,model_index,nGPU,remove_existing=False,h5_test_dir=None):
    p = multiprocessing.Pool(nGPU)
    general_configuration = configuration.copy()
    n_models = general_configuration.pop('n_models')
    params_list = [general_configuration.copy() for i in range(n_models)]
    for i,cur_dict in enumerate(params_list):
        cur_dict['model_dir'] = model_path + '/model-{:02d}'.format(i)
        cur_dict['h5_train_dir'] = h5_dir
        cur_dict['from_multiprocessing_nGPU'] = nGPU
    out = p.map(train_HIPNN_model_wrapper,params_list)
    completed = []
    HIPNN_complete = re.compile('Training complete')
    
    p.close()
    for i in range(n_models):
        log = open(model_path + '/model-{:02d}/training_log.txt'.format(i),'r').read()
        if len(HIPNN_complete.findall(log))==1:
            completed.append(True)
        else:
            completed.append(False)
    return(completed, model_index)
    

def HIPNN_ASE_calculator(HIPNN_model_directory,device="cuda:0"):
    from hippynn.experiment.serialization import load_checkpoint_from_cwd
    from hippynn.tools import active_directory
    from hippynn.interfaces.ase_interface import HippynnCalculator
    import torch
    with active_directory(HIPNN_model_directory):
        bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
    model = bundle["training_modules"].model
    energy_node = model.node_from_name("energy")
    calc = HippynnCalculator(energy_node, en_unit=units.eV)
    calc.to(torch.float32)
    calc.to(torch.device(device))
    return(calc)
    
def HIPNN_ASE_load_ensemble(HIPNN_ensemble_directory,device="cuda:0"):
    model_list = []
    for cur_dir in glob.glob(HIPNN_ensemble_directory + '/model-*/'):
        model_list.append(HIPNN_ASE_calculator(cur_dir,device=device))
    return(model_list)
