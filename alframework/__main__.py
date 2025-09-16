###################################
# Step 1: Specify a configuration #
###################################

import os
import json
import pickle
import time
import sys
from multiprocessing import Process
from pathlib import Path
import numpy as np
import argparse
np.set_printoptions(threshold=np.inf)

# Load ASE library
import ase
from ase import Atoms

import parsl
# Check to see if parsl is available
import alframework
#from alframework.parsl_resource_configs.darwin import config_atdm_ml
from alframework.tools.tools import parsl_task_queue
from alframework.tools.tools import store_current_data
from alframework.tools.tools import load_config_file
from alframework.tools.tools import find_empty_directory
from alframework.tools.tools import system_checker
from alframework.tools.tools import load_module_from_string
from alframework.tools.tools import build_input_dict
from alframework.tools.pyanitools import anidataloader
from alframework.tools.molecules_class import MoleculesObject
#import logging
#logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--master',required=True)
parser.add_argument('--test_builder',action='store_true',default=False)
parser.add_argument('--test_qm',action='store_true',default=False)
parser.add_argument('--test_sampler',action='store_true',default=False)
parser.add_argument('--test_ml',action='store_true',default=False)
args = parser.parse_args()

# Load the master configuration:
master_config = load_config_file(args.master)
if 'master_directory' not in master_config:
    master_config['master_directory'] = None

# Load the builder config:
builder_config = load_config_file(master_config['builder_config_path'], master_config['master_directory'])

# Load the sampler config:
sampler_config = load_config_file(master_config['sampler_config_path'], master_config['master_directory'])

# Load the QM config:
QM_config = load_config_file(master_config['QM_config_path'], master_config['master_directory'])

# Load the ML config:
ML_config = load_config_file(master_config['ML_config_path'], master_config['master_directory'])

# All configs
all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]

# Define queues:
QM_task_queue = parsl_task_queue()
ML_task_queue = parsl_task_queue()
builder_task_queue = parsl_task_queue()
sampler_task_queue = parsl_task_queue()

if (args.test_builder or args.test_qm or args.test_sampler or args.test_ml) and 'parsl_debug_configuration' in master_config:
    parsl_configuration = load_module_from_string(master_config['parsl_debug_configuration'])
else: 
    parsl_configuration = load_module_from_string(master_config['parsl_configuration'])

executor_list = []
# Create List of parsl executors
for curExec in parsl_configuration.executors:
    executor_list.append(curExec.label)

# Load the Parsl config
parsl.load(parsl_configuration)

if master_config.get('plotting_utility', None) is not None:
    analysis_plot = load_module_from_string(master_config['plotting_utility'])
else:
    analysis_plot = None

# Make needed directories
for config in all_configs:
    for entry in config.keys():
        if entry[-3:] == 'dir':
            tempPath = Path(config[entry])
            tempPath.mkdir(parents=True, exist_ok=True)

#############################
# Step 2: Define Parsl tasks#
#############################
# Builder
builder_task = load_module_from_string(master_config['builder_task'])
for cur_Exec in builder_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        builder_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))

# Sampler
sampler_task = load_module_from_string(master_config['sampler_task'])
for cur_Exec in sampler_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        sampler_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))

# QM
qm_task = load_module_from_string(master_config['QM_task'])
for cur_Exec in qm_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        qm_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))

#ML
ml_task = load_module_from_string(master_config['ML_task'])
for cur_Exec in ml_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        ml_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))

##########################################
## Step 3: Evaluate restart possibilites #
##########################################
if os.path.exists(master_config['status_path']):
    with open(master_config['status_path'],'r') as input_file:
        status = json.load(input_file)
else:
    status = {}
    status['current_training_id'] = find_empty_directory(master_config['model_path'])
    if status['current_training_id'] > 0:
        status['current_model_id'] = status['current_training_id'] - 1
    else:
        status['current_model_id'] = -1

    status['current_h5_id'] = find_empty_directory(master_config['h5_path'])
    ##If data exists, start training model
    #if status['current_h5_id'] > 0:
    #     ML_task_queue.add_task(ml_task(ML_config,master_config['h5_path'],master_config['model_path'],status['current_training_id'],remove_existing=False))
    #     status['current_training_id'] = status['current_training_id'] + 1
    status['current_molecule_id'] = 0
    status['lifetime_failed_builder_tasks'] = 0
    status['lifetime_failed_sampler_tasks'] = 0
    status['lifetime_failed_ML_tasks'] = 0
    status['lifetime_failed_QM_tasks'] = 0

with open(master_config['status_path'], "w") as outfile:
    json.dump(status, outfile, indent=2)

######################################
# Step 4: Check if testing requested #
######################################

testing = False

if args.test_builder or args.test_sampler or args.test_qm:
    task_input = build_input_dict(builder_task.func,
                                  [{"moleculeid": 'test_builder', "moleculeids": ['test_builder'],
                                    "builder_config": builder_config}, *all_configs, status],
                                  raise_on_fail=True)

    builder_task_queue.add_task(builder_task(**task_input))
    builder_configuration = builder_task_queue.task_list[0].result()
    queue_output = builder_task_queue.get_task_results()
    test_configuration = queue_output[0][0]
    if not isinstance(test_configuration, MoleculesObject):
        assert isinstance(test_configuration[0], MoleculesObject), 'test_configuration[0] must be a MoleculesObject instance'
        test_configuration = test_configuration[0]
    print("Builder testing returned:")
    print(test_configuration)
    testing = True
    
if args.test_sampler:
    #Check that there is a model available
    next_model = find_empty_directory(master_config['model_path'])
    if status['current_model_id'] < 0:
        raise RuntimeError("Need to train model before testing sampling")
    print(master_config['model_path'].format(status['current_model_id']))
    task_input = build_input_dict(sampler_task.func,
                                  [{"molecule_object": test_configuration, "sampler_config": sampler_config},
                                   *all_configs, status],
                                  raise_on_fail=True)
    sampler_task_queue.add_task(sampler_task(**task_input))
    sampled_configuration = sampler_task_queue.task_list[0].result()
    queue_output = sampler_task_queue.get_task_results()
    test_configuration = queue_output[0][0]
    assert isinstance(test_configuration, MoleculesObject), 'test_configuration must be a MoleculesObject instance'
    print("Sampler testing returned:")
    print(test_configuration)
    testing = True

#def ase_calculator_task(input_system,configuration_list,directory,command,properties=['energy','forces']):
if args.test_qm:
    task_input = build_input_dict(qm_task.func,
                                  [{"molecule_object": test_configuration, "QM_config": QM_config},
                                   *all_configs, status],
                                  raise_on_fail=True)
    QM_task_queue.add_task(qm_task(**task_input))
    qm_result = QM_task_queue.task_list[0].result()
    queue_output,failed = QM_task_queue.get_task_results()
    sys.stdout.write('Job Failed: {:d}'.format(failed))
    test_configuration = queue_output[0]
    assert isinstance(test_configuration, MoleculesObject), 'test_configuration must be a MoleculesObject instance'
    print("Ensure the following data matches that in the QM output and test.h5 after unit conversion:")
    sys.stdout.write('Atomic Numbers: ' + np.array_str(test_configuration[1].get_atomic_numbers()) + '\n')
    sys.stdout.write('Coordinates: \n' + np.array_str(test_configuration[1].get_positions(),precision=4) + '\n')
    sys.stdout.write('PBCs: ' + np.array_str(test_configuration[1].get_pbc()) + '\n')
    if any(test_configuration[1].get_pbc()):
        sys.stdout.write('Cell: \n' + np.array_str(test_configuration[1].get_cell(),precision=4) + '\n')
    for property_key in list(master_config['properties_list']):
        if isinstance(test_configuration[2][property_key], np.ndarray):
            sys.stdout.write(property_key + ':\n' + np.array_str(test_configuration[2][property_key],precision=4) + '\n')
        else:
            sys.stdout.write(property_key + ': ' + str(test_configuration[2][property_key]) + '\n')
    if os.path.exists(master_config['master_directory'] + '/qm_test.h5'):
        os.remove(master_config['master_directory'] + '/qm_test.h5')
    store_current_data(master_config['master_directory'] + '/qm_test.h5',queue_output,master_config['properties_list'])
    test_db = anidataloader(master_config['master_directory'] + '/qm_test.h5')
    
    print("Data written to qm_test.h5:")
    for test_data in test_db:
        sys.stdout.write('Species: ' + " ".join(test_data['species']) + '\n')
        sys.stdout.write("Coordinates:\n" + np.array_str(test_data['coordinates'],precision=4) + '\n')
        if 'cell' in test_data.keys():
            sys.stdout.write('cell: \n' + np.array_str(test_data['cell'],precision=4) + '\n')
        for property_key in list(master_config['properties_list']):
            db_string = master_config['properties_list'][property_key][0]
            sys.stdout.write(db_string + ': \n' + np.array_str(test_data[db_string],precision=4) + '\n')
        
    testing = True
    
#train_ANI_model_task(configuration,data_directory,model_path,model_index,remove_existing=False):
if args.test_ml:
    #configuration,data_directory,model_path,model_index,remove_existing=False
    task_input = build_input_dict(ml_task.func,
                                  [{"ML_config": ML_config}, *all_configs, status],
                                  raise_on_fail=True)
    ML_task_queue.add_task(ml_task(**task_input))
    status['current_training_id'] = status['current_training_id'] + 1
    with open(master_config['status_path'], "w") as outfile:
        json.dump(status, outfile, indent=2)
    ml_result = ML_task_queue.task_list[0].result()
    queue_output = ML_task_queue.get_task_results()
    returned_models = queue_output[0]
    print("ML ensemble training status:")
    print(returned_models)
    for network in returned_models:
        if all(network[0]) and (network[1] > status['current_model_id']):
            print('New Model: {:04d}'.format(network[1]))
            status['current_model_id'] = network[1]
    with open(master_config['status_path'], "w") as outfile:
        json.dump(status, outfile, indent=2)
    testing=True
    
if testing:
    exit()
    
########################
# Step 5: Bootstraping #
########################
    
#If there is no data and no models, start boostrap jobs
if status['current_h5_id'] == 0 and status['current_model_id'] < 0:
    print("Building Bootstrap Set")
    while QM_task_queue.get_exec_done_number() < master_config['bootstrap_set']:
        if QM_task_queue.get_queued_number() < master_config['target_queued_QM']:
            while builder_task_queue.get_number() * master_config.get('maximum_builder_structures', 1) < master_config['parallel_samplers']:
                moleculeids = ['mol-boot-{:010d}'.format(it_ind) for it_ind in
                               range(status['current_molecule_id'], status['current_molecule_id'] + master_config.get('maximum_builder_structures', 1))
                               ]
                task_input = build_input_dict(builder_task.func,
                                              [{"moleculeid": 'mol-boot-{:010d}'.format(status['current_molecule_id']),
                                                "moleculeids": moleculeids, "builder_config": builder_config},
                                               *all_configs, status],
                                              raise_on_fail=True)
                builder_task_queue.add_task(builder_task(**task_input))
                status['current_molecule_id'] = status['current_molecule_id'] + master_config.get('maximum_builder_structures',1)
        
        if builder_task_queue.get_exec_done_number() > master_config['minimum_QM']:
            builder_results, failed = builder_task_queue.get_task_results()
            status['lifetime_failed_builder_tasks'] = status['lifetime_failed_builder_tasks'] + failed
            for structure in builder_results:
                #If builders return a single structure:
                if isinstance(structure, MoleculesObject):
                    task_input = build_input_dict(qm_task.func,
                                                  [{"molecule_object": structure, "QM_config": QM_config},
                                                   *all_configs, status],
                                                  raise_on_fail=True)
                    QM_task_queue.add_task(qm_task(**task_input))
                elif isinstance(structure, list):
                    for substructure in structure:
                        assert isinstance(substructure, MoleculesObject), 'substructure must be a MoleculesObject instance'
                        task_input = build_input_dict(qm_task.func,
                                                      [{"molecule_object": substructure, "QM_config": QM_config},
                                                       *all_configs, status],
                                                      raise_on_fail=True)
                        QM_task_queue.add_task(qm_task(**task_input))
                
        print("### Bootstraping Learning Status at: " + time.ctime() + " ###")
        print("builder status:")
        builder_task_queue.print_status()
        print("QM status:")
        QM_task_queue.print_status()
    
        with open(master_config['status_path'], "w") as outfile:
            json.dump(status, outfile, indent=2)
            
        time.sleep(60)

    print("Saving Bootstrap and training model")
    results_list, failed = QM_task_queue.get_task_results()
    status['lifetime_failed_QM_tasks'] = status['lifetime_failed_QM_tasks'] + failed    
    store_current_data(master_config['h5_path'].format(status['current_h5_id']),
                       results_list,
                       master_config['properties_list'])
    status['current_h5_id'] = status['current_h5_id'] + 1
    
if status['current_model_id'] < 0:
    task_input = build_input_dict(ml_task.func,
                                  [{"ML_config": ML_config}, *all_configs, status],
                                  raise_on_fail=True)
    ML_task_queue.add_task(ml_task(**task_input))
    
    status['current_training_id'] = status['current_training_id'] + 1
    
    with open(master_config['status_path'], "w") as outfile:
        json.dump(status, outfile, indent=2)
    network = ML_task_queue.task_list[0].result()
    if all(network[0]):
        status['current_model_id'] = network[1]
    else:
        print("Bootstrap network failed to train")
        print("User Investigation Required")
        exit()


##################################
## Step 6: Begin Active Learning #
##################################
master_loop_iter = 1
while True:
    #Re-load configurations, but watch for stupid errors
    try:
        # Load the master configuration:
        master_config_new = load_config_file(sys.argv[1])
        if 'master_directory' not in master_config:
            master_config_new['master_directory'] = None

        # Load the builder config:
        builder_config_new = load_config_file(master_config_new['builder_config_path'], master_config_new['master_directory'])
        
        # Load the sampler config:
        sampler_config_new = load_config_file(master_config_new['sampler_config_path'], master_config_new['master_directory'])
        
        # Load the QM config:
        QM_config_new = load_config_file(master_config_new['QM_config_path'], master_config_new['master_directory'])
        
        # Load the ML config:
        ML_config_new = load_config_file(master_config_new['ML_config_path'], master_config_new['master_directory'])
    except Exception as e:
        print("Failed to re-load configuration files:")
        print(e)
    else:
        master_config = master_config_new
        builder_config = builder_config_new
        sampler_config = sampler_config_new
        QM_config = QM_config_new
        ML_config = ML_config_new
        all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]
	
    # Run more builders
    if (QM_task_queue.get_queued_number() < master_config['target_queued_QM']) and \
            (QM_task_queue.get_number() < master_config.get('maximum_completed_QM', 1e12)):
        while sampler_task_queue.get_number() + builder_task_queue.get_number() * master_config.get('maximum_builder_structures', 1) \
                < master_config['parallel_samplers']:
            moleculeids = ['mol-{:04d}-{:010d}'.format(status['current_model_id'], it_ind) for it_ind in
                           range(status['current_molecule_id'], status['current_molecule_id']+master_config.get('maximum_builder_structures',1))]
            task_input = build_input_dict(builder_task.func,
                                          [{"moleculeid": 'mol-{:04d}-{:010d}'.format(status['current_model_id'], status['current_molecule_id']),
                                            "moleculeids": moleculeids, "builder_config": builder_config},
                                           *all_configs, status],
                                          raise_on_fail=True)
            builder_task_queue.add_task(builder_task(**task_input))
            status['current_molecule_id'] = status['current_molecule_id'] + master_config.get('maximum_builder_structures',1)
            
    # Builders go stright into samplers
    if builder_task_queue.get_exec_done_number() > 0:
        structure_list, failed = builder_task_queue.get_task_results()
        status['lifetime_failed_builder_tasks'] = status['lifetime_failed_builder_tasks'] + failed
        # Douple loop to facilitate possiblitiy of multiple sctructures returned by builder
        for structure in structure_list:
            # If builders return a single structure:
            if isinstance(structure, MoleculesObject):
                task_input = build_input_dict(sampler_task.func,
                                              [{"molecule_object": structure, "sampler_config": sampler_config},
                                               *all_configs, status],
                                              raise_on_fail=True)
                sampler_task_queue.add_task(sampler_task(**task_input))
            # If builders return multiple structures
            elif isinstance(structure, list):
                for substructure in structure:
                    assert isinstance(substructure, MoleculesObject), 'substructure must be a MoleculesObject instance'
                    task_input = build_input_dict(sampler_task.func,
                                                  [{"molecule_object": substructure, "sampler_config": sampler_config},
                                                   *all_configs, status],
                                                  raise_on_fail=True)
                    sampler_task_queue.add_task(sampler_task(**task_input))

    # Run more QM
    if sampler_task_queue.get_completed_number() > master_config['minimum_QM']:
        sampler_results, failed = sampler_task_queue.get_task_results()
        status['lifetime_failed_sampler_tasks'] = status['lifetime_failed_sampler_tasks'] + failed
        for structure in sampler_results: #may need [0]
            assert isinstance(structure, MoleculesObject), 'structure must be a MoleculesObject instance'
            if structure.get_atoms() is not None:
                task_input = build_input_dict(qm_task.func,
                                              [{"molecule_object": structure, "QM_config": QM_config}, master_config,
                                               *all_configs, status],
                                              raise_on_fail=True)
                QM_task_queue.add_task(qm_task(**task_input))

    # Train more models
    if (QM_task_queue.get_completed_number() > master_config['save_h5_threshold']) and (ML_task_queue.get_number() < 1):
        #print(QM_task_queue.task_list[0].result())
    	  #store_current_data(h5path, system_data, properties):
        results_list, failed = QM_task_queue.get_task_results()
        status['lifetime_failed_QM_tasks'] = status['lifetime_failed_QM_tasks'] + failed
        #with open('temp-{:04d}.pkl'.format(status['current_h5_id']),'wb') as pickle_file:
        #    pickle.dump(results_list,pickle_file)
        store_current_data(master_config['h5_path'].format(status['current_h5_id']), results_list, master_config['properties_list'])
#        with open('data-bk-{:04d}.pickle'.format(status['current_h5_id']),'wb') as pkbk: 
#            pickle.dump(results_list,pkbk)
        status['current_h5_id'] = status['current_h5_id'] + 1
        task_input = build_input_dict(ml_task.func,
                                      [{"ML_config": ML_config}, *all_configs, status],
                                      raise_on_fail=True)
        ML_task_queue.add_task(ml_task(**task_input))
        status['current_training_id'] = status['current_training_id'] + 1
        
    # Update Model
    if ML_task_queue.get_completed_number() > 0:
        output, failed = ML_task_queue.get_task_results()
        status['lifetime_failed_ML_tasks'] = status['lifetime_failed_ML_tasks'] + failed
        for network in output:
            if all(network[0]) and (network[1] > status['current_model_id']):
                print('New Model: {:04d}'.format(network[1]))
                status['current_model_id'] = network[1]
                status['current_molecule_id'] = 0
    
    # Construct analysis plots
    if (master_loop_iter % master_config.get('update_plots_every', 1000) == 0) and (analysis_plot is not None):
        analysis_input = build_input_dict(analysis_plot, all_configs)
        Process(target=analysis_plot, kwargs=analysis_input).start()
                
    print("### Active Learning Status at: " + time.ctime() + " ###")
    print("builder status:")
    builder_task_queue.print_status()
    print("sampling status:")
    sampler_task_queue.print_status()
    print("QM status:")
    QM_task_queue.print_status()
    print("ML status:")
    ML_task_queue.print_status()
    sys.stdout.flush()
    
    with open(master_config['status_path'], "w") as outfile:
        json.dump(status, outfile, indent=2)
    
    master_loop_iter += 1
    time.sleep(60)
	
