###################################
# Step 1: Specify a configuration #
###################################

import os
import json
import time
import sys
from pathlib import Path
import numpy as np
import argparse
import inspect
np.set_printoptions(threshold=np.inf)

import parsl
from alframework.tools.config import (
    build_input_dict, coordinator_poll_interval, load_config_file, load_module_from_string, stored_label_budget,
)
from alframework.tools.data_shards import store_current_data
from alframework.tools.parsl_tasks import ParslTaskQueue, release_allocations_at_exit, stop_if_executor_failed
from alframework.tools.serialization import data_serializer, use_data_serializer
from alframework.tools.files import hold_lock
from alframework.tools import memory_sampler
from alframework.tools.memory import coordinator_huge_pages, snapshot_memory_trim
from alframework.tools.progress_plots import ProgressPlots
from alframework.tools.h5_parallel import read_shard
from alframework.tools.molecules import MoleculesObject
from alframework.tools.bootstrap import bootstrap_spare_count, next_bootstrap_ids
from alframework.tools.restarts import (
    STORED_AT_BUDGET_STOP, StatusFile, find_empty_directory, label_h5_options, open_held_labels, reconcile_status,
    status_on_change_setting, write_status,
)
from alframework.tools.labeling_coordinator import LabelingCoordinator
from alframework.tools.label_validation import screen_labels
from alframework.tools.label_parts import LabelReceipt, hold_ready, incoming_directory, remove_orphan_worker_parts
from alframework.tools.loop_timing import coordinator_loop_timer
from alframework.tools.capabilities import validate_task_config
from alframework.tools.capacity import capacity_report, task_slots
from alframework.samplers.acquisition import SINGLE_MEMBER, uses_spread
from alframework.tools.sizing import apply_sizing, derive_sizing, describe as describe_sizing, sizing_block
from alframework.tools.scaling import SCALING, apply_scaling, describe as describe_scaling, large_run_warning
from alframework.tools.builder_outputs import builder_outputs
from alframework.tools.sampler_coordinator import SamplerCoordinator
from alframework.tools.sampled_frames import SampledFrame, data_by_reference_settings
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

# Optional: the master configuration's scaling block, one place for the settings that change how ALF runs at
# scale, passed to the configurations that use them (alframework.tools.scaling). Read once, here.
scaling_settings = apply_scaling(master_config, sampler_config, ML_config)
if scaling_settings:
    print(describe_scaling(scaling_settings), flush=True)
poll_interval = coordinator_poll_interval(master_config)
trim_snapshot_memory = snapshot_memory_trim(master_config)
# Before anything large is allocated or any child process (such as Parsl's interchange) starts, which inherits it.
coordinator_huge_pages(master_config)

# All configs
all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]

# One coordinator per run, in every mode, since each one rewrites status. Taken before any
# allocation is requested, so a refused second coordinator holds nothing.
hold_lock(master_config['status_path'] + '.lock')
# Optional: the memory of this process and its children (such as Parsl's interchanges), over the run.
coordinator_memory_sampler = memory_sampler.start(master_config)

# Define queues (the QM queue is created with the QM task below):
ML_task_queue = ParslTaskQueue()
builder_task_queue = ParslTaskQueue()

if (args.test_builder or args.test_qm or args.test_sampler or args.test_ml) and 'parsl_debug_configuration' in master_config:
    parsl_configuration = load_module_from_string(master_config['parsl_debug_configuration'])
else: 
    parsl_configuration = load_module_from_string(master_config['parsl_configuration'])

executor_list = []
# Create List of parsl executors
for curExec in parsl_configuration.executors:
    executor_list.append(curExec.label)
parsl_executors = {executor.label: executor for executor in parsl_configuration.executors}

# Task data: Parsl's dill, or pickle first (parsl_data_serializer), before anything is submitted.
use_data_serializer(data_serializer(master_config))

# Load the Parsl config
release_allocations_at_exit(parsl.load(parsl_configuration))

if master_config.get('plotting_utility', None) is not None:
    # Loaded here for its parameters; each update runs in its own process (alframework.tools.progress_plots).
    analysis_plot = load_module_from_string(master_config['plotting_utility'])
    progress_plots = ProgressPlots(master_config['plotting_utility'])
else:
    analysis_plot = None
# Optional: the run report (alframework.tools.run_report), refreshed with the progress plots in its own process.
run_report_updates = ProgressPlots('alframework.tools.run_report.live_report') if master_config.get('run_report') \
    else None

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
# Every task can check its own configuration (a validate_config attribute), so a mistake stops ALF here.
validate_task_config(builder_task, builder_config, master_config)

# Sampler
sampler_task = load_module_from_string(master_config['sampler_task'])
for cur_Exec in sampler_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        sampler_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))
validate_task_config(sampler_task, sampler_config, master_config)
# Optional: the master configuration's sizing block states what each task holds and how many labels a round adds,
# and the batch sizes and queue limits follow (alframework.tools.sizing). Derived once, here.
sizing_settings = sizing_block(master_config)
sizing_derived = derive_sizing(
    sizing_settings, sampler_slots=task_slots(sampler_task.executors, parsl_executors),
    builder_batches='moleculeids' in inspect.signature(builder_task.func).parameters,
    sampler_batches='molecule_objects' in inspect.signature(sampler_task.func).parameters)
apply_sizing(sizing_derived, master_config, sampler_config, QM_config)
if sizing_settings:
    print(describe_sizing(sizing_settings, sizing_derived), flush=True)
large_run = large_run_warning(master_config, sampler_config, sampler_task)
if large_run:
    print('Warning: ' + large_run, flush=True)
# Single-structure samplers submit one task per structure as before; see sampler_coordinator.
sampling_master = master_config
if args.test_sampler and 'data_by_reference' in master_config:
    # A one-structure sampler test returns its frame directly, so QM can be tested on it.
    sampling_master = dict(master_config)
    sampling_master.pop('data_by_reference')
sampling = SamplerCoordinator(sampler_task, master_config['sampler_task'], sampler_config, sampling_master)
sampler_task_queue = sampling.queue
frame_retirement = sampling.frame_retirement
print("Sampling: " + sampling.describe())

# QM
qm_task = load_module_from_string(master_config['QM_task'])
for cur_Exec in qm_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        qm_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))
# Optional: data_by_reference labels. QM workers write their labels as HDF5 parts beside the label store and
# return receipts; the coordinator records the receipts and merges the parts before each snapshot is published.
# A --test_qm check prints the labeled structure, so it labels the usual way.
reference_settings = data_by_reference_settings(master_config) or {}
label_handoff = None
if reference_settings.get('labels') and not args.test_qm:
    label_handoff = dict(directory=str(incoming_directory(Path(master_config['h5_path'].format(0)).resolve())),
                         properties=master_config['properties_list'],
                         schema=json.dumps(master_config['properties_list'], sort_keys=True),
                         h5_options=label_h5_options(master_config))
# Without a 'batching' block in QM_config, labeling submits one QM task per structure as before.
labeling = LabelingCoordinator(qm_task, master_config['QM_task'], QM_config, master_config,
                               sampled_frame_handoff=sampling.frame_directory is not None,
                               label_handoff=label_handoff)
QM_task_queue = labeling.queue
print("QM labeling: " + labeling.describe())

#ML
ml_task = load_module_from_string(master_config['ML_task'])
for cur_Exec in ml_task.executors:
    if cur_Exec.replace('_executor', '_standby_executor') in executor_list:
        ml_task.executors.append(cur_Exec.replace('_executor', '_standby_executor'))
# The QM task's is called by LabelingCoordinator; the ML task's runs before the bootstrap is labeled.
validate_task_config(ml_task, ML_config, master_config)

# What the run keeps in flight against what its Parsl executors can run, with warnings where they disagree.
# Ensemble tasks that submit one app per member (join apps) are counted per member.
ml_members = ML_config.get('n_models') if 'n_models' in ML_config else (ML_config.get('fine_tune') or {}).get('n_models')
# One member has no spread: refuse it here, before the bootstrap, when the run's acquisition reads the spread. (A
# sampler that doesn't use ALF's acquisition settings has nothing to check; its tasks check what they load.)
if ml_members == 1:
    try:
        reads_spread = uses_spread(sampler_config)
    except ValueError:
        reads_spread = False
    if reads_spread:
        raise ValueError(SINGLE_MEMBER.format(what='The ML configuration (n_models 1)'))
capacity_lines, capacity_warnings = capacity_report(
    master_config, parsl_executors,
    builder_labels=builder_task.executors, builder_width=master_config.get('maximum_builder_structures', 1),
    builder_batches=master_config.get('maximum_builder_structures', 1) > 1,
    sampler_labels=sampler_task.executors, sampler_width=sampling.settings['batch_size'] if sampling.batched else 1,
    qm_labels=qm_task.executors, qm_width=labeling.settings['batch_size'] if labeling.batched else 1,
    ml_labels=[label for label in ('alf_ML_executor', 'alf_ML_standby_executor') if label in parsl_executors],
    n_models=ml_members if getattr(ml_task, 'join', False) else None, member_nodes=ML_config.get('member_nodes', 1))
print('Capacity:\n  ' + '\n  '.join(capacity_lines), flush=True)
for warning in capacity_warnings:
    print('Warning: ' + warning, flush=True)

##########################################
## Step 3: Evaluate restart possibilites #
##########################################
restarting = os.path.exists(master_config['status_path'])
if restarting:
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

held = open_held_labels(master_config, status['current_h5_id'])
if label_handoff is not None and not reference_settings.get('keep_files'):
    orphans = remove_orphan_worker_parts(held)
    if orphans:
        print(f'Restart: removed {orphans} label parts written by QM workers but never recorded.', flush=True)
if restarting:
    for change in reconcile_status(status, master_config, held):
        print('Restart: ' + change)

write_status(master_config['status_path'], status)

# Read once, so that a mistaken edit can't stop or crash a long run.
label_budget = stored_label_budget(master_config)
spare_seeds = bootstrap_spare_count(master_config)
storage_batch_size = master_config.get('label_storage_batch_size')
# Optional: how long each phase of each loop pass takes (alframework.tools.loop_timing).
loop_timer = coordinator_loop_timer(master_config)
# Optional: rewrite the status file only when it changes (write_status_on_change).
status_file = StatusFile(only_on_change=status_on_change_setting(master_config))


def end_loop_pass(loop, delay):
    """Record the end of a timed loop pass, with how much work was in flight."""
    if loop_timer.enabled:
        loop_timer.count(builder_queue=builder_task_queue.get_number(), sampler_queue=sampler_task_queue.get_number(),
                         qm_queue=QM_task_queue.get_number(), ml_queue=ML_task_queue.get_number(),
                         held_labels=held.successful())
        loop_timer.end_pass(loop, delay)


def hold_finished_labels(*, drain=False):
    """Move finished QM work out of memory, so a crash can't lose it. With bounded storage, a pass writes as
    many full parts as are waiting, within a time limit (``label_parts.hold_ready``); ``drain`` writes them all."""
    properties = held.properties if storage_batch_size is not None else master_config['properties_list']

    def take_one_part():
        if frame_retirement is None:
            results, failed = QM_task_queue.get_task_results(max_items=storage_batch_size)
        else:
            results, failed, succeeded_files, failed_files = labeling.take_finished_with_references(storage_batch_size)
        receipts = [result for result in results if isinstance(result, LabelReceipt)]
        if receipts:
            results = [result for result in results if not isinstance(result, LabelReceipt)]
        screen_labels(results, properties)
        loop_timer.count(results_held=len(results) + sum(len(receipt) for receipt in receipts))
        lost = labeling.take_failed_structures(failed)
        status['lifetime_failed_QM_tasks'] = status['lifetime_failed_QM_tasks'] + failed
        status['lifetime_failed_QM_structures'] = status.get('lifetime_failed_QM_structures', 0) + lost
        if receipts:
            held.adopt(receipts)
        held.hold(results, lost)
        if frame_retirement is not None:
            frame_retirement.labels_held(succeeded_files, failed_files, durable=storage_batch_size is not None)

    hold_ready(take_one_part, QM_task_queue.get_completed_number, storage_batch_size, drain=drain)


def store_held_labels(**attributes):
    """Store every held label as the next shard. The caller saves status afterwards.

    Publish the shard before advancing status. A restart that finds it ahead of status moves past it.
    """
    hold_finished_labels(drain=True)
    stored = held.commit(master_config['h5_path'].format(status['current_h5_id']),
                         master_config['properties_list'], attributes)
    if frame_retirement is not None:
        frame_retirement.snapshot_published()
    status['lifetime_stored_labels'] = status.get('lifetime_stored_labels', 0) + stored
    status['current_h5_id'] = status['current_h5_id'] + 1
    held.select_snapshot(master_config['h5_path'].format(status['current_h5_id']))
    if stored and trim_snapshot_memory is not None:
        trim_snapshot_memory()
    return stored

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
    outputs = builder_outputs(queue_output[0][0], master_config['builder_task'])
    if not outputs:
        raise ValueError(f'Builder {master_config["builder_task"]} returned no structures for --test_builder.')
    test_configuration = outputs[0]
    print("Builder testing returned:")
    print(test_configuration)
    testing = True
    
if args.test_sampler:
    #Check that there is a model available
    next_model = find_empty_directory(master_config['model_path'])
    if status['current_model_id'] < 0:
        raise RuntimeError("Need to train model before testing sampling")
    print(master_config['model_path'].format(status['current_model_id']))
    sampling.submit(test_configuration, [{"sampler_config": sampler_config}, *all_configs, status])
    sampling.submit_ready([{"sampler_config": sampler_config}, *all_configs, status], flush=True)
    sampled_configuration = sampler_task_queue.task_list[0].result()
    queue_output = sampler_task_queue.get_task_results()
    test_configuration = queue_output[0][0]
    assert isinstance(test_configuration, MoleculesObject), 'test_configuration must be a MoleculesObject instance'
    print("Sampler testing returned:")
    print(test_configuration)
    testing = True

#def ase_calculator_task(input_system,configuration_list,directory,command,properties=['energy','forces']):
if args.test_qm:
    labeling.submit(test_configuration, [{"QM_config": QM_config}, *all_configs, status])
    labeling.submit_ready([{"QM_config": QM_config}, *all_configs, status], flush=True)
    qm_result = QM_task_queue.task_list[0].result()
    queue_output,failed = QM_task_queue.get_task_results()
    sys.stdout.write('Job Failed: {:d}'.format(failed))
    test_configuration = queue_output[0]
    screen_labels([test_configuration], master_config['properties_list'])
    assert isinstance(test_configuration, MoleculesObject), 'test_configuration must be a MoleculesObject instance'
    print("Ensure the following data matches that in the QM output and test.h5 after unit conversion:")
    sys.stdout.write('Atomic Numbers: ' + np.array_str(test_configuration[1].get_atomic_numbers()) + '\n')
    sys.stdout.write('Coordinates: \n' + np.array_str(test_configuration[1].get_positions(),precision=4) + '\n')
    sys.stdout.write('PBCs: ' + np.array_str(test_configuration[1].get_pbc()) + '\n')
    if any(test_configuration[1].get_pbc()):
        sys.stdout.write('Cell: \n' + np.array_str(test_configuration[1].get_cell(),precision=4) + '\n')
    if not test_configuration.check_convergence():
        # A failed label has no results to show; report why it failed, from the labeler's metadata.
        failure = test_configuration.get_metadata()
        reason = failure.get('qm_error') or failure.get('labeling_error') or 'the QM task recorded no reason'
        stage = failure.get('qm_failed_stage')
        sys.stdout.write('\nLabeling failed' + (f' at stage {stage!r}' if stage else '') + f': {reason}\n')
        sys.exit(1)
    for property_key in list(master_config['properties_list']):
        if isinstance(test_configuration[2][property_key], np.ndarray):
            sys.stdout.write(property_key + ':\n' + np.array_str(test_configuration[2][property_key],precision=4) + '\n')
        else:
            sys.stdout.write(property_key + ': ' + str(test_configuration[2][property_key]) + '\n')
    qm_test_path = master_config['master_directory'] + '/qm_test.h5'
    if os.path.exists(qm_test_path):
        os.remove(qm_test_path)
    store_current_data(qm_test_path, queue_output, master_config['properties_list'])
    test_db = read_shard(qm_test_path, workers=1)

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
    status_file.write(master_config['status_path'], status)
    ml_result = ML_task_queue.task_list[0].result()
    queue_output = ML_task_queue.get_task_results()
    returned_models = queue_output[0]
    print("ML ensemble training status:")
    print(returned_models)
    for network in returned_models:
        if all(network[0]) and (network[1] > status['current_model_id']):
            print('New Model: {:04d}'.format(network[1]))
            status['current_model_id'] = network[1]
    status_file.write(master_config['status_path'], status)
    testing=True
    
if testing:
    exit()
    
########################
# Step 5: Bootstraping #
########################
    
#If there is no data and no models, start boostrap jobs
if status['current_h5_id'] == 0 and status['current_model_id'] < 0:
    print("Building Bootstrap Set")
    finite_bootstrap = master_config.get('finite_bootstrap', False)
    # A restarted finite bootstrap rebuilds only the IDs it doesn't hold, which give the same structures.
    bootstrap_skip = held.molecule_ids() if finite_bootstrap else frozenset()

    def bootstrap_unfinished():
        if finite_bootstrap:
            # Only converged labels count, so each failed seed can be replaced by a spare.
            hold_finished_labels()
            return held.converged() < master_config['bootstrap_set']
        return held.successful() + QM_task_queue.get_exec_done_number() < master_config['bootstrap_set']

    # Seeds held by each running builder task, and seeds whose builder task was lost in this process.
    bootstrap_builder_seeds = {}
    lost_builder_seeds = 0

    def bootstrap_failures():
        """Seeds that didn't give a converged label: unconverged results, and QM and builder tasks lost in
        this process."""
        return held.successful() - held.converged() + held.failed_since_store + lost_builder_seeds

    def bootstrap_seeds_exhausted():
        """No more seeds can be built now, so a waiting partial batch will never fill."""
        return (finite_bootstrap
                and status['current_molecule_id'] >= master_config['bootstrap_set'] + min(spare_seeds, bootstrap_failures())
                and builder_task_queue.get_number() == 0)

    def spares_handed_out():
        return max(0, status['current_molecule_id'] - master_config['bootstrap_set'])

    while bootstrap_unfinished():
        # A spare seed is handed out only once a label has failed, so a clean bootstrap labels no spares.
        bootstrap_limit = master_config['bootstrap_set'] + min(spare_seeds, bootstrap_failures())
        if (bootstrap_seeds_exhausted()
                and QM_task_queue.get_number() == 0
                and not (getattr(labeling, 'buffer', None) and len(labeling.buffer))):
            if held.converged() == 0:
                raise RuntimeError('Finite bootstrap: no seed gave a converged label, so there is nothing to train on; '
                                   'inspect failed work')
            print('Finite bootstrap: {} of {} spare seeds used and {} seeds failed, so it trains on {} labels, fewer '
                  'than bootstrap_set ({})'.format(spares_handed_out(), spare_seeds, bootstrap_failures(),
                                                    held.converged(), master_config['bootstrap_set']), flush=True)
            break
        with loop_timer.phase('builders'):
            if QM_task_queue.get_queued_number() < master_config['target_queued_QM']:
                while builder_task_queue.get_number() * master_config.get('maximum_builder_structures', 1) < master_config['parallel_samplers']:
                    moleculeids, next_molecule_id = next_bootstrap_ids(
                        status['current_molecule_id'], bootstrap_limit,
                        master_config.get('maximum_builder_structures', 1), finite_bootstrap, bootstrap_skip)
                    status['current_molecule_id'] = next_molecule_id
                    if not moleculeids:
                        break
                    task_input = build_input_dict(builder_task.func,
                                                  [{"moleculeid": moleculeids[0],
                                                    "moleculeids": moleculeids, "builder_config": builder_config},
                                                   *all_configs, status],
                                                  raise_on_fail=True)
                    task = builder_task(**task_input)
                    bootstrap_builder_seeds[task] = len(moleculeids)
                    builder_task_queue.add_task(task)

        with loop_timer.phase('builder_harvest'):
            if (builder_task_queue.get_exec_done_number() > master_config['minimum_QM']
                    or builder_task_queue.get_failed_number() > 0
                    or (master_config.get('finite_bootstrap', False) and builder_task_queue.get_completed_number() > 0)):
                builder_results, failed = builder_task_queue.get_task_results()
                status['lifetime_failed_builder_tasks'] = status['lifetime_failed_builder_tasks'] + failed
                # A lost builder's seeds count as failed, so spares can replace them.
                lost_builder_seeds += sum(bootstrap_builder_seeds.get(task, 0) for task in builder_task_queue.last_failed)
                bootstrap_builder_seeds = {task: seeds for task, seeds in bootstrap_builder_seeds.items()
                                           if any(task is running for running in builder_task_queue.task_list)}
                for result in builder_results:
                    for structure in builder_outputs(result, master_config['builder_task']):
                        labeling.submit(structure, [{"QM_config": QM_config}, *all_configs, status])
                builder_results = result = structure = None
        with loop_timer.phase('qm_submit'):
            # Once every seed is built, a partial batch can't fill, so send it ("full_only" would wait forever).
            labeling.submit_ready([{"QM_config": QM_config}, *all_configs, status], flush=bootstrap_seeds_exhausted())
        with loop_timer.phase('hold'):
            hold_finished_labels()

        with loop_timer.phase('print_status'):
            print("### Bootstraping Learning Status at: " + time.ctime() + " ###")
            if finite_bootstrap:
                print("bootstrap: {} of {} labels converged, {} failed, {} of {} spare seeds handed out".format(
                    held.converged(), master_config['bootstrap_set'], bootstrap_failures(),
                    spares_handed_out(), spare_seeds))
            print("builder status:")
            builder_task_queue.print_status()
            print("QM status:")
            QM_task_queue.print_status()

        with loop_timer.phase('write_status'):
            status_file.write(master_config['status_path'], status)
        with loop_timer.phase('executor_check'):
            stop_if_executor_failed(parsl.dfk())

        delay = labeling.poll_delay(poll_interval)
        end_loop_pass('bootstrap', delay)
        time.sleep(delay)

    if finite_bootstrap:
        print("Finite bootstrap finished: {} of {} labels converged, {} seeds failed, {} spare seeds handed out".format(
            held.converged(), master_config['bootstrap_set'], bootstrap_failures(),
            spares_handed_out()), flush=True)
    print("Saving Bootstrap and training model")
    store_held_labels()
    
if status['current_model_id'] < 0:
    task_input = build_input_dict(ml_task.func,
                                  [{"ML_config": ML_config}, *all_configs, status],
                                  raise_on_fail=True)
    ML_task_queue.add_task(ml_task(**task_input))
    
    status['current_training_id'] = status['current_training_id'] + 1
    
    status_file.write(master_config['status_path'], status)
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
scaling_edit_reported = False
while True:
    with loop_timer.phase('reload_config'):
        #Re-load configurations, but watch for stupid errors
        try:
            # Load the master configuration:
            master_config_new = load_config_file(args.master)
            if 'master_directory' not in master_config_new:
                master_config_new['master_directory'] = None

            # Load the builder config:
            builder_config_new = load_config_file(master_config_new['builder_config_path'], master_config_new['master_directory'])

            # Load the sampler config:
            sampler_config_new = load_config_file(master_config_new['sampler_config_path'], master_config_new['master_directory'])

            # Load the QM config:
            QM_config_new = load_config_file(master_config_new['QM_config_path'], master_config_new['master_directory'])

            # Load the ML config:
            ML_config_new = load_config_file(master_config_new['ML_config_path'], master_config_new['master_directory'])

            # The sizing and scaling blocks were read at startup: their settings apply to every reload, and an edit to
            # either waits for a restart.
            apply_sizing(sizing_derived, master_config_new, sampler_config_new, QM_config_new)
            # The scaling block was read at startup; its settings apply to every reload, and an edit to it waits for
            # a restart.
            if master_config_new.get(SCALING) != scaling_settings and not scaling_edit_reported:
                print(f'The master configuration\'s {SCALING} block changed; it is read when the coordinator starts, '
                      'so the run keeps its starting settings.', flush=True)
                scaling_edit_reported = True
            apply_scaling(master_config_new, sampler_config_new, ML_config_new, scaling_settings or {},
                          check_tasks=False)
            poll_interval_new = coordinator_poll_interval(master_config_new)
        except Exception as e:
            print("Failed to re-load configuration files:")
            print(e)
        else:
            master_config = master_config_new
            poll_interval = poll_interval_new
            builder_config = builder_config_new
            sampler_config = sampling.apply_window(sampling.keep_startup_settings(sampler_config_new),
                                                   status.get('lifetime_stored_labels', 0))
            QM_config = labeling.keep_startup_settings(QM_config_new)
            ML_config = ML_config_new
            all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]

    # Once the label budget is reached, start nothing new; only running training finishes.
    admitting = label_budget is None or status.get('lifetime_stored_labels', 0) < label_budget

    # Run more builders
    with loop_timer.phase('builders'):
        if admitting and (QM_task_queue.get_queued_number() < master_config['target_queued_QM']) and \
                (QM_task_queue.get_number() + held.successful() < master_config.get('maximum_completed_QM', 1e12)):
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

    # Harvest failed builders too, so they free capacity for replacement tasks.
    with loop_timer.phase('builder_harvest'):
        if admitting and builder_task_queue.get_completed_number() > 0:
            structure_list, failed = builder_task_queue.get_task_results()
            status['lifetime_failed_builder_tasks'] = status['lifetime_failed_builder_tasks'] + failed
            # Douple loop to facilitate possiblitiy of multiple sctructures returned by builder
            for result in structure_list:
                for structure in builder_outputs(result, master_config['builder_task']):
                    sampling.submit(structure, [{"sampler_config": sampler_config}, *all_configs, status])
            # This loop runs at module level, so these names would keep the last harvest alive until the next one.
            structure_list = result = structure = None
    with loop_timer.phase('sampler_submit'):
        if admitting:
            sampling.submit_ready([{"sampler_config": sampler_config}, *all_configs, status])

    # Run more QM
    with loop_timer.phase('sampler_harvest'):
        sampling.collect_frames()
        if admitting and sampler_task_queue.get_completed_number() > master_config['minimum_QM']:
            sampler_results, failed = sampler_task_queue.get_task_results()
            status['lifetime_failed_sampler_tasks'] = status['lifetime_failed_sampler_tasks'] + failed
            status['lifetime_failed_sampler_structures'] = status.get('lifetime_failed_sampler_structures', 0) + sampling.take_failed_structures(failed)
            for structure in sampler_results: #may need [0]
                assert isinstance(structure, (MoleculesObject, SampledFrame)), 'invalid sampled frame'
            # Without cohort selection this is every result whose atoms is set, as before.
            for structure in sampling.harvest(sampler_results):
                labeling.submit(structure, [{"QM_config": QM_config}, master_config, *all_configs, status])
            # Labeling holds what it needs; releasing these lets each structure go once its QM task finishes,
            # instead of the whole harvest staying in memory for the rest of the run.
            sampler_results = structure = None
    with loop_timer.phase('qm_submit'):
        if admitting:
            labeling.submit_ready([{"QM_config": QM_config}, master_config, *all_configs, status])
    # Including during training, when labels can't go into h5_dir.
    with loop_timer.phase('hold'):
        hold_finished_labels()

    # Train more models
    with loop_timer.phase('store'):
        if admitting and (held.converged() > master_config['save_h5_threshold']) and (ML_task_queue.get_number() < 1):
            store_held_labels()
            task_input = build_input_dict(ml_task.func,
                                          [{"ML_config": ML_config}, *all_configs, status],
                                          raise_on_fail=True)
            ML_task_queue.add_task(ml_task(**task_input))
            status['current_training_id'] = status['current_training_id'] + 1

    # Update Model
    with loop_timer.phase('ml'):
        if ML_task_queue.get_completed_number() > 0:
            output, failed = ML_task_queue.get_task_results()
            status['lifetime_failed_ML_tasks'] = status['lifetime_failed_ML_tasks'] + failed
            for network in output:
                if all(network[0]) and (network[1] > status['current_model_id']):
                    print('New Model: {:04d}'.format(network[1]))
                    status['current_model_id'] = network[1]
                    status['current_molecule_id'] = 0

    # Construct analysis plots
    with loop_timer.phase('plots'):
        if (master_loop_iter % master_config.get('update_plots_every', 1000) == 0) and (analysis_plot is not None):
            analysis_input = build_input_dict(analysis_plot, all_configs)
            progress_plots.update(analysis_input)
        if run_report_updates is not None and master_loop_iter % master_config.get('update_plots_every', 1000) == 0:
            run_report_updates.update({'run_dir': os.getcwd(), 'out_dir': master_config.get('run_report_dir'),
                                       'master_config_path': os.path.abspath(args.master)})

    with loop_timer.phase('print_status'):
        print("### Active Learning Status at: " + time.ctime() + " ###")
        print("builder status:")
        builder_task_queue.print_status()
        print("sampling status:")
        sampler_task_queue.print_status()
        if sampling.status() is not None:
            print("cohorts: " + json.dumps(sampling.status()))
        print("QM status:")
        QM_task_queue.print_status()
        print("ML status:")
        ML_task_queue.print_status()
        sys.stdout.flush()

    with loop_timer.phase('write_status'):
        status_file.write(master_config['status_path'], status)
    with loop_timer.phase('retire_frames'):
        if frame_retirement is not None:
            frame_retirement.checkpointed()
    with loop_timer.phase('executor_check'):
        stop_if_executor_failed(parsl.dfk())

    if not admitting and ML_task_queue.get_number() == 0:
        # Finished labels were paid for, so keep them, although no model trained on them.
        with loop_timer.phase('store'):
            beyond_budget = store_held_labels(**{STORED_AT_BUDGET_STOP: 'true'}) if held.successful() else 0
        status_file.write(master_config['status_path'], status)
        print("Stopping: stored {} labels (maximum_stored_labels is {}), including {} finished labels stored at the "
              "stop without training. Dropped {} builder, {} sampler, and {} unfinished QM structures.".format(
                  status['lifetime_stored_labels'], label_budget, beyond_budget, builder_task_queue.get_number(),
                  sampler_task_queue.get_number(), QM_task_queue.get_number()))
        sys.stdout.flush()
        end_loop_pass('active', 0)
        break

    master_loop_iter += 1
    delay = labeling.poll_delay(sampling.poll_delay(poll_interval))
    end_loop_pass('active', delay)
    time.sleep(delay)

# Release the allocations; a run without a label budget never reaches this.
parsl.dfk().cleanup()
sampling.cleanup_frames()
	
