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
np.set_printoptions(threshold=np.inf)

import parsl
from alframework.tools.config import build_input_dict, coordinator_poll_interval, load_module_from_string, load_workflow_configs
from alframework.tools.data_shards import store_current_data
from alframework.tools.parsl_tasks import ParslTaskQueue, release_allocations_at_exit, stop_if_executor_failed
from alframework.tools.serialization import use_data_serializer
from alframework.tools.files import hold_lock
from alframework.tools import memory_sampler
from alframework.tools.memory import coordinator_huge_pages
from alframework.tools.progress_plots import ProgressPlots
from alframework.tools.h5_parallel import read_shard
from alframework.tools.molecules import MoleculesObject
from alframework.tools.bootstrap import next_bootstrap_ids
from alframework.tools.restarts import (
    STORED_AT_BUDGET_STOP, StatusFile, describe_shard, find_empty_directory, open_held_labels, reconcile_status,
    write_status,
)
from alframework.tools.label_validation import screen_labels
from alframework.tools.label_parts import LabelReceipt, hold_ready, remove_orphan_worker_parts
from alframework.tools.preflight import check_workflow, reapply_settings
from alframework.tools.retraining import SAMPLED_WITHOUT_LABEL, ready_to_retrain, retraining_settings
from alframework.tools.scaling import SCALING
from alframework.tools.builder_outputs import builder_outputs, builder_width
from alframework.tools.sampled_frames import SampledFrame
#import logging
#logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--master',required=True)
parser.add_argument('--test_builder',action='store_true',default=False)
parser.add_argument('--test_qm',action='store_true',default=False)
parser.add_argument('--test_sampler',action='store_true',default=False)
parser.add_argument('--test_ml',action='store_true',default=False)
parser.add_argument('--check_config', action='store_true', help='Validate configuration without starting the run.')
args = parser.parse_args()
debug = args.test_builder or args.test_qm or args.test_sampler or args.test_ml
if args.check_config and debug:
    parser.error('--check_config cannot be combined with a test mode.')

# The master configuration and the four it names, as every reload reads them too.
configs = load_workflow_configs(args.master)
master_config, builder_config, sampler_config = configs.master, configs.builder, configs.sampler
QM_config, ML_config = configs.QM, configs.ML

startup = check_workflow(master_config, builder_config, sampler_config, QM_config, ML_config,
                         debug=debug, test_sampler=args.test_sampler, test_qm=args.test_qm)
if args.check_config:
    print('Configuration check passed.')
    sys.exit(0)

# These settings change the running workflow. Apply them only after successful preflight.
coordinator_huge_pages(master_config)
hold_lock(master_config['status_path'] + '.lock')
coordinator_memory_sampler = memory_sampler.start(master_config)
use_data_serializer(startup.serializer)
release_allocations_at_exit(parsl.load(startup.parsl_config))

all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]
for config in all_configs:
    for entry in config:
        if entry.endswith('dir'):
            Path(config[entry]).mkdir(parents=True, exist_ok=True)

if master_config.get('plotting_utility') is not None:
    analysis_plot = load_module_from_string(master_config['plotting_utility'])
    progress_plots = ProgressPlots(master_config['plotting_utility'])
else:
    analysis_plot = None
run_report_updates = ProgressPlots('alframework.tools.run_report.live_report') if master_config.get('run_report') else None

scaling_settings, sizing_derived = startup.scaling, startup.sizing
poll_interval, label_budget, spare_seeds = startup.poll_interval, startup.label_budget, startup.spare_seeds
trim_snapshot_memory = startup.trim_snapshot_memory
# When to retrain (alframework.tools.retraining); a reload can change it.
retraining = startup.retraining
builder_task, sampler_task, qm_task, ml_task = startup.builder, startup.sampler, startup.qm, startup.ml
sampling, labeling = startup.sampling, startup.labeling
sampling.start_frame_spool()
sampler_task_queue, QM_task_queue = sampling.queue, labeling.queue
frame_retirement = sampling.frame_retirement
label_handoff, reference_settings = startup.label_handoff, startup.reference
ML_task_queue = ParslTaskQueue()
builder_task_queue = ParslTaskQueue()

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
storage_batch_size = master_config.get('label_storage_batch_size')
# Optional: how long each phase of each loop pass takes (alframework.tools.loop_timing).
loop_timer = startup.loop_timer
# Optional: rewrite the status file only when it changes (write_status_on_change).
status_file = StatusFile(only_on_change=startup.rewrite_status_on_change)


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
    snapshot = master_config['h5_path'].format(status['current_h5_id'])
    started = time.perf_counter()
    stored = held.commit(snapshot, master_config['properties_list'], attributes)
    if stored:
        print(f'Stored {snapshot}: {describe_shard(snapshot, stored)}, in {time.perf_counter() - started:.1f} s',
              flush=True)
    if frame_retirement is not None:
        frame_retirement.snapshot_published()
    status['lifetime_stored_labels'] = status.get('lifetime_stored_labels', 0) + stored
    status['current_h5_id'] = status['current_h5_id'] + 1
    status[SAMPLED_WITHOUT_LABEL] = 0
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
    store_current_data(qm_test_path, queue_output, master_config['properties_list'],
                       label_units=master_config.get('label_units'))
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
                # IDs per builder task: maximum_builder_structures for a builder that takes a list, else one.
                width = builder_width(builder_task, master_config)
                while builder_task_queue.get_number() * width < master_config['parallel_samplers']:
                    moleculeids, next_molecule_id = next_bootstrap_ids(
                        status['current_molecule_id'], bootstrap_limit, width, finite_bootstrap, bootstrap_skip)
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
            reloaded = load_workflow_configs(args.master)
            if reloaded.master.get(SCALING) != scaling_settings and not scaling_edit_reported:
                print(f'The master configuration\'s {SCALING} block changed; it is read when the coordinator starts, '
                      'so the run keeps its starting settings.', flush=True)
                scaling_edit_reported = True
            # The sizing and scaling blocks were read at startup: their settings apply to every reload, and an edit to
            # either waits for a restart.
            reapply_settings(reloaded, scaling_settings, sizing_derived)
            # Edits the running coordinators couldn't work with are refused, as they would be at startup.
            sampling.check_reloaded(reloaded.master)
            labeling.check_reloaded(reloaded.master)
            builder_width(builder_task, reloaded.master)
            retraining_new = retraining_settings(reloaded.master)
            poll_interval_new = coordinator_poll_interval(reloaded.master)
        except Exception as e:
            print("Failed to re-load configuration files; the run keeps its previous configuration:")
            print(e)
        else:
            master_config = reloaded.master
            poll_interval = poll_interval_new
            retraining = retraining_new
            builder_config = reloaded.builder
            sampler_config = sampling.apply_window(sampling.keep_startup_settings(reloaded.sampler),
                                                   status.get('lifetime_stored_labels', 0))
            QM_config = labeling.keep_startup_settings(reloaded.QM)
            ML_config = reloaded.ML
            all_configs = [master_config, builder_config, sampler_config, QM_config, ML_config]

    # Once the label budget is reached, start nothing new; only running training finishes.
    admitting = label_budget is None or status.get('lifetime_stored_labels', 0) < label_budget

    # Run more builders
    with loop_timer.phase('builders'):
        if admitting and (QM_task_queue.get_queued_number() < master_config['target_queued_QM']) and \
                (QM_task_queue.get_number() + held.successful() < master_config.get('maximum_completed_QM', 1e12)):
            width = builder_width(builder_task, master_config)
            while sampler_task_queue.get_number() + builder_task_queue.get_number() * width \
                    < master_config['parallel_samplers']:
                moleculeids = ['mol-{:04d}-{:010d}'.format(status['current_model_id'], it_ind) for it_ind in
                               range(status['current_molecule_id'], status['current_molecule_id'] + width)]
                task_input = build_input_dict(builder_task.func,
                                              [{"moleculeid": 'mol-{:04d}-{:010d}'.format(status['current_model_id'], status['current_molecule_id']),
                                                "moleculeids": moleculeids, "builder_config": builder_config},
                                               *all_configs, status],
                                              raise_on_fail=True)
                builder_task_queue.add_task(builder_task(**task_input))
                status['current_molecule_id'] = status['current_molecule_id'] + width

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
            lost = sampling.take_failed_structures(failed)
            status['lifetime_failed_sampler_structures'] = status.get('lifetime_failed_sampler_structures', 0) + lost
            for structure in sampler_results: #may need [0]
                assert isinstance(structure, (MoleculesObject, SampledFrame)), 'invalid sampled frame'
            # Without cohort selection this is every result whose atoms is set, as before.
            for structure in sampling.harvest(sampler_results):
                labeling.submit(structure, [{"QM_config": QM_config}, master_config, *all_configs, status])
            # Structures whose sampling ended without a frame to label, for retraining that counts them.
            status[SAMPLED_WITHOUT_LABEL] = status.get(SAMPLED_WITHOUT_LABEL, 0) + sampling.take_without_label(lost)
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
        if admitting and ML_task_queue.get_number() < 1 and ready_to_retrain(
                retraining, converged=held.converged(), finished=held.finished(),
                sampled_without_label=status.get(SAMPLED_WITHOUT_LABEL, 0)):
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
	
