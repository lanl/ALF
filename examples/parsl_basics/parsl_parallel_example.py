###################################
# Step 1: Specify a configuration #
###################################

import os
import json

import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface

# For IC/NERSC:
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher

import time

import numpy as np

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

import alframework
from alframework.parsl_resource_configs.darwin import config_atdm_ml

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Load the Parsl config
parsl.load(config_atdm_ml)

#############################
# Step 2: Define Parsl apps #
#############################

from parsl import python_app, bash_app

#This is a quick task to figure out parsl parallelism
@python_app(executors=['alf_sampler_executor'])
def Simple_parsl_parallel():
    import os
    import time
    time.sleep(10)
    hostname = os.environ.get('HOSTNAME')
    parsl_rank = os.environ.get('PARSL_WORKER_RANK')
    return([hostname,parsl_rank])

###################################
# Step 3: Create a Parsl workflow #
###################################
class parsl_task_queue():
    def __init__(self):
        #Create a list 
        self.task_list = []
    
    def add_job(self,task):
        self.task_list.append(task)
        #self.task_list[-1].start()
        
    def get_completed_number(self):
        task_status = [task.done() for task in self.task_list]
        return(np.sum(task_status))
        
    def get_running_number(self):
        task_status = [task.running() for task in self.task_list]
        return(np.sum(task_status))
        
    def get_number(self):
        return(len(self.task_list))
    
    def get_queued_number(self):
        return(self.get_number()-self.get_running_number()-self.get_completed_number())
    
    def get_task_results(self):
        results_list = []
        for task in self.task_list:
            if task.done():
                results_list.append(task.result())
                del task
        return(results_list)

jobQueue = parsl_task_queue()

for i in range(12):
    jobQueue.add_job(Simple_parsl_parallel())

for i in range(15):
    print("Running: "+str(jobQueue.get_running_number()))
    print("Queue: "+str(jobQueue.get_queued_number()))
    print("Finished: "+str(jobQueue.get_completed_number()))
    time.sleep(2)
    
output = jobQueue.get_task_results()
print(output)