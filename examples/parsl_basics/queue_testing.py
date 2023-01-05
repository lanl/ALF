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

from alframework.tools.tools import parsl_task_queue

import time

import numpy as np

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

import alframework
from alframework.parsl_resource_configs.chicoma import config_debug

# Uncomment this to see logging info
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Load the Parsl config
parsl.load(config_debug)

#############################
# Step 2: Define Parsl apps #
#############################

from parsl import python_app, bash_app

#This is a quick task to figure out parsl parallelism
@python_app(executors=['alf_sampler_executor'])
def Simple_parsl_parallel():
    import os
    import time
    import numpy
    randNum = numpy.random.rand()
    if randNum <.5:
        raise RuntimeError('You got Unlucky')
    time.sleep(10)
    hostname = os.environ.get('HOSTNAME')
    parsl_rank = os.environ.get('PARSL_WORKER_RANK')
    return([hostname,parsl_rank])

###################################
# Step 3: Create a Parsl workflow #
###################################


jobQueue = parsl_task_queue()

for i in range(12):
    jobQueue.add_task(Simple_parsl_parallel())

for i in range(30):
    task_status = jobQueue.get_task_status()
    print(task_status)
    time.sleep(2)
results = jobQueue.get_task_results()
print(results)