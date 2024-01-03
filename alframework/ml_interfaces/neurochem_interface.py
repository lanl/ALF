import numpy as np
import os
import shutil

import anitraintools as alt

from ase_interface import aniensloader
from ase_interface import ANIENS, batchedensemblemolecule

from alframework.tools.tools import build_input_dict

from parsl import python_app, bash_app

import time

class NeuroChemTrainer():
    def __init__(self, ensemble_size, gpuids, force_training=True, periodic=False, rmhighe=False, rmhighf=False, build_test=True, remove_existing=False):
        self.ensemble_size = ensemble_size
        self.force_training = force_training
        self.periodic = periodic
        self.gpuids = gpuids
        self.rmhighe = rmhighe
        self.rmhighf = rmhighf
        self.build_test = build_test
        self.remove_existing = remove_existing

    def train_models(self, tparam):
        print('Trainer:')
        print(tparam['ensemble_path'], tparam['data_store'], tparam['seed'])
        
        if os.path.isdir(tparam['ensemble_path']):
            if self.remove_existing:
                shutil.rmtree(tparam['ensemble_path'])
                os.mkdir(tparam['ensemble_path'])
            else: 
                raise RuntimeError("model directory already exists: " + tparam['ensemble_path'])
        else:
            os.mkdir(tparam['ensemble_path'])
                

        ndir = tparam['ensemble_path']

        #f = open('TRAIN-'+str(tparam['ids'][0]), 'w')
        #f.write(ndir+' '+tparam['data_store']+'\n')
        #f.close()

        # Setup AEV parameters
        aevparams  = tparam['aev_params']
        prm = alt.anitrainerparamsdesigner(aevparams['elements'],
                                           aevparams['NRrad'],
                                           aevparams['Rradcut'],
                                           aevparams['NArad'],
                                           aevparams['NAang'],
                                           aevparams['Aradcut'],
                                           aevparams['x0'])
        prm.create_params_file(ndir)

        # input parameters
        iptparams = tparam['input_params']
        ipt = alt.anitrainerinputdesigner()
        ipt.set_parameter('atomEnergyFile', 'sae_linfit.dat')
        ipt.set_parameter('sflparamsfile', prm.get_filename())

        for key in iptparams.keys():
            ipt.set_parameter(key, str(iptparams[key]))

        # Set network layers
        netparams = tparam['layers']
        for element_key in netparams.keys():
            for layer_params in netparams[element_key]:
                ipt.add_layer(element_key, layer_params)

        netdict = {'cnstfile': ndir + '/' + prm.get_filename(),
                   'saefile': ndir + '/sae_linfit.dat',
                   'iptsize': prm.get_aev_size(),
                   'atomtyp': prm.params['elm']}

        np.random.seed(tparam['seed'])
        local_seeds = np.random.randint(0, 2 ** 32, size=2)
        print('local seeds:',local_seeds)

        # Declare the training class for all ranks
        ens = alt.alaniensembletrainer(ndir + '/',
                                       netdict,
                                       ipt,
                                       tparam['data_store'],
                                       self.ensemble_size, random_seed=local_seeds[0])
        #
        # Build training cache
        #16,1,1 should probably be exposed to the user, maybe.
        ens.build_strided_training_cache(16, 1, 1, build_test=self.build_test, Ekey='energy',
                                             forces=self.force_training, grad=False, Fkey='forces',
                                             dipole=False,
                                             rmhighf=self.rmhighf, rmhighe=self.rmhighe, pbc=self.periodic)

        # Train a single model, outside interface should handle ensembles?
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpuid)
        ens.train_ensemble(self.gpuids, self.remove_existing)

        all_nets, completed = alt.get_train_stats(self.ensemble_size,ndir + '/')

        return all_nets, completed

def NeuroChemCalculator(model_details):
    model_path = model_details['model_path']
    cns = [f for f in os.listdir(model_path) if '.params' in f][0]
    sae = 'sae_linfit.dat'
    Nn  = model_details['Nn']
    gpu = model_details['gpu']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return ANIENS(batchedensemblemolecule(model_path+'/'+cns, model_path+'/'+sae, model_path, Nn, 0))

#Gen
#configuration,h5_train_dir,ensemble_path,ensemble_index,remove_existing=False,h5_test_dir=None)
@python_app(executors=['alf_ML_executor'])
def train_ANI_model_task(ML_config,h5_dir,model_path,current_training_id,gpus_per_node,remove_existing=False,h5_test_dir=None):
    
    configuration = ML_config.copy()
    #nct = NeuroChemTrainer(ensemble_size, gpuids, force_training=True, periodic=False, rmhighe=False, rmhighf=False, build_test=True)
    nct_input = build_input_dict(NeuroChemTrainer.__init__,[{"gpuids":list(range(gpus_per_node))},configuration])
    #nct = NeuroChemTrainer(8,list(range(gpus_per_node)), force_training=True, periodic=True, rmhighe=False, rmhighf=False, build_test=True,remove_existing=remove_existing)
    nct = NeuroChemTrainer(**nct_input)
    #this is a little awkward
    configuration['ensemble_path'] = model_path.format(current_training_id)
    configuration['data_store'] = h5_dir
    configuration['seed'] = np.random.randint(1e8) #Change before production to a random number generated on the fly
    
    (all_nets, completed) = nct.train_models(configuration)
    
    #No need to return network parameters
    
    #return(all_nets, completed, current_training_id)
    return(completed, model_index)

