import os
import random
import subprocess
import tempfile
import numpy as np
from ase import Atoms
from ase.io import read, write
from copy import deepcopy
from parsl import python_app, bash_app
from alframework.tools.tools import system_checker
import pandas as pd
try:
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
except:
    raise Exception("pip install rdkit")
from tqdm.contrib.concurrent import process_map

def smiles2conf(inputdata):
    """
    Converts smiles string to 3d xyz data
    :input: (smiles, moleculeid)
    :return: filesystem location of xyz file
    """
    smiles, moleculeid, outpath = inputdata

    numConfs = 50
    finalNumConfs = 10
    rms_cutoff = 0.1

    # generate conformer with rule-based method
    m = Chem.AddHs(Chem.MolFromSmiles(smiles.strip()))
    for i, func in enumerate(['srETKDGv3', 'srETKDGv3', 'ETKDG', 'KDG', 'ETDG']):
        try: 
             ids = AllChem.EmbedMultipleConfs(
                  m, numConfs=numConfs, params=eval(f'AllChem.{func}()'))
        except:
            continue
        else:
            if len(list(ids))==0:
                print('Increasing numConfs to 500')
                numConfs = 500
                print(f'Using model {func} for {smiles}')
                break

    if len(list(ids))==0:
        raise RuntimeError('Error: Could not generate conformers')

    # optimize the conformers with MMFF94s force field
    res = AllChem.MMFFOptimizeMoleculeConfs(m, mmffVariant='MMFF94s')
   
    # filter similar conformers based on heavy atom RMSD
    mcp = Chem.RemoveHs(deepcopy(m))
    remove_list = [idx for idx, r in enumerate(res) if r[0] != 0]
    for i in range(len(ids)):
        if i in remove_list: continue
        for j in range(i):
            if j in remove_list or i == j: continue
            dist = Chem.rdMolAlign.GetBestRMS(mcp, mcp, i, j)
            if dist < rms_cutoff: remove_list.append(j)
                
    # select remaining unique conformers based on FF energy evaluation
    nconf = len(ids)
    energy_list = [r[1] for r in res]
    energies = np.array(energy_list)
    unique_ids = np.array([i for i in range(nconf) if i not in remove_list])
    if len(unique_ids) < 1: 
        # this happens when the geometry is generated but fails during MMFF94s optimization
        # we send one conformer on to xTB optimization
        print(smiles, moleculeid)
        unique_ids = np.array([nconf-1])
    unique_energies = energies[unique_ids]
    sorted_ids = unique_ids[np.argsort(unique_energies)]
    selected_ids = sorted_ids[:finalNumConfs]
    final_energies = energies[selected_ids]

    # reoptimize the conformer with GFN2-xTB SEQM method and save to disk
    for confId in selected_ids:
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            filename = 'xtb.xyz'
            Chem.MolToXYZFile(m, filename, confId=int(confId))
            runcmd = ['xtb', filename, '--opt', '--grad', '--ceasefiles']
            proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            outfile = os.path.join(outpath, f'{moleculeid}_{confId}.xyz')
            runcmd = ['cp', 'xtbopt.xyz', outfile]
            _ = subprocess.run(runcmd)
    
    return True

def test():

    smiles = 'CC(C)CC(C)(C)C'    
    molid = 'test'
    success = smiles2conf((smiles, molid))
    return success

if __name__ == '__main__':

    #test()
    #exit()

    outpath = os.path.join(os.getcwd(), 'solute')    
    os.makedirs(outpath, exist_ok=True)

    database = './database.txt'
    if not os.path.exists(database):
        os.system("wget https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt")

    names = ["compound_id",
             "SMILES",
             "iupac name (or alternative if IUPAC is unavailable or not parseable by OEChem)",
             "experimental value (kcal/mol)",
             "experimental uncertainty (kcal/mol)",
             "Mobley group calculated value (GAFF) (kcal/mol)",
             "calculated uncertainty (kcal/mol)",
             "experimental reference (original or paper this value was taken from)",
             "calculated reference",
             "text notes"]
    df = pd.read_csv('database.txt', delimiter=';', header=0, names=names, skiprows=2)
    smiles = df['SMILES'].values
    compound_ids = df['compound_id'].values

    inputdata = list(zip(smiles, compound_ids, [outpath]*len(smiles)))
    success = process_map(smiles2conf, inputdata, max_workers=8)

