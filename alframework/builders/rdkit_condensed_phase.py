# The builder code goes here
import glob
import random
import os
import numpy as np
from parsl import python_app, bash_app
import json
from pathlib import Path

import ase
from ase import Atoms
from ase import neighborlist
from ase.geometry.cell import complete_cell
from ase.io import cfg
from ase.io import read, write
from ase.data import chemical_symbols
from ase import units

from alframework.tools.tools import random_rotation_matrix
from alframework.tools.tools import build_input_dict
from alframework.tools.tools import system_checker
import random
from copy import deepcopy

import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem

#def ASEfromSMILES(SMILES):
#    ps = AllChem.ETKDGv2()
#    ps.useRandomCoords = True 
#    m = Chem.MolFromSmiles(SMILES)
#    m = Chem.AddHs(m)
#    specString=''
#    for a in m.GetAtoms():
#        specString = specString + chemical_symbols[a.GetAtomicNum()]
#    AllChem.EmbedMolecule(m,ps)
#    positions = m.GetConformer().GetPositions()
#    aseAtoms = Atoms(specString,positions=positions)
#    return(aseAtoms)

def ASEfromSMILES(SMILES,elongate=False,maxD=300,dD=10):
    #elongate can either be False, True, or number indicies
    m = Chem.MolFromSmiles(SMILES)
    m = Chem.AddHs(m)
    mLen = len(m.GetAtoms())
    specString=''
    for a in m.GetAtoms():
        specString = specString + chemical_symbols[a.GetAtomicNum()]
    AllChem.EmbedMolecule(m,useRandomCoords=True,ETversion=2,maxAttempts=5)
    positions = m.GetConformer().GetPositions()
    if elongate!=False:
        try: 
            for curD in np.arange(dD,maxD,dD):
                coordMap={0:Geometry.Point3D(0,0,0),mLen-1:Geometry.Point3D(float(curD),0,0)}
                AllChem.EmbedMolecule(m,ETversion=2,coordMap=coordMap)
                positions = m.GetConformer().GetPositions()
        except:
            print("Failed at {:.1f} distance".format(float(curD)))
    aseAtoms = Atoms(specString,positions=positions)
    return(aseAtoms)

@python_app(executors=['alf_sampler_executor'])
def rdkit_condensed_phase_builder_task(moleculeid,builder_config,cell_range,Rrange,solute_molecule_options,solvent_molecules):
    """
    Elements in  builder parameters
        molecule_library_path: path to library of molecular fragments to read in
        solute_molecule_options: listof lists detailing sets of solutes
        solvent_molecules: list or dictionary of solvent molecules. If dictionary, corresponding value is relative weight of solvent
        cell_range: 3X2 list with x, y, and z ranges for cell size 
        Rrange: density range
        min_dist: minimum contact distance between fragments
        max_patience: How many attempts to  make before giving up on build
        center_first_molecule: Boolian,  if true first solute is centered in box and not rotated (useful for large molecules)
        shake: Distance to displace initial configurations
        print_attempt: Boolian,controls printing (set to False)
    """
    
    cell_shape = [np.random.uniform(dim[0],dim[1]) for dim in cell_range]
    
    empty_system = [{'moleculeid':moleculeid},Atoms(cell=cell_shape,pbc=True),{}]
        
    solute_molecules = random.choice(solute_molecule_options)
    
    molecule_library = {}
    for curSMILES in (solute_molecules + solvent_molecules):
        molecule_library[curSMILES] = ASEfromSMILES(curSMILES)

    feed_parameters = {}
    
    feed_parameters['solute_molecules'] = solute_molecules
    feed_parameters['density'] = np.random.uniform(Rrange[0],Rrange[1])
    
    input_parameters = build_input_dict(condensed_phase_builder,[{"start_system":empty_system,"molecule_library":molecule_library,"solute_molecules":solute_molecules,"density":np.random.uniform(Rrange[0],Rrange[1])},builder_config])
    system = condensed_phase_builder(**input_parameters)
    system_checker(system)
    return(system)
