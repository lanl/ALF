import os
import sys
import numpy as np
import math as m
import ase
from ase import io
from ase import build
from ase.data import atomic_masses, atomic_numbers
from ase import geometry
import random as r
from time import time
from collections import Counter

from alframework.tools.tools import random_rotation_matrix   
from alframework.tools.tools import build_input_dict         
from alframework.tools.molecules_class import MoleculesObject

from parsl import python_app, bash_app

from glob import glob

def builder(StructFile,Rattle=0.0,repAtMax=0,CScale=1.0,CRat=0.0,tInt=[],nVac=0,tRep=[], tMain=None):

    ## Get Starting Struct from File to Edit
    stime=time()
    S = ase.io.read(StructFile, format="cfg")
    print("Reading time: {:.4f}".format(time()-stime))
    if tMain == None:
        counts = Counter(S.get_chemical_symbols())
        maxC = 0
        for curK in counts:
            if counts[curK] >= maxC:
                maxC = counts[curK]
                tMain=curK
    print(tMain)

    ## Replicate Structure to be Largest Possible with N < N_Max
    stime=time()
    if repAtMax > 0:
        repV = [1,1,1]
        Natoms = len(S)
        goodV = [0,1,2]
        while len(goodV)>0:
            curAx = r.choice(goodV)
            if (repV[0]+1*(curAx==0))*(repV[1]+1*(curAx==1))*(repV[2]+1*(curAx==2))*Natoms < repAtMax:
                repV[curAx]+=1
            else:
                goodV.remove(curAx)
        S = ase.build.make_supercell(S, np.diag(repV))
    print("Replicate time: {:.4f}".format(time()-stime))
    stime=time()

    #### Scale the Cell lengths by a ration factor
    if CScale != 1:
        CC = S.cell
        #CC[0][0] = S.cell[0][0] * CScale
        #CC[1][1] = S.cell[1][1] * CScale
        #CC[2][2] = S.cell[2][2] * CScale
        CC = CC * CScale
        S.set_cell(CC,scale_atoms=True)
    print("Cell Scale time: {:.4f}".format(time()-stime))
    stime=time()
    

    ### Rattle the Cell Lengths Individually to Create Small Deviatoric Perturbations
    if CRat > 0:
        CC = S.cell
        CC[0][0] = S.cell[0][0] + np.random.normal(loc=0.0, scale=CRat)
        CC[1][1] = S.cell[1][1] + np.random.normal(loc=0.0, scale=CRat)
        CC[2][2] = S.cell[2][2] + np.random.normal(loc=0.0, scale=CRat)
        S.set_cell(CC,scale_atoms=True)
    
    print("C rattle time: {:.4f}".format(time()-stime))
    stime=time()

    ### Rattle All Atom Locations
    if Rattle > 0: 
        S.rattle(stdev=Rattle, rng=np.random)
    
    print("Rattle time: {:.4f}".format(time()-stime))
    stime=time()


    ## Insert Interstitials of a Type
    if len(tInt) > 0:
        from ase import Atoms
        from ase import Atom
        i = 0
        while i < len(tInt):
            newPos = [0, 0, 0]
    

# Define a search grid (adjust resolution as needed)
            grid_resolution = 15
            x = np.linspace(0, S.cell[0, 0], grid_resolution)
            y = np.linspace(0, S.cell[1, 1], grid_resolution)
            z = np.linspace(0, S.cell[2, 2], grid_resolution)

            farthest_point = None
            max_min_distance = -1.0

            # Iterate through grid points
            for p in range(grid_resolution):
                for j in range(grid_resolution):
                    for k in range(grid_resolution):
                        current_point = np.array([x[p], y[j], z[k]])
                        min_distance_to_atoms = float(100000000.0)

                        # Calculate distance to all atoms (including periodic images implicitly via MIC)
                        distance = geometry.get_distances(current_point, S.positions,  cell=S.cell, pbc=True)
                        distance1 = min(distance[1][0])
                        min_distance_to_atoms = distance1

                        # Update if this point is "farthest" so far
                        if min_distance_to_atoms > max_min_distance:
                            max_min_distance = min_distance_to_atoms
                            farthest_point = current_point
            
            newAtom = Atom(tInt[i], farthest_point)
            S += newAtom
            i+=1
    
    print("Interstitial time: {:.4f}".format(time()-stime))
    stime=time()

    Natoms = len(S.positions)
    Elements = S.get_chemical_symbols()
    
    ### Change Atom Types and Mass for a Type Replacement
    if len(tRep) > 0:
        i = 0
        while i < len(tRep):
            NAt = r.randint(0,Natoms-1)
            if Elements[NAt] == tMain:
                zRep = atomic_numbers[tRep[i]]
                mRep = atomic_masses[zRep]
                S.numbers[NAt] = zRep
                masses = S.get_masses()
                masses[NAt] = mRep
                S.set_masses(masses)
                i+=1
    
    print("Replacement time: {:.4f}".format(time()-stime))
    stime=time()

    #### Create Vacancies
    if nVac > 0:
        i = 0
        while i < nVac:
            NAt = r.randint(0,Natoms-1)
            if Elements[NAt] == tMain:
                del S[NAt]
                i+=1
    
    print("Vacancy time: {:.4f}".format(time()-stime))
    stime=time()
    ### Write CFG Files
    #ase.io.write("NewStructure.cfg", S, format="cfg")
    #ase.io.write("POSCAR", S, format="vasp")

    return(S)

@python_app(executors=['alf_sampler_executor'])
def defect_cell_builder_task(moleculeid: str,cfg_dir: str, maxVac:int = 0, maxInt:int = 0, typInt:list = [], maxRep:int = 0, typRep:list = [], rattle:float=0.0, cScaleRange:list = [.99999, 1.0001], cRattle:float = 0.0, maxAt:int=100):
    cfg_list = glob(cfg_dir+'*.cfg')
    
    builder_inputs = {}
    builder_inputs['StructFile'] = r.choice(cfg_list)
    builder_inputs['nVac'] = r.choice(list(range(maxVac+1)))
    builder_inputs['tInt'] = r.choices(typInt,k=r.choice(list(range(maxInt+1))))
    builder_inputs['tRep'] = r.choices(typRep,k=r.choice(list(range(maxRep+1))))
    builder_inputs['Rattle'] = rattle
    builder_inputs['CScale'] = r.uniform(cScaleRange[0],cScaleRange[1])
    builder_inputs['CRat'] = cRattle
    builder_inputs['repAtMax'] = maxAt
    print(builder_inputs)
    
    at = builder(**builder_inputs)
    molecule_object=MoleculesObject(at,moleculeid)
    molecule_object.update_metadata(builder_inputs)
    return(molecule_object)
    
    
    
