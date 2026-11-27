# The builder code goes here
import random
import numpy as np

from ase import Atoms
from ase.data import chemical_symbols

from alframework.tools.config import build_input_dict
from alframework.tools.label_validation import system_checker
from alframework.tools.molecules import MoleculesObject
from alframework.builders.builders import condensed_phase_builder
from alframework.tools.tasks import alf_task


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry

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

@alf_task('builder')
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

    Returns:
        MoleculesObject: Periodic packed geometry with its molecule id and density metadata.
    """
    
    cell_shape = [np.random.uniform(dim[0],dim[1]) for dim in cell_range]
    
    empty_system = MoleculesObject(Atoms(cell=cell_shape, pbc=True), moleculeid)
        
    # Packing consumes the list of solutes; do not mutate the reusable configuration.
    solute_molecules = list(random.choice(solute_molecule_options))
    
    molecule_library = {}
    for curSMILES in dict.fromkeys(solute_molecules + list(solvent_molecules)):
        molecule_library[curSMILES] = ASEfromSMILES(curSMILES)
    
    input_parameters = build_input_dict(condensed_phase_builder,[{"start_system":empty_system,"molecule_library":molecule_library,"solute_molecules":solute_molecules,"solvent_molecules":solvent_molecules,"density":np.random.uniform(Rrange[0],Rrange[1])},builder_config])
    system = condensed_phase_builder(**input_parameters)
    system_checker(system)
    return(system)
