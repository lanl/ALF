import os
import glob
import random
import subprocess
import tempfile
from ase import Atoms
from ase.io import read, write
from parsl import python_app, bash_app
from alframework.tools.tools import system_checker
from alframework.tools.tools import load_module_from_config
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator


def crest_build(start_system, molecule_library_dir, nsolv=[1,2], crest_command="crest", crest_input="", grow_input="", store_dir=None):

    #ensure system adhears to formating convention
    if type(start_system) == list:
        for system in start_system: 
            system_checker(system)
        prefix = start_system[0][0]['moleculeid']
    else: 
        system_checker(start_system)
        prefix = system[0]['moleculeid']
 
    #path to solute/solvent xyz
    solute_xyzs = sorted(glob.glob(os.path.join(molecule_library_dir, 'solute', '*.xyz')))
    solvent_xyzs = sorted(glob.glob(os.path.join(molecule_library_dir, 'solvent', '*.xyz')))
    solute_xyz = random.choice(solvent_xyzs)
    solvent_xyz = random.choice(solute_xyzs)
    nsolv_start, nsolv_stop = nsolv
    nsolv = random.choice(range(int(nsolv_start), int(nsolv_stop)))

    if store_dir is not None:
        os.makedirs(store_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)
        #run crest cluster growth mode
        runcmd = [crest_command, solute_xyz, '-qcg', solvent_xyz, '--nsolv', str(nsolv)] + grow_input.split()
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if store_dir is not None:
            outfile = os.path.join(store_dir, f"{prefix}_build.out")
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
                f.write(proc.stderr)

        #read restraint potential from cluster growth
        with open('grow/wall_potential', 'r') as f:
            wall_potential = f.readlines()

        #make a copy
        with open('wall_potential', 'w') as f:
            for line in wall_potential:
                f.write(line)
       
        #run crest conformer sampling mode
        runcmd = [crest_command, 'grow/cluster.xyz', '--nci', '--cinp', 'wall_potential'] + crest_input.split()
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if store_dir is not None:
            outfile = os.path.join(store_dir, f"{prefix}_conformers.out")
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
                f.write(proc.stderr)

        #load cluster
        conformers_xyz = os.path.join(tmpdirname, 'crest_conformers.xyz')
        conformers = read(conformers_xyz, index=':')
        if store_dir is not None:
            outfile = os.path.join(store_dir, f"{prefix}_conformers.xyz")
            runcmd = ['cp', conformers_xyz, outfile]
            _ = subprocess.run(runcmd)

        #write structures
        if type(start_system) == list:
            for n in range(len(start_system)):
                start_system[n][1] = conformers[n]
                start_system[n][0]['wall_potential'] = wall_potential
        else:
            start_system[1] = conformers[0]
            start_system[0]['wall_potential'] = wall_potential

    return(start_system)


def crest_metad(start_system, ase_calculator, xtb_command='xtb', hmass=2, time=50., temp=400., step=0.5, shake=0,
                dump=100, save=100, kpush=0.05, alp=1.0, Escut=20., Fscut=0.3, store_dir=None):

    #ensure system adhears to formating convention
    system_checker(start_system)
    curr_sys = start_system[1]
    assert "wall_potential" in start_system[0].keys()  

    #read wall_potential to find solute atom indices
    wall_potential = start_system[0]['wall_potential']
    solute_idx = wall_potential[-1].split()[-1]

    if store_dir is not None:
        os.makedirs(store_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        #write input file
        with open('metadyn.inp', 'w') as f:
            #md block
            f.write(f"$md\n  hmass={hmass}\n  time={time}\n  temp={temp}\n  ")
            f.write(f"step={step}\n  shake={shake}\n  dump={dump}\n  $end\n")
            #metad block
            f.write("$metadyn\n atoms={solute_idx}\n  save={save}\n  kpush={kpush}\n  alp={alp}\n$end\n")
            #constraint potential
            for line in wall_potential:
                f.write(line)
            f.write("$end\n")
        
        #input coordinates
        write('input.xyz', curr_sys)

        #run metadynamics
        runcmd = [xtb_command, '--md', '--input', 'metadyn.inp', 'input.xyz']
        proc = subprocess.run(runcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if store_dir is not None:
            outfile = os.path.join(store_dir, f"{prefix}_sample.out")
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
                f.write(proc.stderr)

        #read structures from trajectory
        cluster_xyz = os.path.join(tmpdirname, 'xtb.trj') 

        if store_dir is not None:
            outfile = os.path.join(store_dir, f"{prefix}_trj.xyz")
            runcmd = ['cp', cluster_xyz, outfile]
            _ = subprocess.run(runcmd)

        #loop over snapshots and return the snapshot if it meets criteria
        traj = read(cluster_xyz, format='xyz', index=':')
        for atoms in traj:
            atoms.set_calculator(ase_calculator)
            atoms.calc.calculate(atoms, properties=['energy_stdev','forces_stdev_mean','forces_stdev_max'])
            Es = ase_atoms.calc.results['energy_stdev']
            Fs = ase_atoms.calc.results['forces_stdev_mean']
            Fsmax = ase_atoms.calc.results['forces_stdev_max']

            Ecrit = Es > Escut
            Fcrit = Fs > Fscut
            Fmcrit = Fsmax > 3*Fscut
    
            if Ecrit or Fcrit or Fmcrit:
                start_system[1] = atoms
                start_system[2] = {}
                return(start_system)
        
    #if we get here, no snapshots met the criteria
    start_system[1] = None
    start_system[2] = {}
    return(start_system)

@python_app(executors=['alf_sampler_executor'])
def crest_build_task(moleculeids, builder_config):
    """
    Elements in builder params
        molecule_library_dir: path to library of molecular fragments to read in
        nsolv: int. number of solvent molecules
        crest_command: path to CREST
        crest_input: optional command line arguments for conformer generation
        grow_input: optional command line arguments for solvent docking
        store_dir: optional path to storage directory
    """
    if type(moleculeids) == list:
        empty_systems = [[{'moleculeid':moleculeid}, Atoms(), {}] for moleculeid in moleculeids]
    else:
        empty_systems = [{'moleculeid':moleculeids}, Atoms(), {}]
    system = crest_build(empty_systems, **builder_config)
    if type(system) == list:
        for s in system:
            system_checker(s)
    else:
        system_checker(system)
    return(system)


@python_app(executors=['alf_sampler_executor'])
def crest_metad_task(molecule_object, sampler_config, model_path, current_model_id):
    """
    Elements in sampler params
        xtb_command: path to xTB
        hmass: mass of hydrogen atoms (amu)
        time: integration time (ps)
        temp: temperature (K)
        step: step size (fs)
        shake: bond constraints (0=off, 1=X-H-bonds, 2=all-bonds)
        dump: trajectory write interval (fs)
        save: max number of structures for RMSD collective variable
        kpush: scaling factor for Gaussian potential used in RMSD CV
        alp: width of gaussian potential used in RMSD CV
        Escut: Energy standard deviation threshold for capturing frame
        Fscut: Force standard deviation threshold for capturing frame
        store_dir: optional path to storage directory
    """
    system_checker(molecule_object)
    calc_class = load_module_from_config(sampler_config, 'ase_calculator')
    calculator_list = calc_class(model_path.format(current_model_id) + '/',device='cuda:0')
    ase_calculator = MLMD_calculator(calculator_list,**sampler_config['MLMD_calculator_options'])
    system = crest_metad(molecule_object, ase_calculator, **sampler_config)
    system_checker(system)
    return(system)


