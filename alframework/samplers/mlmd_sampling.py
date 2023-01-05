import numpy as np
import ase
from ase.md.langevin import Langevin
from ase import units
import time
from ase import Atoms
import pickle as pkl

from alframework.tools.tools import annealing_schedule

def mlmd_sampling(moleculeid, ase_atoms, ase_calculator, sample_params, meta_path=None):

    # Load MD parameters
    dt = sample_params['dt']
    maxt = sample_params['maxt']
    Escut = sample_params['Escut']
    Fscut = sample_params['Fscut']
    Ncheck = sample_params['Ncheck']

    # Setup T
    str_Trange = sample_params['srt_temp']
    end_Trange = sample_params['end_temp']
    amp_Trange = sample_params['amp_temp']
    per_Trange = sample_params['per_temp']

    Tamp = np.random.uniform(amp_Trange[0], amp_Trange[1])
    Tper = np.random.uniform(per_Trange[0], per_Trange[1])
    Tsrt = np.random.uniform(str_Trange[0], str_Trange[1])
    Tend = np.random.uniform(end_Trange[0], end_Trange[1])

    T = annealing_schedule(0.0, maxt, Tamp, Tper, Tsrt, Tend)

    # Setup Rho
    str_Rvalue = (1.66054e-24/1.0e-24)*(np.sum(ase_atoms.get_masses())/ase_atoms.get_volume())
    end_Rrange = [sample_params['end_dens'][0]*str_Rvalue, sample_params['end_dens'][1]*str_Rvalue]
    #end_Rrange = self.sample_params['end_dens']
    amp_Rrange = sample_params['amp_dens']
    per_Rrange = sample_params['per_dens']

    Ramp = np.random.uniform(amp_Rrange[0], amp_Rrange[1])
    Rper = np.random.uniform(per_Rrange[0], per_Rrange[1])
    Rend = np.random.uniform(end_Rrange[0], end_Rrange[1])

    #select_dens = self.annealing_schedule(0.0, maxt, Ramp, Rper, str_Rvalue, Rend)

    # Build molecule ID
    #moleculeid = 'mol-{:04d}-{:010d}'.format(model_id,molecule_index)
    print('MolID:', moleculeid)

    #write(moleculeid+'.pdb',ase_atoms, parallel=False)

    # Set the momenta corresponding to T
    #MaxwellBoltzmannDistribution(mol, T / 2 * units.kB)

    # Set the ASE Calculator
    ase_atoms.set_calculator(ase_calculator)

    # Compute energies
    ase_atoms.get_potential_energy()

    # Define thermostat
    dyn = Langevin(ase_atoms, dt * units.fs, friction=0.02, temperature_K=T, communicator=None)

    # Iterate MD
    failed = False

    set_temp = T
    set_dens = str_Rvalue

    sim_start_time = time.time()

    temps = []
    denss = []

    for i in range(int(np.ceil((1000*maxt)/(dt*Ncheck)))):

        # Check if MD should be run
        Es = ase_atoms.calc.Estddev*1000.0
        Fs, Fsmax = ase_atoms.calc.get_Fstddev()

        Ecrit = Es > Escut
        Fcrit = Fs > Fscut
        Fmcrit = Fsmax > 3*Fscut
    
        if Ecrit or Fcrit or Fmcrit:
            #print('MD FAIL (',model_id,',',self.counter,',',"{0:.4f}".format(time.time()-self.str_time),') -- Uncertainty:', "{0:.4f}".format(Es), "{0:.4f}".format(Fs), "{0:.4f}".format(Fsmax), (i*Ncheck*dt)/1000)
            last_bad = Atoms(ase_atoms.get_chemical_symbols(),positions=ase_atoms.get_positions(wrap=True),cell=ase_atoms.get_cell(),pbc=ase_atoms.get_pbc())
            failed = True

            break

        # Set the temperature
        set_temp = annealing_schedule((i * Ncheck * dt) / 1000, maxt, Tamp, Tper, Tsrt, Tend)
        dyn.set_temperature(temperature_K = set_temp)

        # Set the density
        set_dens = (1.0e-24 / 1.66054e-24) * annealing_schedule((i * Ncheck * dt) / 1000, maxt, Ramp, Rper, str_Rvalue, Rend)
        scalar = np.power(np.sum(ase_atoms.get_masses()) / (ase_atoms.get_volume() * set_dens), 1. / 3.)
        ase_atoms.set_cell(scalar * ase_atoms.get_cell(), scale_atoms=True)

        temps.append(set_temp)
        denss.append(set_dens)

        # Run MD
        dyn.run(Ncheck)

    meta_dict = {"moleculeid" : moleculeid,
                    "realtime_simulation" : time.time()-sim_start_time,
                    "Es"    : Es,
                    "Fs"    : Fs,
                    "Fsmax" : Fsmax,
                    "simulationtime" : (i*Ncheck*dt)/1000,
                    "temps" : temps,
                    "denss" : denss,
                    "system_comp": np.unique(ase_atoms.get_chemical_symbols(),return_counts=True),
                    "Tamp" : Tamp,
                    "Tper" : Tper,
                    "Tsrt" : Tsrt,
                    "Tend" : Tend,
                    "Ramp" : Ramp,
                    "Rper" : Rper,
                    "Rsrt" : str_Rvalue,
                    "Rend" : Rend,
                    "Ecrit" : Ecrit,
                    "Fcrit" : Fcrit,
                    "Fmcrit" : Fmcrit,
                    "chemical_symbols" : ase_atoms.get_chemical_symbols(),
                    "positions" : ase_atoms.get_positions(wrap=True),
                    "cell" : ase_atoms.get_cell()
                }
    if meta_path is not None:
        pkl.dump( meta_dict, open( meta_path+"/metadata-"+moleculeid+'.p', "wb" ) )
    
    ase_atoms.calc = None  #replace calculator for return
    
    if failed:
        return [moleculeid, ase_atoms]
    else:
        print('MD SUCCESS',self.counter)
        return [moleculeid, None]
