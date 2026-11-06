# ALF: Open-Source Active Learning Framework for Atomistic Modeling

## 📄[Documentation](https://lanl.github.io/ALF/)

This code automates the construction of datasets for machine learned interatomic potentials (MLIPs) through active learning. By automating job execution utilizing the [Parsl](https://parsl-project.org/) framework, the active learning process can run for many iterations without human intervention. ALF breaks the process down into 4 fundamental tasks:

1) Initial system construction (Bootstrapping)
2) ML interatomic potential training
3) ML configurational sampling
4) Electronic structure labeling
   
<br>
<img width="2314" height="1246" alt="alf_White" src="https://github.com/user-attachments/assets/e6af02d2-6d73-4f4c-89c8-2dac8304a42f" />
Overview of the ALF workflow.
<br>
<br>
<br>


ALF tracks model uncertainty through a general-purpose ensemble calculator. During each active learning iteration, a committee of independently trained MLIP models evaluates newly sampled configurations, and ALF uses the disagreement among the ensemble predictions to identify regions of the potential energy surface that are insufficiently represented in the current training dataset. These high-uncertainty configurations are then selected for ground-truth labeling and added to the training dataset.

ALF uses Parsl to manage task execution across HPC clusters and job schedulers (e.g., SLURM). Because resource layouts differ between computing environments, each run names its own Parsl configuration in `parsl_configuration` in the master configuration file. Each example includes a `parsl_configs.py` template to edit for your machine, and `alframework/parsl_resource_configs` holds profiles for the Darwin and Chicoma clusters. These files control machine-specific settings such as scheduler type, queue or partition name, allocation account, walltime, node counts, worker counts, launch commands, and software environment setup.

## 📋 Requirements: 

Installing ALF also installs its core Python dependencies, including [Parsl](https://parsl-project.org/), [NumPy](https://numpy.org/), and [ASE](https://wiki.fysik.dtu.dk/ase/). A complete workflow also needs:

1) A QM software package with an ALF interface (ORCA, VASP, Q-Chem, PySEQM, GPU4PySCF, or another ASE calculator)
2) An MLIP model - we provide an interface to the open-source and flexible [HIPPYNN](https://github.com/lanl/hippynn) architecture

## ✅ Installation

Clone the repository and install ALF using pip:

```bash
git clone https://github.com/lanl/ALF.git
cd ALF
python -m pip install -e .
```

See the [installation guide](https://lanl.github.io/ALF/getting_started/installation.html) for optional extras, such as the TorchSim and ALCHEMI dynamics engines.

## ⚙️ Configuration:

To control job flow in the active learning framework, 5 json files are used:

1. Master configuration, such as `master_config.json`
2. Builder configuration, such as `builder_config.json`
3. ML configuration, such as `hippynn_config.json`
4. Sampler configuration, such as `mlmd_config.json`
5. QM configuration, such as `orca_config.json`

The master configuration file controls how all pieces of the framework are assembled and defines where to find the other 4 files, so their names are up to you. Each of the other 4 files passes inputs to one of the four subtasks enumerated in the first section. To see how these files relate to one another, please see the `examples` folder and the [configuration guide](https://lanl.github.io/ALF/user_guide/configuration.html). For active learning over excited states, see the [excited states guide](https://lanl.github.io/ALF/user_guide/excited_states.html) and the `examples/AcAc_excited_pyseqm` and `examples/CPP_excited_gpu4pyscf` examples.

## 🧪 Testing: 

Once the environment is constructed with the required packages, it is important to test individual operations done by the active learning framework for errors. This is done to ensure all processes complete successfully when run in active learning. Testing each of the four subprocesses is enabled in the following way:

```
python -m alframework --master master_config.json --test_builder #Test structure building
python -m alframework --master master_config.json --test_sampler #Test mlmd sampling
python -m alframework --master master_config.json --test_ml
python -m alframework --master master_config.json --test_qm
```

These commands pass errors back to the terminal to enable easier debugging. Errors encountered in the active learning phase are logged and counted in the status file rather than stopping the run.

## ▶️ Execution:

Once each task has been tested, active learning can be started with:
```
python -m alframework --master master_config.json
```

It is generally advised to run the master process on a head node inside a terminal multiplexer ([screen](https://www.gnu.org/software/screen/), [tmux](https://github.com/tmux/tmux/wiki), [zellij](https://zellij.dev/about/)) for session persistence. This will allow the ALF master process to continue to run over multiple days/weeks, even after you disconnect. It will automatically interface with the queueing system and run future jobs on compute nodes. 

## 📃 Citations:

If you use ALF in your research, **citations to papers 1-4 and this repository are mandatory**.
Please also consider citing paper 5.  

[1] Code release paper and molten salts case study   
*in preparation - link will appear here*

[2] ALF-produced MLIP for bulk aluminum    
Justin S. Smith, Benjamin Nebgen, Nithin Mathew, Jie Chen, Nicholas Lubbers, Leonid Burakovsky, Sergei Tretiak, Hai Ah Nam, Timothy Germann, Saryu Fensin, Kipton Barros. "Automated discovery of a robust interatomic potential for aluminum" Nat. Comm. 2021,  12, 1257. 
https://doi.org/10.1038/s41467-021-21376-0

[3] Uncertainty-driven dynamics for active learning - UDD sampler     
Nicholas Lubbers, Ying Wai Li, Richard Messerly, Sergei Tretiak, Justin S. Smith, Benjamin Nebgen. "Uncertainty-driven dynamics for active learning of interatomic potentials" Nat. Comp. Sci. 2023, 1968. 
https://doi.org/10.1038/s43588-023-00406-5 

[4] ALF-trained reactive potential for organics    
Shuhao Zhang, Malgorzata Makos, Ryan Jadrich, Elfi Kraka, Kipton Barros, Benjamin Nebgen, Sergei Tretiak, Olexandr Isayev, Nicholas Lubbers, Richard Messerly, Justin Smith. "Exploring the frontiers of chemistry with a general reactive machine learning potential".
https://doi.org/10.26434/chemrxiv-2022-15ct6

[5] Original implementation of ALF and proof of concept study    
Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, Adrian E. Roitberg. "Less is more: Sampling chemical space with active learning". J. Chem. Phys. 2018, 148, 241733.
https://doi.org/10.1063/1.5023802
