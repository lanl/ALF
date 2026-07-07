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


ALF tracks model uncertainty through a general-purpose ensemble calculator. During each active learning iteration, a committee of independently trained MLIP models evaluates newly sampled configuration, and ALF uses the disagreement among the ensemble predictions to identify regions of the potential energy surface that are insufficiently represented in the current training dataset. These high-uncertainty configurations are then selected for ground-truth labeling and added to the training dataset.

ALF uses Parsl to manage task execution across HPC clusters and job schedulers (e.g., SLURM). Because resource layouts differ between computing environments, users should modify the Parsl configuration files in `alframework/parsl_resource_configs` before running production workflows. These files control machine-specific settings such as scheduler type, queue or partition name, allocation account, walltime, node counts, worker counts, launch commands, and software environment setup.

## 📋 Requirements: 

The requirements for this software are evolving, though generally, they will include the following: 
1) [Parsl](https://parsl-project.org/)
2) [NumPy](https://numpy.org/)
3) [ASE](https://wiki.fysik.dtu.dk/ase/)
4) A QM software package with interface (usually ASE)
5) A MLIP model - we provide an interface to the open-source and flexible [HIPPYNN](https://github.com/lanl/hippynn) architecture

## ✅ Installation

Clone the repository and install ALF using pip:

```bash
git clone https://github.com/lanl/ALF.git
cd ALF
python -m pip install -e .
```

## ⚙️ Configuration:

To control job flow in the active learning framework, 5 json files are used:

1. `master_config.json`
2. `builder_config.json`
3. `ml_config.json`
4. `mlmd_config.json`
5. `qm_config.json`

The master configuration file controls how all pieces of the framework are assembled and defines where to find the other 4 files. Each of the other 4 files passes inputs to one of the four subtasks enumerated in the first section. To see how these files relate to one another, please see the `examples` folder.

With the json files completed, the `PYTHONPATH` environment variable must be set to the directory where `alframework` is held. Eventually, this step will be replaced by making `alframework` an installable package. 

## 🧪 Testing: 

Once the environment is constructed with the required packages, it is important to test individual operations done by the active learning framework for erorrs. This is done to ensure all processes complete successufuly when run in active learning. Testing each of of the four sub processes is enabled in the following way:

```
python -m alframework --master master_config.json --test_builder #Test structure building
python -m alframework --master master_config.json --test_sampler #Test mlmd sampling
python -m alframework --master master_config.json --test_ml
python -m alframework --master master_config.json --test_qm
```

These functions will execute in such a way as to pass errors back to the front end to enable easier debugging. Errors encountered in the active learning phase.

## ▶️ Execution:

Once each task has been tested, active learning can be started with:
```
python -m alframework --master master_config.json
```

It is generally advised to run the master process on a head node inside a terminal multiplexer ([screen](https://www.gnu.org/software/screen/), [tmux](https://github.com/tmux/tmux/wiki), [zellij](https://zellij.dev/about/)) for session persistence. This will allow the ALF master process to continue to run over multiple days/weeks, even after you disconnect. It will automatically interface with the queueing system and run future jobs on compute nodes. 

## 📃 Citations:

If you use ALF in your research, **citations to the papers 1-3 and this repository are mandatory**.
Please, also consider citing other examples below.  

[1] Code release paper and molten salts case study   
*in preparation - link will appear here*

[2] ALF-prouced MLIP for bulk aluminum    
Justin S. Smith, Benjamin Nebgen, Nithin Mathew, Jie Chen, Nicholas Lubbers, Leonid Burakovsky, Sergei Tretiak, Hai Ah Nam, Timothy Germann, Saryu Fensin, Kipton Barros. "Automated discovery of a robust interatomic potential for aluminum" Nat. Comm. 2021,  12, 1257. 
https://doi.org/10.1038/s41467-021-21376-0

[2] Uncertainty-driven dynamics for active learning - UDD sampler     
Nicholas Lubbers, Ying Wai Li, Richard Messerly, Sergei Tretiak, Justin S. Smith, Benjamin Nebgen. "Uncertainty-driven dynamics for active learning of interatomic potentials" Nat. Comp. Sci. 2023, 1968. 
https://doi.org/10.1038/s43588-023-00406-5 

[3] ALF-trained reactive potential for organics    
Shuhao Zhang, Malgorzata Makos, Ryan Jadrich, Elfi Kraka, Kipton Barros, Benjamin Nebgen, Sergei Tretiak, Olexandr Isayev, Nicholas Lubbers, Richard Messerly, Justin Smith. "Exploring the frontiers of chemistry with a general reactive machine learning potential".
https://doi.org/10.26434/chemrxiv-2022-15ct6

[4] Original implementation of ALF and proof of concept study    
Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, Adrian E. Roitberg. "Less is more: Sampling chemical space with active learning". J. Chem. Phys. 2018, 148, 241733.
https://doi.org/10.1063/1.5023802
