# Active Learning Framework:

This code automates the construction of datasets for machine learned (ML) interatomic potentials through active learning. By automating job execution utilizing the [Parsl](https://parsl-project.org/) framework, the active learning process can run for many iterations without human intervention. It breaks the process down into 4 fundamental tasks:
1) system construction (building)
2) sampling (samplers)
3) ML interatomic potential training
4) electronic structure calculations

The framework uses a general purpose ensembling calculator that enables the tracking of model uncertainty during the training process.

This package is closely interfaced to the Parsl framework which enables task execution through a queueing system or smaller clusters. Some requirements, such as the Parsl resource config file in `alframework/parsl_resource_configs` will need to be modified for your computer system. 

This code is currently a work in progress and documentation is still sparse. 

# Requirements: 

The requirements for this software are evolving, though generally, they will include the following: 
1) [NumPy](https://numpy.org/)
2) [ASE](https://wiki.fysik.dtu.dk/ase/)
3) a QM software package with interface (usually ASE)
4) a ML interatomic potential model ([HIPPYNN](https://github.com/lanl/hippynn) is open source and an interface is provided) 

# Configuration:

To control job flow in the active learning framework, 5 json files are used. The master configuration file controls how all pieces of the framework are assembled and defines where to find the other 4 files. Each of the other 4 files passes inputs to one of the four subtasks enumerated in the first section. To see how these files relate to one another, please see the `examples` folder.

With the json files completed, the `PYTHONPATH` environment variable must be set to the directory where `alframework` is held. Eventually, this step will be replaced by making `alframework` an installable package. 

# Testing: 

Once the environment is constructed with the required packages, it is important to test individual operations done by the active learning framework for erorrs. This is done to ensure all processes complete successufuly when run in active learning. Testing each of of the four sub processes is enabled in the following way:

```
python -m alframework master.json --test_builder #Test structure building
python -m alframework master.json --test_sampler #Test mlmd sampling
python -m alframework master.json --test_ml 
python -m alframework master.json --test_qm
```

These functions will execute in such a way as to pass errors back to the front end to enable easier debugging. Errors encountered in the active learning phase.

# Execution:

Once each task has been tested, active learning can be started with:
```
python -m alframework master.json
```

It is generally advised to run the master process on a head node. It will automatically interface with the queueing system and run future jobs on compute nodes. 

# Citations

[1] Justin S. Smith, Benjamin Nebgen, Nithin Mathew, Jie Chen, Nicholas Lubbers, Leonid Burakovsky, Sergei Tretiak, Hai Ah Nam, Timothy Germann, Saryu Fensin, Kipton Barros "Automated discovery of a robust interatomic potential for aluminum" Nat. Comm. 2021,  12, 1257. 
https://doi.org/10.1038/s41467-021-21376-0

[2] Maksim Kulichenko, Kipton Barros, Nicholas Lubbers, Ying Wai Li, Richard Messerly, Sergei Tretiak, Justin S. Smith, Benjamin Nebgen "Uncertainty-driven dynamics for active learning of interatomic potentials" Nat. Comp. Sci. 2023, 1968. 
https://doi.org/10.1038/s43588-023-00406-5 

[3] Shuhao Zhang, Malgorzata Makos, Ryan Jadrich, Elfi Kraka, Kipton Barros, Benjamin Nebgen, Sergei Tretiak, Olexandr Isayev, Nicholas Lubbers, Richard Messerly, Justin Smith "Exploring the frontiers of chemistry with a general reactive machine learning potential". 
https://doi.org/10.26434/chemrxiv-2022-15ct6
