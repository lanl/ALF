Configuration
=============

ALF is primarily configured with JSON files.

Typical setup includes five files:

1. A master configuration
2. Builder configuration
3. Sampler configuration
4. ML configuration
5. QM configuration

How configuration is organized
------------------------------

The master configuration is the entry point. It defines where ALF should find
the builder, sampler, ML, and QM configuration files, along with key paths and
task definitions used at runtime.

Each stage-specific file then defines options for one part of the workflow:

1. Builders: initial structures and system construction behavior
2. Samplers: configuration-space exploration and sampling controls
3. ML: model training and retraining settings
4. QM: electronic structure calculation settings and interfaces

How ALF uses these files
------------------------

When you run ALF with ``--master``, ALF loads the master JSON first, then
loads the four stage-specific JSON files referenced by it. Those settings are
used to initialize tasks, queues, resource configs, and data paths.

In practice, this means most configuration changes can be made without editing
Python source code.

Practical tips
--------------

1. Keep all JSON files for a run in one directory for reproducibility.
2. Use descriptive filenames so multiple runs are easier to track.
3. Validate each stage with ALF test flags before long production runs.
