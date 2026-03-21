Samplers
========

Samplers drive data generation by exploring configuration space with QBC uncertainty quantification.

Available sampler modules
-------------------------

ALF currently includes several sampler-related modules:

1. ``alframework.samplers.mlmd_sampling``
   Main uncertainty-driven sampling workflow using Langevin dynamics. This is the
   primary sampler used in many ALF runs.
2. ``alframework.samplers.reactive_sampler``
   Reactive sampling workflows based on NEB and dimer-style methods for constructing
   reaction-focused datasets.

Choosing a sampler
------------------

Use ``mlmd_sampling`` for general active-learning loops where uncertainty
thresholds are used to decide when to call QM.

Use ``reactive_sampler`` when the target problem is reaction-pathway sampling
and transition-state exploration.


Sampler API links
-----------------

You can link from this guide directly to API pages:

* :doc:`Samplers package API <../api_documentation/alframework.samplers>`
* :doc:`MLMD sampling module <../api_documentation/alframework.samplers.mlmd_sampling>`
* :doc:`Reactive sampler module <../api_documentation/alframework.samplers.reactive_sampler>`
