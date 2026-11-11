Welcome to ALF's documentation!
===============================

ALF, the Active Learning Framework, automates the construction of training
datasets for machine-learned interatomic potentials. It coordinates structure
generation, ML-driven sampling, quantum-mechanical labeling, and model
retraining in a single active-learning loop.

The documentation is organized around the main ways users interact with ALF:
setting up a run, understanding workflow components, adapting examples, and
checking API details.

What is ALF?
------------

ALF uses Parsl tasks to connect builders, samplers, machine-learning interfaces, and
electronic-structure interfaces. An ensemble of models drives sampling, frames where the models
disagree are labeled with QM, and the ensemble is retrained on the new data. ALF handles
ground-state and excited-state systems, can fine-tune existing models such as foundation models,
and runs from a workstation to many HPC nodes.

Start with :doc:`getting_started/what_is_alf` for a high-level overview, then
use :doc:`user_guide/configuration` and the example pages to connect the
pieces to a concrete run.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/what_is_alf
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: The Active-Learning Loop

   user_guide/configuration
   user_guide/builders
   user_guide/samplers
   user_guide/acquisition
   user_guide/ml_interfaces
   user_guide/qm_interfaces
   user_guide/monitoring

.. toctree::
   :maxdepth: 2
   :caption: Extended Workflows

   user_guide/excited_states
   user_guide/teacher_student
   user_guide/fine_tuning

.. toctree::
   :maxdepth: 2
   :caption: Running On Clusters

   user_guide/parsl
   user_guide/batching
   user_guide/running_at_scale

.. toctree::
   :maxdepth: 2
   :caption: Reference

   user_guide/units
   user_guide/glossary

.. toctree::
   :maxdepth: 2
   :caption: Examples: Ground State

   examples/simple_water
   examples/periodic_water
   examples/seeded_active_learning
   examples/molten_salt
   examples/il
   examples/uo2
   examples/reactive_sampling

.. toctree::
   :maxdepth: 2
   :caption: Examples: Fine-Tuning

   examples/fine_tuning
   examples/mace_fine_tuning
   examples/uma_fine_tuning

.. toctree::
   :maxdepth: 2
   :caption: Examples: Excited States

   examples/acac_excited_pyseqm
   examples/cpp_excited_gpu4pyscf

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api_documentation/modules
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
