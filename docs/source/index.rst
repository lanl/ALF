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

ALF uses configurable Parsl tasks to connect builders, samplers, machine
learning interfaces, and electronic structure interfaces. Candidate structures
are sampled with an ensemble model, uncertain configurations are sent to QM,
and the resulting data are written back into training datasets for subsequent
model updates.

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
   :caption: User Guide

   user_guide/configuration
   user_guide/builders
   user_guide/samplers
   user_guide/ml_interfaces
   user_guide/qm_interfaces
   user_guide/parsl

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/simple_water
   examples/molten_salt
   examples/reactive_sampling

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
