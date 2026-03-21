ML Interfaces
=============

ML interfaces provide training and inference hooks for machine-learned potentials.

Available ML interface modules
------------------------------

ALF currently includes two ML interface modules:

1. ``alframework.ml_interfaces.hippynn_interface``
   HIPPYNN-based training and model-management utilities, including support for
   ensemble workflows used during active learning.
2. ``alframework.ml_interfaces.neurochem_interface``
   NeuroChem/ANI training and calculator interface utilities for ANI-based
   models and ensembles.

Choosing an ML interface
------------------------

Use ``hippynn_interface`` when your workflow is centered on HIPPYNN models and
you want ALF to orchestrate retraining in the active-learning loop.

Use ``neurochem_interface`` when your workflow depends on ANI/NeuroChem
tooling and model formats.

In both cases, the selected ML task is configured through your ALF master and
ML configuration JSON files.

ML interface API links
----------------------

You can link from this guide directly to API pages:

* :doc:`ML interfaces package API <../api_documentation/alframework.ml_interfaces>`
* :doc:`HIPPYNN interface module <../api_documentation/alframework.ml_interfaces.hippynn_interface>`
* :doc:`NeuroChem interface module <../api_documentation/alframework.ml_interfaces.neurochem_interface>`
