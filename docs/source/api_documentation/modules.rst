alframework
===========

Shared Tools
------------

Use the defining modules when importing shared helpers in workflow code:

* :doc:`Workflow configuration and task arguments <alframework.tools.config>`
* :doc:`HDF5 training shards <alframework.tools.data_shards>`
* :doc:`Label sources <alframework.tools.label_sources>`
* :doc:`Parsl queues and shutdown <alframework.tools.parsl_tasks>`
* :doc:`Bootstrap seed allocation <alframework.tools.bootstrap>`
* :doc:`Restart state and held labels <alframework.tools.restarts>`
* :doc:`Molecule records <alframework.tools.molecules>`
* :doc:`Labeling coordinator <alframework.tools.labeling_coordinator>`
* :doc:`Worker environment setup <alframework.tools.worker_setup>`
* :doc:`Structure and label validation <alframework.tools.label_validation>`
* :doc:`Sampler screen audit records <alframework.tools.screen_audit>`
* :doc:`Geometry and rotations <alframework.tools.geometry>`
* :doc:`Annealing schedules <alframework.tools.annealing>`

The old ``alframework.tools.tools`` import paths remain available for existing
scripts, configurations, and pickled references.
Renamed tool modules also accept their previous import paths and emit a
warning that names the canonical module.

.. toctree::
   :maxdepth: 4

   alframework
