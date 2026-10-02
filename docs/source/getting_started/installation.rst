Installation
============

ALF is installed as a Python package, but a complete production workflow also
depends on the external QM, ML, scheduler, MPI, and GPU software required.

Prerequisites
-------------

Before installing ALF, use an environment with Python ``>3.9``. A fresh virtual
environment or conda environment is recommended so ALF's Python dependencies do
not conflict with other scientific software stacks.

The Python install includes ALF's core Python dependencies, including Parsl and
ASE. It does not install external executables or site-specific software such as
ORCA, VASP, Q-Chem, scheduler modules, MPI launchers, CUDA/ROCm drivers, or
cluster resource configuration.

Basic Install
-------------

Clone the repository and install ALF in editable mode:

.. code-block:: bash

   git clone https://github.com/lanl/ALF.git
   cd ALF
   python -m pip install -e .

For a quick import check:

.. code-block:: bash

   python -c "import alframework; print('ALF import OK')"

Testing Install
---------------

Install the lightweight unit-test dependencies with the ``tests`` extra:

.. code-block:: bash

   python -m pip install -e ".[tests]"
   python -m pytest

These tests are intended to run without external QM executables, trained ML
models, GPUs, or Parsl executors.

Documentation Install
---------------------

Install the documentation dependencies with the ``docs`` extra:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/_autobuild/html

The generated HTML pages are written to ``docs/_autobuild/html``.

Optional Scientific Backends
----------------------------

Some workflows need additional packages or external programs. The optional
``full`` extra currently installs ``pyscf`` and ``hippynn``:

.. code-block:: bash

   python -m pip install -e ".[full]"

This is not required for every user and does not cover every supported backend.
HIPPYNN, NeuroChem/ANI, ORCA, VASP, Q-Chem, RDKit, MPI, and GPU toolchains are
workflow-specific. ASE-supported QM engines can still require separately
installed executables, environment variables, pseudopotential paths, and license
or module setup.

See :doc:`../user_guide/ml_interfaces`, :doc:`../user_guide/qm_interfaces`, and
:doc:`../user_guide/builders` for backend-specific configuration patterns.

HPC And Parsl Setup
-------------------

ALF installs Parsl as a Python dependency, but users still need to adapt the
Parsl configuration for the machine where ALF will run. In the example
directories, ``parsl_configs.py`` controls scheduler partitions, accounts, QoS,
walltime, worker initialization, launchers, CPU resources, and GPU resources.

See :doc:`../user_guide/parsl` for the executor labels ALF expects and for
templates covering common Slurm CPU/GPU layouts.

Next Steps
----------

* :doc:`quickstart` for the recommended first run.
* :doc:`../examples/simple_water` for the full simple-water walkthrough.
* :doc:`../user_guide/parsl` for cluster and executor setup.
* :doc:`../user_guide/ml_interfaces` for HIPPYNN and ML backend setup.
* :doc:`../user_guide/qm_interfaces` for ORCA, VASP, Q-Chem, and ASE-backed QM
  setup.
