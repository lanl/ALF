Installation
============

ALF is installed as a Python package, but a complete production workflow also
depends on the external QM, ML, scheduler, MPI, and GPU software required.

Prerequisites
-------------

Before installing ALF, use an environment with Python ``>3.9``. ALF's automated
tests run on Python 3.12, and some optional dynamics engines need a recent
Python (see `Optional Scientific Backends`_). A fresh virtual environment or
conda environment is recommended so ALF's Python dependencies do not conflict
with other scientific software stacks.

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

These tests run without external QM executables, pre-trained ML models, GPUs, or
a cluster. Tests that exercise the ALF coordinator run real Parsl tasks on local
threads. Tests that need an optional package, such as HIPPYNN or RDKit, are
skipped when it is missing, and pytest prints the reason at the end of its
output.

Documentation Install
---------------------

Install the documentation dependencies with the ``docs`` extra:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-apidoc --separate --implicit-namespaces --module-first -d 5 \
     -o docs/source/api_documentation alframework
   sphinx-build -b html docs/source docs/_autobuild/html

The ``sphinx-apidoc`` step generates the API pages that the user guide links to.
The generated HTML pages are written to ``docs/_autobuild/html``.

Optional Scientific Backends
----------------------------

Some workflows need additional packages or external programs. These extras
install the Python packages that ALF can pull in directly:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Extra
     - Installs
     - Use it for
   * - ``full``
     - ``pyscf`` and ``hippynn``
     - HIPPYNN training and sampling.
   * - ``topology``
     - ``rdkit``
     - The RDKit condensed-phase builder, ``rdkit_condensed_phase_builder_task``.
   * - ``torchsim``
     - ``torch-sim-atomistic``
     - The TorchSim dynamics engine. Needs Python 3.12 or newer.
   * - ``alchemi``
     - ``nvalchemi-toolkit``
     - The ALCHEMI dynamics engine. Needs Python 3.11 or newer.

For example:

.. code-block:: bash

   python -m pip install -e ".[full,torchsim]"

Install the PyTorch build that matches your hardware (CPU, CUDA, or ROCm) before
installing the ``torchsim`` or ``alchemi`` extras. ALCHEMI also needs
``torchvision``, which must come from the same PyTorch index as ``torch``:

.. code-block:: bash

   # CPU example. Pick the index URL for your hardware from pytorch.org.
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -e ".[alchemi]"

Dynamics engines must be installed in the environment that the sampler workers
use, which :doc:`Parsl's <../user_guide/parsl>` ``worker_init`` setting controls.
That environment can differ from the one running the ALF master process.

Other backends are installed separately:

* **PySEQM:** install from `its repository <https://github.com/lanl/PYSEQM>`__.
  ALF's automated tests run against commit ``52850b4``.
* **GPU4PySCF:** needs ``gpu4pyscf``, ``pyscf``, and a CuPy build that matches the
  worker's CUDA runtime.
* **ORCA, VASP, and Q-Chem:** install the executables yourself. These and other
  ASE-supported QM engines can also require environment variables,
  pseudopotential paths, and license or module setup.
* **NeuroChem/ANI, MPI, and GPU toolchains:** these are workflow-specific.

See :doc:`../user_guide/ml_interfaces`, :doc:`../user_guide/qm_interfaces`,
:doc:`../user_guide/samplers`, and :doc:`../user_guide/builders` for
backend-specific configuration patterns.

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
* :doc:`../user_guide/qm_interfaces` for ORCA, VASP, Q-Chem, PySEQM, GPU4PySCF,
  and ASE-backed QM setup.
