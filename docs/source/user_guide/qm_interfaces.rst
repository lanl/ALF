QM Interfaces
=============

QM interfaces connect ALF to electronic structure engines. They receive
candidate structures from the sampler, run or parse a quantum-mechanical
calculation, and return labeled data for storage in ALF's HDF5 training sets.

Supported QM Packages
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Package
     - Associated interface
   * - ORCA
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - VASP
     - ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   * - QChem
     - ``alframework.qm_interfaces.qchem_DFT_interface.qchem_dft_calculator_task``
   * - ASE-supported calculators
     - ``alframework.qm_interfaces.ase_calculator_interface.ase_calculator_task``

When a QM engine already has a reliable ASE calculator, the generic ASE task is
usually the easiest integration route. Write a small QM config that identifies
the ASE calculator class, command, and calculator options, then let ASE handle
input writing, execution, and result extraction.

.. _qm-interface-extension-points:

How ALF Uses QM Interfaces
--------------------------

ALF uses QM code through a small set of master-configuration hooks:

QM task
   The ``QM_task`` field in ``master_config.json`` is an import string for the
   labeling task. ALF submits this task through Parsl on the
   ``alf_QM_executor`` executor.

QM configuration
   The ``QM_config_path`` field points to an engine-specific JSON file. This
   file usually contains the executable command, method settings, CPU count,
   input blocks, and any engine-specific options.

Scratch directory
   The ``QM_scratch_dir`` field controls where per-structure calculation
   directories are created. QM tasks typically create one subdirectory per
   ``MoleculesObject`` id.

Properties
   The ``properties_list`` field defines which properties ALF requests and how
   those properties are written to HDF5. It also records whether each property
   is system-level or atom-level and includes the unit conversion used when
   storing data.

Common Configuration Fields
---------------------------

Master configuration
   These fields connect the ALF active-learning loop to a QM engine.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``QM_task``
        - Import string for the labeling task.
      * - ``QM_config_path``
        - JSON file containing engine-specific QM settings.
      * - ``QM_scratch_dir``
        - Directory for per-structure QM input, output, and scratch files.
      * - ``properties_list``
        - Requested properties, HDF5 dataset names, property scope, and unit
          conversion factors.

QM configuration
   Exact fields differ by backend, but QM config files commonly include:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``QM_run_command``
        - Executable or launch command used by the QM task.
      * - ``ncpu``
        - CPU count passed to engines or input writers when supported.
      * - environment file
        - Optional shell setup file, such as ``orca_env_file`` or
          ``qchem_env_file``.
      * - method/input blocks
        - Engine-specific method, basis, SCF, memory, k-point, and property
          settings.
      * - scratch/output paths
        - Backend-specific controls for where input, output, and restart files
          are written.

For HPC runs, the QM command and Parsl resource configuration must agree. For
example, an MPI VASP command such as ``srun -n 128 vasp_std`` should be paired
with an ``alf_QM_executor`` configuration that requests the matching nodes,
tasks, walltime, modules, and launcher behavior. See :doc:`parsl` for
practical resource examples.

.. _qm-interface-new-engine-template:

Template For Adding A New QM Engine
-----------------------------------

Preferred ASE route
   Use this path when ASE already provides a calculator for the engine.
   Configure ``ase_calculator_task`` as the QM task and include an
   ``ASE_calculator`` import string in the QM config:

   .. code-block:: json

      {
        "QM_task": "alframework.qm_interfaces.ase_calculator_interface.ase_calculator_task",
        "QM_config_path": "my_qm_config.json",
        "QM_scratch_dir": "qm_scratch/"
      }

   .. code-block:: json

      {
        "ASE_calculator": "ase.calculators.mycode.MyCode",
        "QM_run_command": "mycode_executable",
        "xc": "PBE",
        "basis": "def2-SVP"
      }

   The generic task loads the ASE calculator class, creates a
   molecule-specific scratch directory, runs ``calc.calculate(...)``, stores
   ``calc.results``, sets the convergence flag from ``calc.converged``, and
   returns the updated ``MoleculesObject``.

Custom parser route
   Use this path when ASE does not support the engine or when ALF needs
   backend-specific parsing and convergence checks. Define a Parsl task with
   the ALF QM task signature:

   .. code-block:: python

      import os
      from parsl import python_app


      @python_app(executors=["alf_QM_executor"])
      def my_qm_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
          molecule_id = molecule_object.get_moleculeid()
          directory = os.path.join(QM_scratch_dir, molecule_id)
          os.makedirs(directory, exist_ok=False)

          atoms = molecule_object.get_atoms()
          requested_properties = list(properties_list.keys())

          write_mycode_input(atoms, QM_config, directory, requested_properties)
          run_mycode(QM_config["QM_run_command"], directory)

          results = parse_mycode_output(directory, requested_properties)
          converged = parse_mycode_convergence(directory)

          molecule_object.store_results(results)
          molecule_object.set_converged_flag(converged)
          return molecule_object

   The ``results`` dictionary should use the same property names requested in
   ``properties_list``. Energies and forces should be converted consistently
   with the unit factors in the master configuration before they are stored or
   interpreted downstream.

Implementation checklist
   Before using a new QM backend in production, check that:

   * The QM task runs on ``alf_QM_executor`` without requiring ML/GPU imports.
   * The scratch directory is unique for each ``MoleculesObject`` id.
   * Requested properties match the keys in ``properties_list``.
   * Convergence is set explicitly with ``set_converged_flag``.
   * Failed or incomplete calculations return a non-converged
     ``MoleculesObject`` or raise an error that ALF can record.
   * The launch command matches the Parsl resource allocation, especially for
     MPI engines.

QM Interface API Links
----------------------

You can link from this guide directly to API pages:

* :doc:`QM interfaces package API <../api_documentation/alframework.qm_interfaces>`
* :doc:`ASE calculator interface module <../api_documentation/alframework.qm_interfaces.ase_calculator_interface>`
* :doc:`ORCA interface module <../api_documentation/alframework.qm_interfaces.orca5_interface>`
* :doc:`QChem interface module <../api_documentation/alframework.qm_interfaces.qchem_DFT_interface>`
* :doc:`VASP interface module <../api_documentation/alframework.qm_interfaces.vaspase_interface>`
* :doc:`Parsl execution guide <parsl>`

Related Examples
----------------

See :doc:`../examples/simple_water` and :doc:`../examples/reactive_sampling`
for ORCA-backed workflows, and :doc:`../examples/molten_salt` for a
VASP-backed workflow.
