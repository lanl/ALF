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
   * - `PySCF <https://pyscf.org>`__ and `GPU4PySCF <https://github.com/pyscf/gpu4pyscf>`__
     - ``alframework.qm_interfaces.pyscf.pyscf_task``, on CPU or GPU, for molecules and periodic
       cells. The default for new examples, since both are free and pip-installable. See
       :ref:`qm-interfaces-pyscf`.
   * - `ORCA <https://www.faccts.de/orca/>`__
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``, or
       ``alframework.qm_interfaces.orca5_interface.orca_double_calculator_task``
       to run each calculation twice and accept it only if the results agree.
   * - `VASP <https://vasp.at/>`__
     - ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   * - `Q-Chem <https://www.q-chem.com/>`__
     - ``alframework.qm_interfaces.qchem_DFT_interface.qchem_dft_calculator_task``
   * - `PySEQM <https://github.com/lanl/PYSEQM>`__
     - ``alframework.qm_interfaces.pyseqm_interface.pyseqm_task`` for ground states.
       For excited states, see :doc:`excited_states`.
   * - `GPU4PySCF <https://pyscf.org/user/gpu.html>`__
     - ``alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task``
       for excited states, or for ground-state DFT with ``nroots: 0``. See :doc:`excited_states` and
       :ref:`configuration-property-names`.
   * - `ASE-supported calculators <https://docs.ase-lib.org/ase/calculators/calculators.html#supported-calculators>`__
     - ``alframework.qm_interfaces.ase_calculator_interface.ase_calculator_task``
   * - Foundation models, such as MACE and UMA
     - ``alframework.qm_interfaces.teacher_interface.teacher_batch_task`` or ``teacher_task``,
       to label with a model instead of QM. See :ref:`qm-teacher-labeling`.

When a QM engine already has a reliable ASE calculator, the generic ASE task is
usually the easiest integration route. Write a small QM config that identifies
the ASE calculator class, command, and calculator options, then let ASE handle
input writing, execution, and result extraction.

.. note::

   New VASP configurations should use
   ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``.
   The older ``alframework.qm_interfaces.vaspase_interface.VASPGenerator`` helper
   is deprecated and kept only for backward compatibility. For custom VASP-style
   workflows, prefer the generic ``ase_calculator_task`` whenever the engine can
   be represented as an ASE calculator.

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
   storing data. See :doc:`units` for how these multipliers are applied.

Batched labeling
   A ``batching`` block in the QM config labels several structures in one task,
   either natively or by job packing. See :doc:`batching`.

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

.. note::
   ALF does not choose a universally appropriate QM method, basis set,
   pseudopotential, dispersion correction, or charge / spin state setting for a
   workflow. These choices should be made explicitly in the engine-specific QM
   configuration and checked against the intended chemistry. For ORCA and
   Q-Chem, the method, basis, and property requests come from the configured
   input/rem blocks, while the current task wrappers call the generators as
   neutral singlets unless customized. For VASP through the current ASE
   interface, INCAR-style settings come from ``QM_config["input"]`` and ASE's
   VASP calculator behavior. The deprecated legacy VASP helper is the exception
   with ALF compatibility defaults documented further below.

For HPC runs, the QM command and Parsl resource configuration must agree. For
example, an MPI VASP command such as ``srun -n 128 vasp_std`` should be paired
with an ``alf_QM_executor`` configuration that requests the matching nodes,
tasks, walltime, modules, and launcher behavior. See :doc:`parsl` for
practical resource examples.

ORCA double calculations
   ``orca_double_calculator_task`` runs the same ORCA calculation twice and stores
   the mean. A structure counts as converged only if both runs converge and agree
   within the ``Ediff`` and ``Fdiff`` values in the QM config.

PySEQM ground states
   ``pyseqm_task`` evaluates a batch of molecules in one call, and handles atom
   sorting and padding internally. It has no single-structure form, so the QM config
   must select ``"mode": "native"``:

   .. code-block:: json

      {
        "method": "AM1",
        "scf_eps": 1e-8,
        "device": "cuda",
        "batching": {"mode": "native", "batch_size": 64}
      }

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``method``
        - Semiempirical method. The default is ``AM1``.
      * - ``scf_eps``, ``scf_converger``
        - SCF settings. The defaults are ``1e-8`` and ``[2]``.
      * - ``pyseqm_parameters``
        - Optional extra PySEQM parameters, applied on top of these fields.
      * - ``device``
        - Torch device. The default is ``cpu``. For GPU labeling, place
          ``alf_QM_executor`` on GPU nodes.

   PySEQM reports energies in eV and forces in eV/Angstrom. It doesn't support
   periodic systems, and a molecule's ``charge`` and ``mult`` come from its metadata.
   A molecule whose SCF doesn't converge is marked unconverged, and the rest of the
   batch is kept.

Legacy VASP helper
   ``alframework.qm_interfaces.vaspase_interface.VASPGenerator`` is deprecated
   and kept only for older imports. Do not start new configuration files from
   ``vaspase_interface.py``. This compatibility helper is not covered by the
   current integration tests and is unsupported for new workflows. Existing
   users should migrate to
   ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   or the generic ``ase_calculator_task`` described above.

   The legacy helper also applies ALF-specific VASP defaults before merging
   user-provided ``vasp_options``:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - VASP setting
        - Legacy default
      * - ``xc``
        - ``"pbe"``
      * - ``prec``
        - ``"Accurate"``
      * - ``ncore``
        - ``1`` unless ``vasp_options["ncore"]`` is provided.
      * - ``lreal``
        - ``"Auto"``
      * - ``nelm``
        - ``120`` unless ``vasp_options["nelm"]`` is provided.
      * - ``ivdw``
        - ``0`` unless ``vasp_options["ivdw"]`` is provided.

   These are compatibility defaults from the old helper, not universal
   recommendations for VASP calculations. In particular, users should be aware
   that the legacy path sets ``LREAL = Auto`` by default; review this setting
   when reproducing old calculations, comparing force labels, or migrating to
   the ASE calculator interface. The old helper also maps ``vasp_options["kpoints"]``
   to ASE's ``kpts`` option and falls back to atomic-number magnetic moments
   unless ``magmom`` is provided.

.. _qm-interfaces-pyscf:

PySCF And GPU4PySCF
-------------------

``pyscf_task`` runs DFT with PySCF on the CPU or GPU4PySCF on a GPU, one structure per task. It labels
molecules and periodic cells; ALF picks which from each structure's ``pbc``. The same calculation is
available outside ALF as an ASE calculator; see `Outside ALF`_.

PySCF comes with ALF. For the GPU, install GPU4PySCF and a CuPy build matching the QM workers' CUDA, for
example ``pip install "alf[gpu4pyscf]"`` for CUDA 12. Each GPU QM worker must see exactly one GPU; see
:ref:`parsl-gpu4pyscf-workers`.

**Properties.** ``pyscf_task`` uses state names; see :ref:`configuration-property-names`:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``E0``, ``F0``
     - Ground-state energy (eV) and forces (eV/Å), for molecules and periodic cells. Store them as
       ``energy`` and ``forces`` for ground-state training and fine-tuning.
   * - ``E1``, ``F1``, ... and ``dE01``, ...
     - TDA excited states and gaps, for singlet molecules, with ``nroots`` excited states. See
       :doc:`excited_states`.
   * - ``stress``
     - Periodic stress in eV/Å³, in ASE's order and sign, ``(1/V) dE/dε``.

For ground-state labels:

.. code-block:: json

   "properties_list": {"E0": ["energy", "system", 1.0], "F0": ["forces", "atomic", 1.0]}

**Settings,** in the QM configuration:

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Setting
     - Default
     - Meaning
   * - ``backend``
     - ``cpu``
     - ``cpu`` for PySCF, ``gpu`` for GPU4PySCF.
   * - ``xc``, ``basis``
     - ``pbe``, ``def2-svp``
     - Functional and, for molecules, basis set.
   * - ``charge``, ``multiplicity``
     - ``0``, ``1``
     - Used when a structure's metadata has no ``charge`` or ``mult``. Open shells use UKS. Periodic cells
       must be neutral.
   * - ``density_fit``, ``auxbasis``
     - ``true``, ``null``
     - Density fitting for molecules.
   * - ``nroots``
     - ``0``
     - Number of TDA excited states, for molecules. ``0`` labels the ground state only.
   * - ``scf_conv_tol``, ``scf_max_cycle``
     - ``1e-10``, ``100``
     - SCF convergence. ``1e-8`` is usually enough for periodic cells.
   * - ``num_threads``, ``max_memory_mb``, ``verbosity``
     - ``1``, ``null``, ``0``
     - CPU threads, memory, and PySCF's output level.
   * - ``periodic``
     - see below
     - Settings for periodic cells.

The ``periodic`` block:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Setting
     - Default
     - Meaning
   * - ``basis``, ``pseudo``
     - ``gth-dzvp``, ``gth-pbe``
     - Basis and pseudopotentials for periodic cells.
   * - ``kpts``
     - ``[1, 1, 1]``
     - k-point mesh. ``[1, 1, 1]`` is the Γ point.
   * - ``df``
     - ``multigrid``
     - How the Coulomb and exchange-correlation terms are computed: ``multigrid`` (Γ point only, fast,
       no stress on the CPU), ``fftdf`` (k-points and stress), or ``gdf`` (GPU only).
   * - ``ke_cutoff``
     - ``null``
     - Kinetic-energy cutoff in Hartree, for ``fftdf``; PySCF chooses one if unset.
   * - ``smearing_eV``
     - ``null``
     - Fermi smearing width. The energy is then the free energy ``E - σS``, whose gradient the forces
       are. Useful for metals, or when the SCF won't converge.

What each backend gives. The GPU column follows GPU4PySCF 1.9's release notes; ALF has so far checked
the CPU column, against finite differences and PySCF itself:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Calculation
     - CPU (PySCF)
     - GPU (GPU4PySCF)
   * - Molecule, ground state, forces
     - Yes
     - Yes
   * - Molecule, excited states, forces
     - Yes, with ``density_fit: false``
     - Yes
   * - Periodic, forces at Γ
     - ``multigrid`` or ``fftdf``
     - ``multigrid``, ``fftdf``, or ``gdf``
   * - Periodic, forces with k-points
     - ``fftdf``
     - ``fftdf`` or ``gdf``
   * - Periodic, stress
     - ``fftdf``
     - Yes
   * - Periodic, excited states
     - No
     - No

For nonadiabatic couplings, optical properties, or implicit solvent, use ``gpu4pyscf_excited_state_task``;
see :doc:`excited_states`.

A structure whose SCF or gradient fails is marked unconverged, with ``qm_error`` and ``qm_failed_stage`` in
its metadata, and isn't stored. Randomly packed structures can be hard for periodic DFT: a 5 Å box of five
waters, built with ``min_dist: 1.25`` and ``shake: 0.05``, had a 0.2 eV gap and didn't converge, while
boxes built with ``min_dist: 1.6`` and ``shake: 0.02`` converged at once. Prefer realistic builder
settings to smearing.

On a laptop CPU: a nine-water cluster at ωB97X/def2-SVP took about 25 s, and a periodic box of 15 waters
(7.7 Å) at PBE/GTH-DZVP with ``multigrid`` about a minute.

Outside ALF
~~~~~~~~~~~

``alframework.qm_interfaces.pyscf.calculator.PySCFCalculator`` takes the same settings as keyword
arguments, and works anywhere ASE does:

.. code-block:: python

   from ase import units
   from ase.io import read
   from ase.md.verlet import VelocityVerlet
   from alframework.qm_interfaces.pyscf.calculator import PySCFCalculator

   atoms = read("water_box.xyz")  # A periodic cell
   atoms.calc = PySCFCalculator(backend="gpu", xc="pbe", periodic={"df": "fftdf"})
   atoms.get_stress()
   VelocityVerlet(atoms, timestep=0.5 * units.fs).run(10)

Besides ``energy``, ``forces``, and ``stress``, it gives ``state_energies`` and ``state_forces`` for
excited states, through ``calc.get_property``. ``calc.metadata`` records the method, timings, and
versions. PySCF's own ASE calculator, ``pyscf.pbc.tools.pyscf_ase.PySCF``, needs a ready-made PySCF
method instead, and scales stress by one factor of the Bohr radius where three are needed.

Foundation-Model Teachers
-------------------------

A foundation model such as MACE or UMA can label structures in place of QM, so that a smaller
HIPPYNN student learns from it. Its settings, safeguards, and a walkthrough are in
:doc:`teacher_student`.

.. _qm-interface-new-engine-template:

Adding New QM Engines
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
   ``properties_list``. Custom tasks should document the units they place in
   ``MoleculesObject.results`` so the master configuration can apply the
   correct HDF5 conversion factor.

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
* :doc:`PySEQM interface module <../api_documentation/alframework.qm_interfaces.pyseqm_interface>`
* :doc:`PySCF interface module <../api_documentation/alframework.qm_interfaces.pyscf>`
* :doc:`PySCF ASE calculator <../api_documentation/alframework.qm_interfaces.pyscf.calculator>`
* :doc:`GPU4PySCF interface module <../api_documentation/alframework.qm_interfaces.pyscf.gpu4pyscf>`
* :doc:`Legacy VASP interface module <../api_documentation/alframework.qm_interfaces.vaspase_interface>`
* :doc:`Parsl execution guide <parsl>`

Related Examples
----------------

See :doc:`../examples/simple_water` and :doc:`../examples/reactive_sampling`
for ORCA-backed workflows, and :doc:`../examples/molten_salt` for a
VASP-backed workflow.
