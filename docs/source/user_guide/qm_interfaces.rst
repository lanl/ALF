QM Interfaces
=============

QM interfaces connect ALF to electronic structure engines. They receive
candidate structures from the sampler, run or parse a quantum-mechanical
calculation, and return labeled data for storage in ALF's HDF5 training sets.

.. _qm-interfaces-layout:

Where The Interfaces Live
-------------------------

The paths below start with ``alframework.qm_interfaces``.

``ase.generic``
   ``ase_calculator_task``, which labels with any ASE calculator.
``ase.orca``, ``ase.qchem``
   The checks and settings ``ase_calculator_task`` adds for ASE's own ORCA and Q-Chem calculators. See
   :ref:`qm-ase-orca-qchem`.
``ase.workflow``
   ``ase_workflow_task``, which runs several ASE calculations on one geometry. See :ref:`qm-ase-workflows`.
``ase.vasp``
   ``VASP_ase_calculator_task``.
``legacy``
   The older ORCA, Q-Chem and VASP interfaces, which write their programs' input and parse the output
   themselves: ``legacy.orca5_interface``, ``legacy.qchem_DFT_interface`` and ``legacy.vaspase_interface``.
``pyscf``, ``pyseqm_interface``, ``teacher_interface``
   PySCF and GPU4PySCF, PySEQM, and foundation-model teachers.

Moved modules
   These modules moved into ``ase`` and ``legacy``. Their old paths still work, in configurations, imports
   and pickled objects, with a ``FutureWarning`` naming the new path; update configurations to the new
   paths. The tasks and classes keep their names.

   .. list-table::
      :header-rows: 1
      :widths: 50 50

      * - Old path
        - New path
      * - ``ase_calculator_interface.ase_calculator_task``, ``.fresh_scratch_directory``
        - ``ase.generic``, with the same names
      * - ``ase_calculator_interface.VASP_ase_calculator_task``
        - ``ase.vasp.VASP_ase_calculator_task``
      * - ``orca5_interface``
        - ``legacy.orca5_interface``
      * - ``qchem_DFT_interface``
        - ``legacy.qchem_DFT_interface``
      * - ``vaspase_interface``
        - ``legacy.vaspase_interface``

   A moved task does the same as before. Changing a legacy ORCA task to ``ase_calculator_task`` with ASE's
   ORCA calculator is a different interface, with different units; see :ref:`qm-ase-orca-qchem`.

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
     - ``alframework.qm_interfaces.ase.generic.ase_calculator_task`` with ASE's
       ``ase.calculators.orca.ORCA``; see :ref:`qm-ase-orca-qchem`. To repeat a calculation and check that
       it agrees, see :ref:`qm-ase-workflows`. The legacy tasks in ``legacy.orca5_interface`` remain for
       properties ASE doesn't read, such as quadrupoles.
   * - `VASP <https://vasp.at/>`__
     - ``alframework.qm_interfaces.ase.vasp.VASP_ase_calculator_task``
   * - `Q-Chem <https://www.q-chem.com/>`__
     - ``alframework.qm_interfaces.legacy.qchem_DFT_interface.qchem_dft_calculator_task``, or
       ``ase_calculator_task`` with ASE's Q-Chem calculator for energies and forces; see
       :ref:`qm-ase-orca-qchem`.
   * - `PySEQM <https://github.com/lanl/PYSEQM>`__
     - ``alframework.qm_interfaces.pyseqm_interface.pyseqm_task`` for ground states.
       For excited states, see :doc:`excited_states`.
   * - `GPU4PySCF <https://pyscf.org/user/gpu.html>`__
     - ``alframework.qm_interfaces.pyscf.gpu4pyscf.gpu4pyscf_excited_state_task``
       for excited states, or for ground-state DFT with ``nroots: 0``. See :doc:`excited_states` and
       :ref:`configuration-property-names`.
   * - `ASE-supported calculators <https://docs.ase-lib.org/ase/calculators/calculators.html#supported-calculators>`__
     - ``alframework.qm_interfaces.ase.generic.ase_calculator_task``
   * - Foundation models, such as MACE and UMA
     - ``alframework.qm_interfaces.teacher_interface.teacher_batch_task`` or ``teacher_task``,
       to label with a model instead of QM. See :ref:`qm-teacher-labeling`.

When a QM engine already has a reliable ASE calculator, the generic ASE task is
usually the easiest integration route. Write a small QM config that identifies
the ASE calculator class, command, and calculator options, then let ASE handle
input writing, execution, and result extraction.

.. note::

   New VASP configurations should use
   ``alframework.qm_interfaces.ase.vasp.VASP_ase_calculator_task``.
   The older ``alframework.qm_interfaces.legacy.vaspase_interface.VASPGenerator`` helper
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
   input/rem blocks. The legacy ORCA generator uses a neutral singlet; the
   legacy Q-Chem task uses the molecule's ``charge`` and ``mult`` metadata,
   defaulting to 0 and 1, and so do ORCA and Q-Chem through ASE.
   For VASP through the current ASE
   interface, INCAR-style settings come from ``QM_config["input"]`` and ASE's
   VASP calculator behavior. The deprecated legacy VASP helper is the exception
   with ALF compatibility defaults documented further below.

For HPC runs, the QM command and Parsl resource configuration must agree. For
example, an MPI VASP command such as ``srun -n 128 vasp_std`` should be paired
with an ``alf_QM_executor`` configuration that requests the matching nodes,
tasks, walltime, modules, and launcher behavior. See :doc:`parsl` for
practical resource examples.

Legacy ORCA double calculations
   ``orca_double_calculator_task`` runs the same ORCA calculation twice and stores
   the mean. A structure counts as converged only if both runs converge and agree
   within the ``Ediff`` and ``Fdiff`` values in the QM config. The workflow task's
   ``mean_if_agree`` does the same through ASE; see :ref:`qm-ase-workflows`.

Legacy Q-Chem task
   ``qchem_dft_calculator_task`` runs ``QM_run_command`` with the scratch directory as its
   last argument, from ``<scratch>/<prefix>/`` beside ``<prefix>.in``; the command must
   print Q-Chem's output. ``qchem_env_file``, if set, is sourced with Bash first. A command
   or environment that fails, or output missing a requested property or holding a nonfinite
   value, gives an unconverged label with the reason in ``error``. With ``store_path`` set,
   the output and error streams are kept there.

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
   ``alframework.qm_interfaces.legacy.vaspase_interface.VASPGenerator`` is deprecated
   and kept only for older imports. Do not start new configuration files from
   ``vaspase_interface.py``. This compatibility helper is not covered by the
   current integration tests and is unsupported for new workflows. Existing
   users should migrate to
   ``alframework.qm_interfaces.ase.vasp.VASP_ase_calculator_task``
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

Install PySCF with ALF's ``pyscf`` extra, ``pip install "alf[pyscf]"``; the ``full`` extra includes it too.
For the GPU, install GPU4PySCF and a CuPy build matching the QM workers' CUDA, for example
``pip install "alf[gpu4pyscf]"`` for CUDA 12. Each GPU QM worker must see exactly one GPU; see
:ref:`parsl-gpu4pyscf-workers`. GPU4PySCF runs on NVIDIA GPUs only: its kernels are written for CUDA,
and AMD support is an open request upstream (`gpu4pyscf#215
<https://github.com/pyscf/gpu4pyscf/issues/215>`__). On AMD GPUs, use PySCF on the CPU or another QM
code.

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
        "QM_task": "alframework.qm_interfaces.ase.generic.ase_calculator_task",
        "QM_config_path": "my_qm_config.json",
        "QM_scratch_dir": "qm_scratch/"
      }

   .. code-block:: json

      {
        "ASE_calculator": "ase.calculators.mycode.MyCode",
        "QM_run_command": "mycode_executable",
        "input": {
          "xc": "PBE",
          "basis": "def2-SVP"
        }
      }

   Put the calculator's options in ``input``. Older configurations that list them beside the ALF settings
   still work, but a configuration can't mix the two. ``ASE_calculator``, ``QM_run_command``, ``batching``,
   ``initial_magnetic_moments`` and ``remove_scratch_on_success`` are ALF settings, never calculator options,
   and ALF sets ``atoms``, ``directory``, ``command`` and ``profile`` itself, so ``input`` can't contain them. ``label`` may be a file name, without a directory.
   Calculators that run in ALF's process, such as EMT, need no ``QM_run_command``.

   ALF checks the configuration when the coordinator starts. For each label it then creates a fresh
   calculator in a fresh scratch directory (moving one left by an earlier attempt to ``.previous-N``),
   calculates on a copy of the molecule's geometry, and keeps only results from that calculation. A label
   fails, and the molecule returns unconverged, when:

   * the calculator sets ``converged`` to anything but ``True``;
   * a requested property is missing, not numeric, the wrong shape, or not finite;
   * the calculator moved the geometry.

   A failed label records what went wrong in ``qm_error`` and ``qm_error_type``, and the step it happened
   at in ``qm_failed_stage``. In a sequential batch the other structures keep their labels.

   Scratch directories are kept by default. With ``"remove_scratch_on_success": true``, ALF deletes a
   molecule's scratch directory once its label has passed every check; a failed label's directory is always
   kept for inspection. Turn it on for production runs whose calculators leave large files, such as VASP's
   ``CHGCAR`` and ``WAVECAR``.

   ``initial_magnetic_moments`` gives a starting moment, in Bohr magnetons, for each element:

   .. code-block:: json

      {"initial_magnetic_moments": {"Fe": 2.2, "Co": 1.6}}

   ASE calculators take initial moments per atom, so a per-atom list such as VASP's ``magmom`` fits only one
   structure size, while ALF labels structures of many sizes. ALF sets each atom's moment from its element
   on the copy it calculates on, and elements left out start at zero. Calculators that read initial moments
   from the atoms, such as VASP, then use them. A configuration can't give both this and ``magmom`` in
   ``input``, and ORCA and Q-Chem refuse it: they take the spin state from the multiplicity.

   Property names in ``properties_list`` must be the calculator's property names, such as ``energy`` and
   ``forces``; the first entry of each property gives the HDF5 dataset to store it in. Calculators built on
   ASE's profiles take their command from their own ASE configuration; ALF passes ``QM_run_command`` to them
   only through settings written for that calculator, as it does for ORCA below.

.. _qm-ase-orca-qchem:

ORCA And Q-Chem Through ASE
---------------------------

``ase_calculator_task`` runs ORCA and Q-Chem with ASE's own calculators, ``ase.calculators.orca.ORCA`` and
``ase.calculators.qchem.QChem``: ASE writes the input and reads the output. ALF adds checks of the
configuration at startup and supplies each molecule's charge and multiplicity. A subclass of either
calculator is treated as any other ASE calculator. Both label nonperiodic molecules, as single points.

ORCA
   This needs an ASE version that provides ``OrcaProfile``. A configuration:

   .. code-block:: json

      {
        "ASE_calculator": "ase.calculators.orca.ORCA",
        "QM_run_command": "/path/to/orca-or-wrapper",
        "input": {
          "orcasimpleinput": "EnGrad PBE def2-SVP TightSCF",
          "orcablocks": "%pal nprocs 4 end\n%maxcore 2000",
          "charge": 0,
          "mult": 1
        }
      }

   ORCA labels ``energy``, ``free_energy``, ``forces`` and ``dipole``. Forces need ``EnGrad``, and a run that
   doesn't end with ORCA's normal-termination line fails. ``orcasimpleinput`` must name the method and basis,
   and can't request an optimization, dynamics or a scan; coordinates come from the molecule, so
   ``orcablocks`` can't contain a ``coords`` block. Set cores with ``%pal`` and match them to the QM
   executor's resources.

   ``QM_run_command`` is the ORCA executable, or a wrapper script that ends with
   ``exec /path/to/orca "$@"``; ASE adds the input file name and captures the output. It can't contain
   ``source``, redirection or ASE's ``PREFIX`` placeholder, so load modules in the executor's ``worker_init``
   or in the wrapper. See `ASE's ORCA documentation <https://docs.ase-lib.org/ase/calculators/orca.html>`__.

Q-Chem
   ASE's Q-Chem calculator labels energies and forces:

   .. code-block:: json

      {
        "ASE_calculator": "ase.calculators.qchem.QChem",
        "QM_run_command": "/path/to/qchem-wrapper PREFIX.inp PREFIX.out",
        "input": {"method": "pbe", "basis": "def2-svp", "nt": 4}
      }

   ``QM_run_command`` is the whole command, with ASE's ``PREFIX`` placeholders and any launch flags; it is
   used as written. Without it, ASE builds its usual command from ``np`` and ``nt``. This is not the
   command form of the legacy Q-Chem task, so its wrappers can't be reused unchanged. ``jobtype`` may be
   ``SP`` or ``FORCE``; leave it out and ASE chooses ``FORCE`` when forces are requested. Other ``$rem``
   values follow ASE's conventions, usually strings. For other Q-Chem blocks and properties, use the legacy
   task.

Charge, spin and units
   ALF copies the molecule's ``charge`` and ``mult`` metadata into the atoms given to the calculator; the
   configured ``charge`` and ``mult`` (``multiplicity`` for Q-Chem) apply when a molecule has none, and
   otherwise 0 and 1. Both must be integers, and the multiplicity positive. The label's metadata records
   the charge and multiplicity used, so the shard stores the same state as the calculation. Other ASE
   calculators read ``atoms.info`` themselves, if at all.

   ASE gives energies in eV and forces in eV/Angstrom, so their multipliers in ``properties_list`` are
   ``1.0``:

   .. code-block:: json

      {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}

   ORCA dipoles are in electron-Angstrom about the coordinate origin: ASE moves ORCA's center-of-mass dipole
   there, which matters for charged molecules. The legacy ORCA task reports dipoles differently.

   **Changing a legacy ORCA run's task to this one also changes its units.** The legacy task's Hartree and
   Hartree/Bohr multipliers would convert ASE's results a second time: set them to ``1.0``, drop properties
   ASE doesn't read, and start in new output directories. For quadrupoles and Hirshfeld charges, keep the
   legacy task. ``examples/simple_water`` labels with this configuration.

PySCF and GPU4PySCF
   ``pyscf_task`` and ``gpu4pyscf_excited_state_task`` call their calculators directly, with their own state
   names, gaps, energy offsets, periodic stress and GPU checks; use them rather than ``ase_calculator_task``
   with ``PySCFCalculator``.

Before a production run, label one structure with ``--test_qm`` on the installation you'll use, and check
the method, state, units and properties against the program's own output.

.. _qm-ase-workflows:

Several ASE Calculations On One Geometry
----------------------------------------

``alframework.qm_interfaces.ase.workflow.ase_workflow_task`` takes the same arguments as
``ase_calculator_task`` and runs a list of stages, each an ASE configuration as above with a ``name``, in
one QM worker:

.. code-block:: json

   {
     "result_mode": "last",
     "stages": [
       {"name": "guess", "ASE_calculator": "ase.calculators.orca.ORCA",
        "QM_run_command": "/path/to/orca-or-wrapper",
        "properties": ["energy"],
        "input": {"orcasimpleinput": "PBE def2-SVP TightSCF"}},
       {"name": "target", "ASE_calculator": "ase.calculators.orca.ORCA",
        "QM_run_command": "/path/to/orca-or-wrapper",
        "inputs_from_stages": [
          {"stage": "guess", "source": "orca.gbw", "destination": "guess.gbw"}
        ],
        "input": {"orcasimpleinput": "EnGrad PBE def2-TZVP TightSCF MORead",
                  "orcablocks": "%moinp \"guess.gbw\""}}
     ]
   }

Every stage runs on the original sampled geometry with a fresh calculator in
``<scratch>/<molecule>/<stage name>``. Stage names must be unique and usable as directory names. ALF validates
the workflow when the coordinator starts.

In ``last`` mode, ``inputs_from_stages`` declares files to pass on from earlier successful stages of the same
attempt. Give the source and destination as single filenames and use each destination once per receiving
stage. ALF places nonempty regular files before constructing the receiving calculator. The last stage to read
a file gets the file itself, moved without copying, so a large ``WAVECAR`` or ``CHGCAR`` isn't duplicated;
earlier readers get copies. Configure the receiving backend to read those files and check its output to
confirm it did so. A missing or unreadable file fails the label at ``<stage>:input_files``.

Each stage may set its own ``initial_magnetic_moments``. ``remove_scratch_on_success`` belongs to the
workflow, beside ``stages``, and removes the molecule's whole directory, with every stage, after a good label.

Every stage must succeed. Stages use the full ``properties_list`` by default. In ``last`` mode a preparatory
stage may request a subset via ``properties``. The final stage supplies the full training label.

``result_mode`` picks the label:

``last`` (the default)
   The final stage's results. See ``examples/staged_ase_qm`` for one-structure recipes for ORCA orbital warm
   starts and VASP charge-density restarts.

``mean_if_agree``
   Two or more identical independent stages using the full property list. The label is their mean if every pair
   agrees within ``agreement``, which gives the energy tolerance in eV and the force tolerance in
   eV/Angstrom:

   .. code-block:: json

      {"agreement": {"energy": 0.027211386024367243, "forces": 0.005142206709048065}}

   These are 1e-3 Hartree and 1e-4 Hartree/Bohr, the legacy ORCA double calculation's ``Ediff`` and
   ``Fdiff``. Every energy difference, and every force component's, must be below its tolerance, and the
   repeats' labels must have the same shapes. This catches a calculation that doesn't reproduce; it can't
   tell whether the electronic state is the right one. The ionic-liquid example uses it.

A failed stage stops the workflow and leaves the molecule unconverged. ``qm_failed_stage`` names the stage and
step, such as ``target:calculation``. ``qm_workflow`` records each stage's name, calculator, directory,
convergence, electronic state and declared inputs. For ``mean_if_agree`` it also records the largest
deviations. A retried structure's directory, with all its stages, moves to ``.previous-N``. Sequential QM
batching applies to whole workflows.

Engines Without An ASE Calculator
---------------------------------

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
* :doc:`ASE calculator interface module <../api_documentation/alframework.qm_interfaces.ase.generic>`
* :doc:`ORCA through ASE <../api_documentation/alframework.qm_interfaces.ase.orca>`
* :doc:`Q-Chem through ASE <../api_documentation/alframework.qm_interfaces.ase.qchem>`
* :doc:`ASE workflow task <../api_documentation/alframework.qm_interfaces.ase.workflow>`
* :doc:`VASP ASE task <../api_documentation/alframework.qm_interfaces.ase.vasp>`
* :doc:`ORCA interface module <../api_documentation/alframework.qm_interfaces.legacy.orca5_interface>`
* :doc:`QChem interface module <../api_documentation/alframework.qm_interfaces.legacy.qchem_DFT_interface>`
* :doc:`PySEQM interface module <../api_documentation/alframework.qm_interfaces.pyseqm_interface>`
* :doc:`PySCF interface module <../api_documentation/alframework.qm_interfaces.pyscf>`
* :doc:`PySCF ASE calculator <../api_documentation/alframework.qm_interfaces.pyscf.calculator>`
* :doc:`GPU4PySCF interface module <../api_documentation/alframework.qm_interfaces.pyscf.gpu4pyscf>`
* :doc:`Legacy VASP interface module <../api_documentation/alframework.qm_interfaces.legacy.vaspase_interface>`
* :doc:`Parsl execution guide <parsl>`

Related Examples
----------------

See :doc:`../examples/simple_water` and :doc:`../examples/reactive_sampling`
for ORCA-backed workflows, and :doc:`../examples/molten_salt` for a
VASP-backed workflow. :doc:`../examples/periodic_water` labels periodic boxes with ``pyscf_task``, and
``simple_water`` has a PySCF configuration beside its ORCA one.
