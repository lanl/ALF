.. _qm-teacher-labeling:

Teacher–Student Labeling (Distillation)
=======================================

A foundation model such as MACE or UMA can label structures in place of QM. ALF then trains a
smaller, faster HIPPYNN model, the *student*, on those labels. This is a form of knowledge
distillation: the student learns to reproduce the teacher on the structures ALF's sampling finds,
at a fraction of the teacher's cost to run.

Storage, training, sampling, and promotion work exactly as with QM labels. Only the labeling task
changes. Only ground-state energies and forces are supported.

The teacher is a pretrained checkpoint as its authors publish it, such as MACE-MP-0 or
uma-s-1p2p1. A model that ALF has fine-tuned isn't a supported teacher; use it directly instead,
as :doc:`fine_tuning` describes.

When To Use It
--------------

* You need a model that is much faster than the foundation model, for long or large simulations.
* The foundation model is accurate enough for your system, so its labels are worth learning.
* QM labels would be too expensive for the amount of data the student needs.

If the foundation model itself isn't accurate enough, label with QM instead, or fine-tune the
foundation model on QM labels with :doc:`fine_tuning`.

Setting It Up
-------------

Install the teacher's package in the QM workers' environment, with ALF's ``teacher-mace`` or
``teacher-uma`` extra. Use a separate environment for each, because ``mace-torch`` and
``fairchem-core`` have needed different versions of ``e3nn``.

Download the checkpoint once, and point the configuration at the local file. Compute nodes often
have no network, and ALF records the checkpoint's hash with every label.

In the QM configuration:

.. code-block:: json

   {
     "teacher": {
       "loader": "alframework.qm_interfaces.teacher_interface.load_mace_teacher",
       "name": "mace-mp-0-small",
       "checkpoint": "teachers/mace-mp-0-small.model",
       "dtype": "float64"
     },
     "batching": {"mode": "native", "batch_size": 32, "max_wait_seconds": 30.0}
   }

Set ``QM_task`` to ``alframework.qm_interfaces.teacher_interface.teacher_batch_task``, which
evaluates each batch in one pass. ``teacher_task`` labels one structure per task, alone or with
``"mode": "sequential"``.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Meaning
   * - ``loader``
     - ``load_mace_teacher``, ``load_uma_teacher``, or your own function; see below.
   * - ``checkpoint``
     - Local checkpoint file. A relative path is taken from the master directory.
   * - ``name``
     - Recorded with every label. The default is the checkpoint's file name.
   * - ``task``
     - Required for UMA, with no default, because each task is a different level of theory:
       ``omol`` for molecules, ``omat`` for materials, and so on.
   * - ``dtype``
     - ``float64`` (the default) or ``float32``. Prefer ``float64`` for labels.
   * - ``device``
     - Optional. By default, each worker picks a GPU as the samplers do; see :doc:`parsl`.
   * - ``allow_periodic``
     - ``true`` lets a teacher whose task was trained only on non-periodic structures, such as UMA's
       ``omol``, label periodic ones. By default ALF refuses them; see below.
   * - ``allow_mixed_labels``
     - ``true`` lets a run add teacher labels to an ``h5_dir`` that already holds QM labels or
       another teacher's. By default ALF refuses, because a student would train on mixed
       reference levels.
   * - ``options``
     - Extra keyword arguments for the model's constructor.

``properties_list`` must contain ``energy`` (system) and may contain ``forces`` (atomic), each with a
unit factor of ``1.0``, since teachers give eV and eV/Å. ALF stops at startup otherwise. This catches
the Hartree factors in a QM configuration's ``properties_list``.

How labels are judged
   A label is converged if its energy and forces are finite. There's no force limit, because large
   forces are often real and the student needs them. Use the training filters,
   ``remove_high_forces_cut`` and ``remove_high_forces_std``, to keep outliers out of training while
   keeping them stored. A structure with an element the teacher wasn't trained on is marked
   unconverged with a reason. So is a charged or open-shell structure, for a teacher that assumes
   neutral singlets, as MACE does. UMA's ``omol`` task receives charge and multiplicity from each
   molecule's ``charge`` and ``mult`` metadata, which default to 0 and 1. A periodic structure is
   marked unconverged for ``omol``, which was trained only on non-periodic structures, unless
   ``allow_periodic`` is ``true``. If one structure breaks a batch, ALF splits the batch and
   retries, so only that structure is lost.

Choosing a UMA task
   ``task`` picks the level of theory, and fairchem's
   `task notes <https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.md>`__ say what
   each suits. ``omol`` is for molecules and clusters, with charge and spin multiplicity. Its
   training data are all non-periodic, so fairchem advises caution with periodic systems, and ALF
   refuses them unless you set ``allow_periodic``. ``omat`` is for inorganic materials, ``omc`` for
   molecular crystals, ``odac`` for metal-organic frameworks, and ``oc20`` for catalysis. These
   periodic tasks take no charge or spin, so ALF gives them neutral structures only.

   The student must see the same structures as the teacher. A non-periodic student, with
   ``cell_key`` set to ``null``, can't learn a periodic teacher's labels, and a periodic student
   can't learn a cluster teacher's labels. Set the builder's ``periodic`` setting to match; see
   :doc:`builders`.

What's recorded
   Each label carries ``label_provenance``: the teacher's name, package and version, checkpoint path
   and SHA-256, task, data type, and units. ALF stores it on each HDF5 shard, as the
   ``label_provenance`` attribute. ``alframework.tools.tools.read_label_provenance`` reads it back.
   QM shards don't have it. After each training, the ensemble directory gets ``label_sources.json``,
   listing the teachers its data came from and any shards without a recorded source.

Practical notes
   * Teacher labels are cheap, so sampling and training become the slow steps. Larger
     ``save_h5_threshold`` and batch sizes than for QM usually make sense.
   * Choose a teacher trained on chemistry like yours. For example, MACE-MP is trained on materials,
     and MACE-OFF on organic molecules with a limited set of elements.
   * A student with a shorter cutoff than the teacher's reach can't reproduce it exactly.
   * Check each checkpoint's license. Some have restrictions, and data labeled with a teacher may be
     subject to its terms. ALF ships no model weights; see
     :ref:`fine-tuning-licenses` for where MACE and UMA models come from and their terms.
   * A worker uses its teacher from one thread at a time, because models such as MACE keep state
     between calls.

Your own teacher
   A loader is called as ``loader(settings, device)``, and returns an object with
   ``evaluate(atoms_list)``, which gives energies in eV and forces in eV/Å, in input order. It also
   needs ``package``, ``supported_numbers`` (a set of atomic numbers, or ``None``), and
   ``supports_charge_spin``. ``TorchSimTeacher`` wraps any TorchSim model in batches, and
   ``AseTeacher`` wraps any ASE calculator, one structure at a time.

Walkthrough: A Water Student From MACE-MP
-----------------------------------------

This uses ``examples/simple_water``, which has a teacher configuration beside its ORCA one:
``master_config_teacher.json`` and ``qm_config_teacher_mace.json``. Run each step from that
directory. The outputs shown came from running these steps on a CPU.

1. Get the checkpoint
~~~~~~~~~~~~~~~~~~~~~

On a machine with network access and ``mace-torch`` installed, let MACE download its small MP model:

.. code-block:: bash

   python -c "from mace.calculators import mace_mp; mace_mp(model='small')"

MACE saves it under ``~/.cache/mace/``. Copy that ``.model`` file to
``teachers/mace-mp-0-small.model`` in the example directory, where ``qm_config_teacher_mace.json``
looks for it.

**For UMA instead,** the weights are gated on Hugging Face, so the download needs your account. Only
the download does: compute nodes read the local file and never need a token.

1. Sign in at `huggingface.co/facebook/UMA <https://huggingface.co/facebook/UMA>`__ and accept the
   license. Access may need approval.
2. Create a read token at `huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`__.
   A fine-grained token needs "Read access to contents of all public gated repos you can access".
3. On a machine with network access, such as a login node, with ``fairchem-core`` installed, which
   provides the ``hf`` command:

   .. code-block:: bash

      hf auth login      # Paste the token. It is saved to ~/.cache/huggingface/token.
      hf auth whoami     # Check which account is logged in.
      hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir teachers

   The checkpoint lands at ``teachers/checkpoints/uma-s-1p2p1.pt``, where
   ``qm_config_teacher_uma.json`` looks for it. It is about 2.3 GB.

Use ``qm_config_teacher_uma.json`` as ``QM_config_path`` in ``master_config_teacher.json`` to label
with UMA. Its ``task`` is ``omol``, for molecules and clusters, and it reads each structure's
``charge`` and ``mult`` metadata. ``omol`` would refuse the structures if they were periodic, but
the example builds clusters. Keep a UMA run's shards apart from a MACE run's, since ALF refuses to mix labels
from two teachers.

2. Check the teacher's labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build one structure and label it, without starting the loop:

.. code-block:: bash

   python -m alframework --master master_config_teacher.json --test_qm

ALF prints the structure, the teacher's results, and the same values read back from ``qm_test.h5``.
For a cluster of 12 water molecules, abridged:

.. code-block:: text

   QM labeling: native batches of up to 32 structures (timeout, 30 s)
   Job Failed: 0
   Atomic Numbers: [8 1 1 8 1 1 8 1 1 ...]
   PBCs: [False False False]
   energy: -146.4834530769415
   ...
   Data written to qm_test.h5:
   Species: H H H H H H H H H H H H H H H H H H H H H H H H O O O O O O O O O O O O
   energy:
   [-146.4835]

The stored species are sorted by atomic number, and the energy is in eV, unchanged by the factor of
1.0 in ``properties_list``. The structure isn't periodic, because the example's builder sets
``"periodic": false``: ORCA and the student both treat each structure as a cluster, so the teacher
must too.

3. See what was recorded
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alframework.tools.tools import read_label_provenance
   read_label_provenance("qm_test.h5")

.. code-block:: text

   [{'adapter': 'TorchSimTeacher', 'teacher': 'mace-mp-0-small',
     'checkpoint': '.../teachers/mace-mp-0-small.model',
     'checkpoint_sha256': '2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736',
     'package': 'mace-torch', 'package_version': '0.3.17', 'dtype': 'float64', 'task': None,
     'units': {'energy': 'eV', 'forces': 'eV/Å'}, ...}]

Every shard of the run carries the same record.

4. Run the loop
~~~~~~~~~~~~~~~

Set the machine settings in ``parsl_configs.py``. The teacher runs on ``alf_QM_executor``, so give
that executor GPUs if you have them. Then start ALF:

.. code-block:: bash

   python -m alframework --master master_config_teacher.json

The run writes to its own paths, so it can share the directory with the ORCA configuration:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Contents
   * - ``h5store_teacher/data-*.h5``
     - Teacher labels, each shard with its ``label_provenance``.
   * - ``h5store_teacher_held/``
     - Labels waiting for the next store; see :ref:`configuration-restarts`.
   * - ``models_teacher/model-*/``
     - The student ensembles, each with ``label_sources.json``.
   * - ``status_teacher.txt``
     - Progress and restart state.

5. Check the student against QM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A student can only be as good as its teacher, and its training metrics compare it with the teacher.
To measure it against QM, keep a few QM-labeled shards in their own directory, for example from a
short run of the ORCA configuration, ``master_config.json``, or of the PySCF one,
``master_config_pyscf.json``, which needs no separate QM program. They must be in eV and eV/Å, which both
configurations' factors give. Then add to ``hippynn_config.json``:

.. code-block:: json

   "h5_test_dir": "h5store_qm_test"

Each member then trains on all the teacher labels except its validation split, and its test metrics
in ``training_log.txt`` compare it with those QM labels. A relative path is taken from the master
directory. Keep this directory apart from ``h5_dir``, which the teacher's mixed-label check guards.

Next Steps
----------

* If the student matches the teacher but the teacher doesn't match QM, a student won't fix that.
  Fine-tune MACE or UMA on QM labels that ALF selects instead, and use the fine-tuned model
  directly; see :doc:`fine_tuning`.
* :doc:`batching` covers batch sizes for the teacher and for sampling.
* :doc:`units` lists the units each part of ALF uses.
