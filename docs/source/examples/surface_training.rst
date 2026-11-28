Train on CO/Cu(111) with UMA or HIPPYNN
=======================================

The files in ``examples/surface_training`` show two active learning workflows
for pristine Cu(111) slabs and CO/Cu(111) adslabs. Both use the ALF surface builder
and VASP RPBE energies and forces. The example starts from perturbed clean
slabs and from adslabs with CO at atop, bridge, and hollow sites, then keeps the
periodic cell fixed during MD.

Use ``master_uma.json`` to fine-tune the ``oc20`` task of a local UMA-S
checkpoint and sample through TorchSim. Use ``master_hippynn.json`` to train
a periodic HIPPYNN ensemble from scratch. The two master configurations keep
separate output directories. Copy a fixed snapshot of finalized HDF5 shards
to a fresh run directory to train the two models on the same reference data.

Start with ``--check_config``, ``--test_builder``, and
``--test_qm`` on the selected master. VASP and the appropriate PAW datasets
are required for labeling. The UMA run also needs the local checkpoint and a
GPU worker with Fairchem and TorchSim. See
``examples/surface_training/README.md`` for setup, training and sampling
checks, and guidance on MACE and r²SCAN comparisons.

See :doc:`../user_guide/builders` for surface settings and
:doc:`../user_guide/fine_tuning_uma` for UMA settings.

Chalcogenide adsorption with UMA OC25
-------------------------------------

Use ``master_chalcogenide_uma.json`` to run the built-in surface builder on
conventional zinc-blende bulk cells. ``builder_chalcogenide.json`` cuts (110)
slabs of CdSe, CdTe, and HgTe, then places methylamine or methylphosphine from
SMILES. HgSe has a separate builder config for evaluation. Each decorated
slab has 151 atoms, with a substrate thickness of 17.1–18.3 Å in a 48 Å
cell. The same master runs VASP labeling, iterative UMA OC25 fine-tuning,
and batched TorchSim dynamics with a fixed cell and all atoms mobile.
The README covers the inputs and benchmark comparisons.
