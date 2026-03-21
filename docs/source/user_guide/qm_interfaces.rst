QM Interfaces
=============

QM interfaces connect ALF to electronic structure engines.

Available QM interface modules
------------------------------

ALF currently includes several QM interface modules:

1. ``alframework.qm_interfaces.ase_calculator_interface``
   Generic QM task wrappers for use with ASE calculators.
2. ``alframework.qm_interfaces.orca5_interface``
   Interface for ORCA, including task and parsing utilities for single-point calculations and
   property extraction.
3. ``alframework.qm_interfaces.vaspase_interface``
   Interface for the Vienna ab initio Simulation Package (VASP) plane-wave electronic structure code.

Choosing a QM interface
-----------------------

Use ``ase_calculator_interface`` when your QM engine is exposed through an ASE
calculator and you want a general integration path.

Use ``orca5_interface`` when your workflow is built around ORCA input blocks
and ORCA-specific output parsing.

Use ``vaspase_interface`` when your workflow requires VASP-specific controls
and file handling beyond the generic ASE path.

QM interface API links
----------------------

You can link from this guide directly to API pages:

* :doc:`QM interfaces package API <../api_documentation/alframework.qm_interfaces>`
* :doc:`ASE calculator interface module <../api_documentation/alframework.qm_interfaces.ase_calculator_interface>`
* :doc:`ORCA interface module <../api_documentation/alframework.qm_interfaces.orca5_interface>`
* :doc:`VASP interface module <../api_documentation/alframework.qm_interfaces.vaspase_interface>`
