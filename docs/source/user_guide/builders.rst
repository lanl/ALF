Builders
========

Builders construct initial structures and simulation boxes used in active learning.

Available builder modules
-------------------------

ALF currently includes three builder modules with different use cases:

1. ``alframework.builders.builders``
   General-purpose builders for loading structures and creating condensed-phase
   systems. This module includes task wrappers and helper logic for common
   ALF workflows.
2. ``alframework.builders.moltensalt_builder``
   Specialized utilities for charge-balanced atomic-system generation and box
   construction used in molten-salt-style simulations.
3. ``alframework.builders.reactive_builder``
   Reactive-system builder utilities for loading reactant / transition-state /
   product structures and preparing them for sampling.

Choosing a builder
------------------

Use the general ``builders`` module for most workflows that start from a
molecule library and standard condensed-phase system setup.

Use ``moltensalt_builder`` when you need random neutral ionic systems with
density and minimum-distance constraints.

Use ``reactive_builder`` when your workflow is centered on known reactive
pathways and reaction-structure sets.

Builder API links
-----------------

You can link from this guide directly to API pages:

* :doc:`Builders package API <../api_documentation/alframework.builders>`
* :doc:`Core builders module <../api_documentation/alframework.builders.builders>`
* :doc:`Molten-salt builder module <../api_documentation/alframework.builders.moltensalt_builder>`
* :doc:`Reactive builder module <../api_documentation/alframework.builders.reactive_builder>`
