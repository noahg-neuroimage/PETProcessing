PiB Pipeline Tutorial
=====================

--------
Overview
--------

This tutorial details the required steps to construct an pipeline for a project using the PiB tracer. As it is unlikely
that all the details will fit each neuroimaging project perfectly, this tutorial will explain how to alter the steps to
accommodate some other project designs.

A broad overview of the steps is given below, followed by a more detailed discussion of each step.

#. Create an instance of the :class:`~petpal.pipelines.pipelines.BIDS_Pipeline` Class.
#. Create :class:`~petpal.pipelines.steps_containers.StepsContainer` objects for preprocessing and kinetic modeling.
#. Add steps to both containers.
#. Add containers to the pipeline.
#. Order the steps in the pipeline using its ``add_dependency()`` and ``update_dependency()`` methods.
#. (Recommended) Verify pipeline's construction using its ``plot_dependency_graph()`` method.
#. Run the pipeline.

--------------------
In-depth Walkthrough
--------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1. BIDS_Pipeline Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



