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
#. Order the steps in the pipeline using its :meth:`~petpal.pipelines.steps_containers.StepsPipeline.add_dependency()` and :meth:`~petpal.pipelines.steps_containers.StepsPipeline.update_dependencies()` methods.
#. (Recommended) Verify pipeline's construction using its :meth:`~petpal.pipelines.steps_containers.StepsPipeline.plot_dependency_graph()` method.
#. Run the pipeline.

--------------------
In-depth Walkthrough
--------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1. BIDS_Pipeline Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is most compact and efficient to define all necessary paths for a pipeline when instantiating the instance.
An example of this process is shown below for a PiB pipeline which will use T1w (T1-weighted) MRI as the anatomical
image, `Link freesurfer <https://surfer.nmr.mgh.harvard.edu/>`_ outputs (aparc+aseg.nii.gz in this case) as the
segmentation image, and a segmentation table stored as 'dseg.tsv'.

.. code-block:: python

    import petpal

    sub_id = '001'
    ses_id = '001'
    seg_path = f'/example/path/to/Data/PiB_BIDS/derivatives/freesurfer/sub-{sub_id}/ses-{ses_id}/aparc+aseg.nii.gz'
    seg_table_path = f'/example/path/to/Data/PiB_BIDS/derivatives/freesurfer/dseg.tsv'
    anat_path = f'/example/path/to/Data/PiB_BIDS/sub-{sub_id}/ses-{ses_id}/anat/sub-{sub_id}_ses-{ses_id}_T1w.nii.gz'
    bids_dir = '/example/path/to/Data/PiB_BIDS'

    PiB_Pipeline = petpal.pipelines.pipelines.BIDS_Pipeline(sub_id=sub_id,
                                                            ses_id=ses_id,
                                                            pipeline_name='PiB_Pipeline',
                                                            raw_anat_img_path=anat_path,
                                                            segmentation_img_path=seg_path,
                                                            segmentation_label_table_path=seg_table_path,
                                                            bids_root_dir=bids_dir)



