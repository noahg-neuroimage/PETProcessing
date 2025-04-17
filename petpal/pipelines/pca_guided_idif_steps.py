from ..input_function import pca_guided_idif
from ..pipelines.preproc_steps import ImageToImageStep
from ..pipelines.steps_base import StepsAPI, ObjectBasedStep
from ..utils.bids_utils import gen_bids_like_filepath, parse_path_to_get_subject_and_session_id, snake_to_camel_case


class PCAGuidedIDIFMixin(StepsAPI):
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 verbose: bool
                 ):
        StepsAPI.__init__(self)
        self.init_kwargs = {'input_image_path': input_image_path,
                            'mask_image_path': mask_image_path,
                            'output_tac_path': output_tac_path,
                            'num_pca_components': num_pca_components,
                            'verbose': verbose
                            }
        self.call_kwargs = {}
        self.input_image_path = input_image_path
        self.mask_image_path = mask_image_path
        self.output_tac_path = output_tac_path
        self.num_pca_components = num_pca_components
        self.verbose = verbose
        self.name = 'pca_guided_idif_base'


    @property
    def input_image_path(self):
        return self.init_kwargs['input_image_path']

    @input_image_path.setter
    def input_image_path(self, value):
        self.init_kwargs['input_image_path'] = value

    @property
    def mask_image_path(self):
        return self.init_kwargs['mask_image_path']

    @mask_image_path.setter
    def mask_image_path(self, value):
        self.init_kwargs['mask_image_path'] = value

    @property
    def output_tac_path(self) -> str:
        return self.init_kwargs['output_tac_path']

    @output_tac_path.setter
    def output_tac_path(self, value):
        self.init_kwargs['output_tac_path'] = value

    @property
    def num_pca_components(self):
        return self.init_kwargs['num_pca_components']

    @num_pca_components.setter
    def num_pca_components(self, value):
        self.init_kwargs['num_pca_components'] = value

    @property
    def verbose(self):
        return self.init_kwargs['verbose']

    @verbose.setter
    def verbose(self, value):
        self.init_kwargs['verbose'] = value

    def set_input_as_output_from(self, *sending_steps: ImageToImageStep) -> None:
        """
        Sets the input image paths based on the output paths from other steps in the pipeline.
        The first sending step will set the input image path, and the second sending step will
        set the second image path.

        Args:
            sending_steps (tuple[ImageToImageStep]): Two pipeline steps whose outputs will be used
                as the input image path and second image input path.

        Raises:
            AssertionError: If the number of provided sending steps is not exactly two.
        """
        assert len(sending_steps) == 2, "ImagePairToArrayStep must have 2 sending ImageToImageStep steps."
        if isinstance(sending_steps[0], ImageToImageStep):
            self.input_image_path = sending_steps[0].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[0])
        if isinstance(sending_steps[1], ImageToImageStep):
            self.mask_image_path = sending_steps[1].output_image_path
        else:
            super().set_input_as_output_from(sending_steps[1])

    def infer_outputs_from_inputs(self,
                                  out_dir: str,
                                  der_type: str = 'tacs',
                                  suffix: str = 'tac',
                                  ext: str = '.tsv',
                                  **extra_desc):
        """
        Infers the output array path based on the inputs and specified parameters.

        This method generates a BIDS-like derivatives filepath for the output based on the subject and
        session IDs extracted from the input image path.

        Args:
            out_dir (str): Directory where the output array will be saved.
            der_type (str, optional): Type of derivative. Will set the sub-directory in `out_dir`. Defaults to 'tacs'.
            suffix (str, optional): Suffix for the output filename. Defaults to 'tac'.
            ext (str, optional): File extension for the output file. Defaults to '.tsv'.
            **extra_desc: Additional descriptive parameters for the output filename.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, suffix=suffix, bids_dir=out_dir,
                                          modality=der_type, ext=ext, desc=step_name_in_camel_case, **extra_desc)
        self.output_tac_path = filepath


class PCAGuidedFitIDIFStep(ObjectBasedStep, PCAGuidedIDIFMixin):
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_array_path: str,
                 num_pca_components: int,
                 verbose: bool,
                 alpha: float,
                 beta: float,
                 method: str,
                 **meth_kwargs):
        PCAGuidedIDIFMixin.__init__(self,
                                    input_image_path=input_image_path,
                                    mask_image_path=mask_image_path,
                                    output_tac_path=output_array_path,
                                    num_pca_components=num_pca_components,
                                    verbose=verbose)
        ObjectBasedStep.__init__(self,
                                 name='pca_guided_fit_idif',
                                 class_type=pca_guided_idif.PCAGuidedIdifFitter,
                                 init_kwargs={**self.init_kwargs},
                                 call_kwargs=(self.call_kwargs | {'alpha' : alpha,
                                                                  'beta'  : beta,
                                                                  'method': method,
                                                                  **meth_kwargs}), )

        self.alpha = alpha
        self.beta = beta
        self.method = method

    @property
    def alpha(self):
        return self.call_kwargs['alpha']

    @alpha.setter
    def alpha(self, value):
        self.call_kwargs['alpha'] = value

    @property
    def beta(self):
        return self.call_kwargs['beta']

    @beta.setter
    def beta(self, value):
        self.call_kwargs['beta'] = value

    @property
    def method(self):
        return self.call_kwargs['method']

    @method.setter
    def method(self, value):
        self.call_kwargs['method'] = value


class PCAGuidedTopVoxelsIDIFStep(ObjectBasedStep, PCAGuidedIDIFMixin):
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_array_path: str,
                 num_pca_components: int,
                 verbose: bool,
                 selected_component: int,
                 num_of_voxels: int):
        PCAGuidedIDIFMixin.__init__(self,
                                    input_image_path=input_image_path,
                                    mask_image_path=mask_image_path,
                                    output_tac_path=output_array_path,
                                    num_pca_components=num_pca_components,
                                    verbose=verbose)
        ObjectBasedStep.__init__(self,
                                 name='pca_guided_top_voxels_idif',
                                 class_type=pca_guided_idif.PCAGuidedTopVoxelsIDIF,
                                 init_kwargs={'input_image_path'  : input_image_path,
                                              'mask_image_path'   : mask_image_path,
                                              'output_tac_path'   : output_array_path,
                                              'num_pca_components': num_pca_components,
                                              'verbose'           : verbose
                                              },
                                 call_kwargs=(self.call_kwargs | {'selected_component': selected_component,
                                                                  'num_of_voxels'     : num_of_voxels})
                                 )

        self.selected_component = selected_component
        self.num_of_voxels = num_of_voxels

    @property
    def selected_component(self):
        return self.call_kwargs['selected_component']

    @selected_component.setter
    def selected_component(self, value):
        self.call_kwargs['selected_component'] = value

    @property
    def num_of_voxels(self):
        return self.call_kwargs['num_of_voxels']

    @num_of_voxels.setter
    def num_of_voxels(self, value):
        self.call_kwargs['num_of_voxels'] = value
