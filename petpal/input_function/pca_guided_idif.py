import lmfit
import numpy as np

class PCAGuidedIdif(object):
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 verbose: bool = False):
        self.image_path = input_image_path
        self.mask_path = mask_image_path
        self.output_tac_path = output_tac_path
        self.num_components = num_pca_components
        self.verbose = verbose

