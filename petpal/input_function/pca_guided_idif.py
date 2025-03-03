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

    @staticmethod
    def _generate_quantile_params(num_compnents: int = 3,
                                  value: float = 0.5,
                                  lower: float = 1e-4,
                                  upper: float = 0.999):
        tmp_dict = {'value': value, 'lower': lower, 'upper': upper}
        return lmfit.create_params(**{f'pc{i}' : tmp_dict for i in range(num_compnents)})
