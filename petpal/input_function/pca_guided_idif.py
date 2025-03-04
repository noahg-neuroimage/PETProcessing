from typing import Callable
import lmfit
import numpy as np
import ants

from ..preproc.image_operations_4d import extract_roi_voxel_tacs_from_image_using_mask
from ..utils.image_io import get_frame_timing_info_for_nifti

_MAX_SCAN_IN_MINS_ = 200.0

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

        self.mask_voxel_tacs = extract_roi_voxel_tacs_from_image_using_mask(input_image=ants.image_read(self.image_path),
                                                                            mask_image=ants.image_read(self.mask_path),
                                                                            verbose=self.verbose)

        self.mask_avg = np.mean(self.mask_voxel_tacs, axis=0)
        self.mask_std = np.mean(self.mask_voxel_tacs, axis=0)

    @staticmethod
    def get_frame_reference_times(image_path: str) -> np.ndarray[float]:
        image_timing_info_dict = get_frame_timing_info_for_nifti(image_path=image_path)
        frame_reference_times = image_timing_info_dict['center']
        if frame_reference_times[-1] >=_MAX_SCAN_IN_MINS_:
            return frame_reference_times / 60.0
        else:
            return frame_reference_times

    @staticmethod
    def _generate_quantile_params(num_components: int = 3,
                                  value: float = 0.5,
                                  lower: float = 1e-4,
                                  upper: float = 0.999) -> lmfit.Parameters:
        tmp_dict = {'value': value, 'lower': lower, 'upper': upper}
        return lmfit.create_params(**{f'pc{i}': tmp_dict for i in range(num_components)})

    @staticmethod
    def calculate_voxel_mask_from_quantiles(params: lmfit.Parameters,
                                            pca_values_per_voxel: np.ndarray[float],
                                            quantile_flags: np.ndarray[bool]) -> np.ndarray[bool]:
        voxel_mask = np.ones(len(pca_values_per_voxel), dtype=bool)
        quantile_values = params.valuesdict().values()
        for pca_component, quantile, flag in zip(pca_values_per_voxel.T, quantile_values, quantile_flags):
            voxel_mask *= (pca_component > np.quantile(pca_component, quantile)) ^ flag
        return voxel_mask

    @staticmethod
    def _objective_function_voxel_term(voxel_nums: float) -> float:
        return np.log(1.0 + np.exp(-voxel_nums / 6.0))

    def residual(self,
                 params: lmfit.Parameters,
                 pca_values_per_voxel: np.ndarray[float],
                 voxel_tacs: np.ndarray,
                 quantile_flags: np.ndarray[bool],
                 alpha: float,
                 beta: float,
                 mask_function: Callable) -> float:
        voxel_mask = mask_function(params, pca_values_per_voxel, quantile_flags)
        valid_voxel_number = np.sum(voxel_mask)
        masked_voxels = voxel_tacs[voxel_mask]


        tacs_avg = np.mean(masked_voxels, axis=0) if valid_voxel_number > 1 else self.mask_avg
        tacs_std = np.std(masked_voxels, axis=0) if valid_voxel_number > 1 else self.mask_std
