import lmfit
import numpy as np
import ants
from lmfit.minimizer import MinimizerResult

from ..preproc.image_operations_4d import extract_roi_voxel_tacs_from_image_using_mask as extract_masked_voxels
from ..utils.image_io import get_frame_timing_info_for_nifti
from ..utils.data_driven_image_analyses import temporal_pca_analysis_of_image_over_mask as temporal_pca_over_mask

_MAX_SCAN_IN_MINS_ = 200.0

class PCAGuidedIdif(object):
    def __init__(self,
                 input_image_path: str,
                 mask_image_path: str,
                 output_tac_path: str,
                 num_pca_components: int,
                 alpha: float,
                 beta: float,
                 verbose: bool = False):
        self.image_path = input_image_path
        self.mask_path = mask_image_path
        self.output_tac_path = output_tac_path
        self.num_components = num_pca_components
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta

        self.mask_voxel_tacs = extract_masked_voxels(input_image=ants.image_read(self.image_path),
                                                     mask_image=ants.image_read(self.mask_path),
                                                     verbose=self.verbose)

        self.mask_avg = np.mean(self.mask_voxel_tacs, axis=0)
        self.mask_std = np.mean(self.mask_voxel_tacs, axis=0)

        self.mask_peak_arg = np.argmax(self.mask_avg)
        self.mask_peak_val = self.mask_avg[self.mask_peak_arg]

        self.tac_times_in_mins = self.get_frame_reference_times(image_path=self.image_path)

        self.pca_obj, self.pca_fit = temporal_pca_over_mask(input_image=ants.image_read(self.image_path),
                                                            mask_image=ants.image_read(self.mask_path),
                                                            num_components=self.num_components)

        self.pca_filter_flags = self.get_pca_component_filter_flags(self.pca_obj.components_)
        self.filter_signs = self.get_pca_filter_signs_from_flags(self.pca_filter_flags)

        self._fitting_params = self._generate_quantile_params(num_components=self.num_components)

        self.fitting_obj = lmfit.Minimizer(userfcn=self.residual,
                                           params=self._fitting_params,
                                           fcn_args=(self.pca_fit, self.pca_filter_flags, self.mask_voxel_tacs,
                                                     self.alpha, self.beta))

        self.fit_result: MinimizerResult | None = None
        self.result_params: lmfit.Parameters | None = None
        self.fit_quantiles: np.ndarray | None = None
        self.fit_voxel_mask: np.ndarray | None = None
        self.fit_mask_voxel_tacs: np.ndarray | None = None

        self.idif_vals : np.ndarray | None = None
        self.idif_errs : np.ndarray | None = None


    def run(self, method: str = 'ampgo', **method_kwargs):
        self.fit_result = self.fitting_obj.minimize(method=method, **method_kwargs)
        self.result_params = self.fit_result.params
        self.fit_quantiles = np.asarray(list(self.fit_result.params.valuesdict().values()))
        self.fit_voxel_mask = self.calculate_voxel_mask_from_quantiles(params=self.fit_result.params,
                                                                       pca_values_per_voxel=self.pca_fit,
                                                                       quantile_flags=self.pca_filter_flags,)
        self.fit_mask_voxel_tacs = self.mask_voxel_tacs[self.fit_voxel_mask]

        self.idif_vals = np.mean(self.fit_mask_voxel_tacs, axis=0)
        self.idif_errs = np.std(self.fit_mask_voxel_tacs, axis=0)

    @staticmethod
    def get_pca_component_filter_flags(pca_components: np.ndarray[float],
                                       comp_min_val: float = 0.0,
                                       threshold: float = 0.1) -> np.ndarray[bool]:
        pca_components_positive_pts = np.mean(pca_components > comp_min_val, axis=1)
        pca_components_filter_flags = ~(pca_components_positive_pts > threshold)
        return pca_components_filter_flags

    @staticmethod
    def get_pca_filter_signs_from_flags(pca_component_filter_flags: np.ndarray[bool]) -> list[str]:
        return ['>' if sgn else '<' for sgn in ~pca_component_filter_flags]

    @staticmethod
    def get_frame_reference_times(image_path: str) -> np.ndarray[float]:
        image_timing_info_dict = get_frame_timing_info_for_nifti(image_path=image_path)
        return image_timing_info_dict.center_in_mins

    @staticmethod
    def _generate_quantile_params(num_components: int = 3,
                                  value: float = 0.5,
                                  lower: float = 1e-4,
                                  upper: float = 0.999) -> lmfit.Parameters:
        tmp_dict = {'value': value, 'min': lower, 'max': upper}
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
    def _voxel_term_func(voxel_nums: float) -> float:
        return np.log(1.0 + np.exp(-voxel_nums / 6.0))

    @staticmethod
    def _noise_term_func(tac_stderrs: np.ndarray[float]) -> float:
        return np.sqrt(np.mean(tac_stderrs ** 2))

    @staticmethod
    def _smoothness_term_func(tac_values: np.ndarray[float]) -> float:
        return np.sum(np.abs(np.diff(tac_values, prepend=tac_values[0]) / np.max(tac_values)))

    @staticmethod
    def _peak_term_func(tac_peak_ratio):
        return np.log(1.0 + np.exp(-tac_peak_ratio * 1.5))

    def residual(self,
                 params: lmfit.Parameters,
                 pca_values_per_voxel: np.ndarray[float],
                 quantile_flags: np.ndarray[bool],
                 voxel_tacs: np.ndarray,
                 alpha: float,
                 beta: float) -> float:
        voxel_mask = self.calculate_voxel_mask_from_quantiles(params, pca_values_per_voxel, quantile_flags)
        valid_voxels_number = np.sum(voxel_mask)
        masked_voxels = voxel_tacs[voxel_mask]

        tacs_avg = np.mean(masked_voxels, axis=0) if valid_voxels_number > 1 else self.mask_avg
        tacs_std = np.std(masked_voxels, axis=0) if valid_voxels_number > 1 else self.mask_std

        voxel_term = self._voxel_term_func(voxel_nums=valid_voxels_number)
        noise_term = self._noise_term_func(tac_stderrs=tacs_std)

        peak_ratio = tacs_avg[self.mask_peak_arg] / self.mask_peak_val
        peak_term = alpha * self._peak_term_func(tac_peak_ratio=peak_ratio) if alpha != 0.0 else 0.0
        smth_term = beta * self._smoothness_term_func(tac_values=tacs_avg) if beta != 0.0 else 0.0

        return voxel_term + noise_term + peak_term + smth_term
