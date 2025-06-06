"""
Module to run partial volume correction on a parametric PET image using the symmetric geometric
transfer matrix (sGTM) method.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import ants

from ..utils.useful_functions import check_physical_space_for_ants_image_pair
from ..utils.scan_timing import ScanTimingInfo
from ..utils.time_activity_curve import TimeActivityCurve
from ..utils.bids_utils import gen_bids_like_filename, parse_path_to_get_subject_and_session_id
from ..preproc.segmentation_tools import unique_segmentation_labels

class Sgtm:
    """
    Handle sGTM partial volume correction on parametric images.
    """
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 fwhm: float | tuple[float, float, float],
                 zeroth_roi: bool = False):
        """
        Initialize running sGTM

        Args:
            input_image_path (str): Path to input parametric image on which sGTM will be run.
            segmentation_image_path (str): Path to segmentation image to which parametric image is
                aligned which is used to deliniate regions for PVC.
            fwhm (float | tuple[float, float, float]): Full width at half maximum of the Gaussian 
                blurring kernel for each dimension.
            zeroth_roi (bool): If False, ignores the zero label in calculations, often used to 
                exclude background or non-ROI regions.
        """
        self.input_image = ants.image_read(input_image_path)
        self.segmentation_image = ants.image_read(segmentation_image_path)
        self.fwhm = fwhm
        self.zeroth_roi = zeroth_roi
        self.sgtm_result = self.run_sgtm(input_image=self.input_image,
                                         segmentation_image=self.segmentation_image)



    @property
    def sigma(self):
        """
        Blurring kernal sigma for sGTM based on the input FWHM.
        """
        resolution = self.segmentation_image.spacing
        if isinstance(self.fwhm, float):
            sigma = [(self.fwhm / 2.355) / res for res in resolution]
        else:
            sigma = [(fwhm_i / 2.355) / res_i for fwhm_i, res_i in zip(self.fwhm, resolution)]
        return sigma


    @property
    def unique_labels(self):
        """
        Get unique ROIs for sGTM.
        """
        return unique_segmentation_labels(segmentation_img=self.segmentation_image,
                                          zeroth_roi=self.zeroth_roi)


    @staticmethod
    def get_omega_matrix(voxel_by_roi_matrix: np.ndarray) -> np.ndarray:
        """
        Get the Omega matrix for sGTM. See :meth:`run_sgtm` for details.
        """
        omega = voxel_by_roi_matrix.T @ voxel_by_roi_matrix
        return omega


    @staticmethod
    def solve_sgtm(omega: np.ndarray,
                   voxel_by_roi_matrix: np.ndarray,
                   input_numpy: np.ndarray) -> tuple:
        """
        Set up and solve linear equation for sGTM.
        """
        t_vector = voxel_by_roi_matrix.T @ input_numpy.ravel()
        t_corrected = np.linalg.solve(omega, t_vector)
        condition_number = np.linalg.cond(omega)

        return t_corrected, condition_number


    @staticmethod
    def get_voxel_by_roi_matrix(unique_labels, segmentation_numpy, sigma):
        """
        Get the ``V`` matrix for sGTM.
        """
        voxel_by_roi_matrix = np.zeros((segmentation_numpy.size, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            masked_roi = (segmentation_numpy == label).astype('float32')
            blurred_roi = gaussian_filter(masked_roi, sigma=sigma)
            voxel_by_roi_matrix[:, i] = blurred_roi.ravel()

        return voxel_by_roi_matrix


    def run_sgtm(self,
                 input_image: ants.ANTsImage,
                 segmentation_image: ants.ANTsImage) -> tuple[np.ndarray, np.ndarray, float]:
        r"""
        Apply Symmetric Geometric Transfer Matrix (SGTM) method for Partial Volume Correction 
        (PVC) to PET images based on ROI labels.

        This method involves using a matrix-based approach to adjust the PET signal intensities for
        the effects of partial volume averaging.

        Args:
            input_image (nib.Nifti1Image): The 3D PET image Nifti1 object.
            segmentation_image (nib.Nifti1Image): The 3D ROI image, Nifti1 object, must have the
                same dimensions as `input_image`.
            fwhm (float | tuple[float, float, float]): Full width at half maximum of the Gaussian 
                blurring kernel for each dimension.
            zeroth_roi (bool): If False, ignores the zero label in calculations, often used to 
                exclude background or non-ROI regions.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - np.ndarray: Array of unique ROI labels.
                - np.ndarray: Corrected PET values after applying PVC.
                - float: Condition number of the omega matrix, indicating the numerical stability
                    of the inversion.

        Raises:
            AssertionError: If `input_image` and `segmentation_image` do not have the same
                dimensions.

        Examples:
            .. code-block:: python

                input_image = nib.load('path_to_pet_image.nii')
                segmentation_image = nib.load('path_to_roi_image.nii')
                fwhm = (8.0, 8.0, 8.0)  # or fwhm = 8.0
                labels, corrected_values, cond_number = sgtm(input_image, segmentation_image, fwhm)

        Notes:
            The SGTM method uses the matrix :math:`\Omega` (omega), defined as:

            .. math::
            
                \Omega = V^T V

            where :math:`V` is the matrix obtained by applying Gaussian filtering to each ROI,
            converting each ROI into a vector. The element :math:`\Omega_{ij}` of the matrix
            :math:`\Omega` is the dot product of vectors corresponding to the i-th and j-th ROIs,
            representing the spatial overlap between these ROIs after blurring.

            The vector :math:`t` is calculated as:

            .. math::
            
                t = V^T p

            where :math:`p` is the vectorized PET image. The corrected values,
            :math:`t_{corrected}`, are then obtained by solving the linear system:

            .. math::
            
                \Omega t_{corrected} = t

            This provides the estimated activity concentrations corrected for partial volume
            effects in each ROI.
        """
        if input_image.shape != segmentation_image.shape:
            raise AssertionError("PET and ROI images must be the same dimensions")
        input_numpy = input_image.numpy()
        segmentation_numpy = segmentation_image.numpy()

        unique_labels = self.unique_labels

        voxel_by_roi_matrix = Sgtm.get_voxel_by_roi_matrix(unique_labels=unique_labels,
                                                           segmentation_numpy=segmentation_numpy,
                                                           sigma=self.sigma)
        omega = Sgtm.get_omega_matrix(voxel_by_roi_matrix=voxel_by_roi_matrix)
        t_corrected, condition_number = Sgtm.solve_sgtm(omega=omega,
                                                        voxel_by_roi_matrix=voxel_by_roi_matrix,
                                                        input_numpy=input_numpy)

        return unique_labels, t_corrected, condition_number


    def run_sgtm_4d(self,
                    input_image: ants.core.ANTsImage,
                    segmentation_image: ants.core.ANTsImage) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run sgtm on 4d
        """
        if not check_physical_space_for_ants_image_pair(input_image,
                                                        segmentation_image):
            raise AssertionError("PET and ROI images must be the same dimensions")
        pet_frame_list = input_image.ndimage_to_list()
        segmentation_numpy = segmentation_image.numpy()

        unique_labels = self.unique_labels

        voxel_by_roi_matrix = Sgtm.get_voxel_by_roi_matrix(unique_labels=unique_labels,
                                                           segmentation_numpy=segmentation_numpy,
                                                           sigma=self.sigma)
        omega = Sgtm.get_omega_matrix(voxel_by_roi_matrix=voxel_by_roi_matrix)

        frame_results = []
        for frame in pet_frame_list:
            input_numpy = frame.numpy()
            t_corrected, _cond_num = Sgtm.solve_sgtm(omega=omega,
                                                     voxel_by_roi_matrix=voxel_by_roi_matrix,
                                                     input_numpy=input_numpy)
            frame_results += [t_corrected]

        return frame_results


    def save_results(self, out_tsv_path: str):
        """
        Saves the result of an sGTM calculation.
        """
        sgtm_result_array = np.array([self.sgtm_result[0],self.sgtm_result[1]]).T
        np.savetxt(out_tsv_path,sgtm_result_array,header='Region\tMean',fmt=['%.0f','%.2f'])


    def save_results_by_region(self,
                               input_image_path: str,
                               sgtm_result: np.ndarray,
                               out_tac_dir: str):
        """
        Saves the result of an sGTM calculation.
        """
        frame_timing = ScanTimingInfo.from_nifti(image_path=input_image_path)
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(path=input_image_path)


        tac_array = np.array([sgtm_result[i] for i in range(len(sgtm_result))]).T

        for label, i in enumerate(self.unique_labels):
            pvc_tac = TimeActivityCurve(times=frame_timing.center_in_mins,
                                        activity=tac_array[i,:])
            tac_filename = gen_bids_like_filename(sub_id=sub_id,
                                                  ses_id=ses_id,
                                                  suffix='tac',
                                                  seg=int(label),
                                                  ext='.tsv')
            out_tac_path = os.path.join(out_tac_dir, tac_filename)
            pvc_tac.to_tsv(filename=out_tac_path)
