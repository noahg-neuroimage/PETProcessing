"""
Module to run partial volume correction on a parametric PET image using the symmetric geometric
transfer matrix (sGTM) method.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import ants


class Sgtm:
    """
    Handle sGTM partial volume correction on parametric images.
    """
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 fwhm: float | tuple[float, float, float],
                 zeroth_roi: bool = False,
                 out_tsv_path: str = None):
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
        self.out_tsv_path = out_tsv_path
        self.sgtm_result = self.run_sgtm(input_image=self.input_image,
                                         segmentation_image=self.segmentation_image,
                                         fwhm=self.fwhm,
                                         zeroth_roi=self.zeroth_roi)
        if self.out_tsv_path:
            self.save_results()

    @staticmethod
    def run_sgtm(input_image: ants.ANTsImage,
                segmentation_image: ants.ANTsImage,
                fwhm: float | tuple[float, float, float],
                zeroth_roi: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
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
                - float: Condition number of the omega matrix, indicating the numerical stability of the inversion.

        Raises:
            AssertionError: If `input_image` and `segmentation_image` do not have the same dimensions.

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

            where :math:`V` is the matrix obtained by applying Gaussian filtering to each ROI, converting each ROI into a
            vector. The element :math:`\Omega_{ij}` of the matrix :math:`\Omega` is the dot product of vectors
            corresponding to the i-th and j-th ROIs, representing the spatial overlap between these ROIs after blurring.

            The vector :math:`t` is calculated as:

            .. math::
            
                t = V^T p

            where :math:`p` is the vectorized PET image. The corrected values, :math:`t_{corrected}`, are then obtained
            by solving the linear system:

            .. math::
            
                \Omega t_{corrected} = t

            This provides the estimated activity concentrations corrected for partial volume effects in each ROI.
        """
        assert input_image.shape == segmentation_image.shape, "PET and ROI images must be the same dimensions"
        input_numpy = input_image.numpy()
        segmentation_numpy = segmentation_image.numpy()
        resolution = input_image.spacing
        if isinstance(fwhm, float):
            sigma = [(fwhm / 2.355) / res for res in resolution]
        else:
            sigma = [(fwhm_i / 2.355) / res_i for fwhm_i, res_i in zip(fwhm, resolution)]

        unique_labels = np.unique(segmentation_numpy)
        if not zeroth_roi:
            unique_labels = unique_labels[unique_labels != 0]


        flattened_size = input_numpy.size
        voxel_by_roi_matrix = np.zeros((flattened_size, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            masked_roi = (segmentation_numpy == label).astype('float32')
            blurred_roi = gaussian_filter(masked_roi, sigma=sigma)
            voxel_by_roi_matrix[:, i] = blurred_roi.ravel()

        omega = voxel_by_roi_matrix.T @ voxel_by_roi_matrix

        t_vector = voxel_by_roi_matrix.T @ input_numpy.ravel()
        t_corrected = np.linalg.solve(omega, t_vector)
        condition_number = np.linalg.cond(omega)

        return unique_labels, t_corrected, condition_number

    def save_results(self):
        """
        Saves the result of an sGTM calculation.
        """
        sgtm_result_array = np.array([self.sgtm_result[0],self.sgtm_result[1]]).T
        np.savetxt(self.out_tsv_path,sgtm_result_array,header='Region\tMean',fmt=['%.0f','%.2f'])
