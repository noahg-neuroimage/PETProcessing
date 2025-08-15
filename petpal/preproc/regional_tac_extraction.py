"""
Regional TAC extraction
"""
import re
import os
import pathlib
import numpy as np
import ants


from .image_operations_4d import extract_mean_roi_tac_from_nifti_using_segmentation
from .segmentation_tools import combine_regions_as_mask
from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..utils.useful_functions import check_physical_space_for_ants_image_pair
from ..utils.time_activity_curve import TimeActivityCurve


def extract_roi_voxel_tacs_from_image_using_mask(input_image: ants.core.ANTsImage,
                                                 mask_image: ants.core.ANTsImage,
                                                 verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an image using a mask image.

    This function returns all the voxel TACs, and unlike
    :func:`extract_mean_roi_tac_from_nifti_using_segmentation` does not calculate the mean over
    all the voxels.

    Args:
        input_image (ants.core.ANTsImage): Input 4D-image from which to extract ROI voxel tacs.
        mask_image (ants.core.ANTsImage): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Array of voxel TACs of shape (num_voxels, num_frames)

    Raises:
         AssertionError: If input image is not 4D-image.
         AssertionError: If mask image is not in the same physical space as the input image.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc import regional_tac_extraction
            tac_func = regional_tac_extraction.extract_roi_voxel_tacs_from_image_using_mask
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Run ROI extraction and save
            time_series = tac_func(input_image=pet_img, mask_image=masked_region_img).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')
            
    """
    assert len(input_image.shape) == 4, "Input image must be 4D."
    assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
        "Images must have the same physical dimensions.")

    out_voxels = apply_mask_4d(input_arr=input_image.numpy(),
                               mask_arr=mask_image.numpy(),
                               verbose=verbose)
    return out_voxels


def apply_mask_4d(input_arr: np.ndarray,
                  mask_arr: np.ndarray,
                  verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an array using a mask array.

    This function applies a 3D mask to a 4D image, returning the time series for each voxel in a
    single flattened numpy array.

    Args:
        input_arr (np.ndarray): Input 4D-image from which to extract ROI voxel tacs.
        mask_arr (np.ndarray): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel
            tacs.

    Returns:
        out_voxels (np.ndarray): Time series of each voxel in the mask, as a flattened numpy array.

    Raises:
         AssertionError: If input array is not 4D.
         AssertionError: If input and mask array shapes are mismatched.

    Example:

        .. code-block:: python

            import ants
            import numpy as np

            from petpal.preproc.regional_tac_extraction import apply_mask_4d
            
            # Read images
            pet_img = ants.image_read("/path/to/pet.nii.gz")
            masked_region_img = ants.image_read("/path/to/mask_region.nii.gz")

            # Get underlying arrays
            pet_arr = pet_img.numpy()
            masked_region_arr = masked_region_img.numpy()

            # Run ROI extraction and save
            time_series = apply_mask_4d(input_arr=pet_arr, mask_arr=masked_region_arr).T
            np.savetxt("time_series.tsv", time_series, delimiter='\t')

    """
    assert len(input_arr.shape) == 4, "Input array must be 4D."
    assert input_arr.shape[:3] == mask_arr.shape, (
            "Array must have the same physical dimensions.")

    x_inds, y_inds, z_inds = mask_arr.nonzero()
    out_voxels = input_arr[x_inds, y_inds, z_inds, :]
    if verbose:
        print(f"(ImageOps): Output TACs have shape {out_voxels.shape}")
    return out_voxels


def voxel_average_w_uncertainty(pet_voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatially average flattened PET voxels and get the standard deviation as well.
    
    Args:
        pet_voxels (np.ndarray): 1D or 2D array of PET voxels.
    
    Returns:
        average_w_uncertainty (tuple[np.ndarray, np.ndarray]): Average and standard deviation of
            PET voxels."""
    pet_average = pet_voxels.mean((0))
    pet_uncertainty = pet_voxels.std((0))
    return (pet_average, pet_uncertainty)


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               out_tac_prefix: str = '', ):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    pet_numpy = ants.image_read(input_image_path).numpy()
    seg_numpy = ants.image_read(segmentation_image_path).numpy()

    scan_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)
    tac_times_in_mins = scan_timing_info.center_in_mins

    for i, _maps in enumerate(label_map['mapping']):
        region_mask = combine_regions_as_mask(segmentation_img=seg_numpy,
                                              label=int(regions_map[i]))
        pet_masked_region = apply_mask_4d(input_arr=pet_numpy,
                                          mask_arr=region_mask)
        extracted_tac, tac_uncertainty = voxel_average_w_uncertainty(pet_masked_region)
        region_tac_file = TimeActivityCurve(times=tac_times_in_mins,
                                            activity=extracted_tac,
                                            uncertainty=tac_uncertainty)
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir,
                                        f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        region_tac_file.to_tsv(filename=out_tac_path)


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: list[int] | int,
            out_tac_path: str,
            time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for a single region, given a mask,
    4D PET image, and region mapping. Computes the average of the PET image 
    within each region. Writes a tsv table with region name, frame start time,
    and mean value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifti_with_same_filename(input_image_4d_path)
    pet_numpy = ants.image_read(input_image_4d_path).numpy()
    seg_numpy = ants.image_read(roi_image_path).numpy()

    region_mask = combine_regions_as_mask(segmentation_img=seg_numpy,
                                            label=region)
    pet_masked_region = apply_mask_4d(input_arr=pet_numpy,
                                        mask_arr=region_mask)
    extracted_tac, tac_uncertainty = voxel_average_w_uncertainty(pet_masked_region)
    region_tac_file = TimeActivityCurve(times=pet_meta[time_frame_keyword],
                                        activity=extracted_tac,
                                        uncertainty=tac_uncertainty)
    region_tac_file.to_tsv(filename=out_tac_path)


class WriteRegionalTacs:
    """
    Write regional TACs

    Attrs:
        pet_img
        seg_img
        tac_extraction_func
        out_tac_prefix
        out_tac_dir
        scan_timing
    """
    def __init__(self,
                 input_image_path: str | pathlib.Path,
                 segmentation_path: str | pathlib.Path,
                 out_tac_prefix: str,
                 out_tac_dir: str | pathlib.Path):
        self.pet_img = ants.image_read(filename=input_image_path)
        self.seg_img = ants.image_read(filename=segmentation_path)
        self.tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
        self.out_tac_prefix = out_tac_prefix
        self.out_tac_dir = out_tac_dir
        self.scan_timing = ScanTimingInfo.from_nifti(input_image_path)


    def set_tac_extraction_func(self, tac_extraction_func: callable):
        """Sets the tac extraction function used to a different function.
        
        The selected function must take an input image, label image, and a single label mapping as
        inputs, and return an estimation of activity and uncertainty of that estimation as outputs.
        """
        self.tac_extraction_func = tac_extraction_func


    @staticmethod
    def capitalize_first_char_of_str(input_str: str):
        """
        Capitalize only the first character of a string, leaving the remainder unchanged.
        Args:
            input_str (str): The string to capitalize the first character of.
        Returns:
            output_str (str): The string with only the first character capitalized.
        """
        output_str = input_str[0].capitalize()+input_str[1:]
        return output_str


    @staticmethod
    def str_to_camel_case(input_str):
        """
        Take a string and return the string converted to camel case.

        Special characters (? * - _ / \\) are removed and treated as word separaters. Different
        words are then capitalized at the first character, leaving other alphanumeric characters
        unchanged.
        """
        split_str = re.split(r'[-_?*/\\]', input_str)
        capped_split_str = []
        capitalize_first = WriteRegionalTacs.capitalize_first_char_of_str
        for part in split_str:
            capped_str = capitalize_first(input_str=part)
            capped_split_str += [capped_str]
        camel_case_str = ''.join(capped_split_str)
        return camel_case_str


    def extract_tac_and_write(self,
                              region_mapping,
                              region_name):
        """
        Run self.tac_extraction_func on one region and save results to image.
        """
        extracted_tac, uncertainty = self.tac_extraction_func(input_img=self.pet_img,
                                            segmentation_img=self.seg_img,
                                            region=int(region_mapping))
        region_tac_file = TimeActivityCurve(times=self.scan_timing.center_in_mins,
                                            activity=extracted_tac,
                                            uncertainty=uncertainty)
        out_tac_path = os.path.join(self.out_tac_dir,
                                    f'{self.out_tac_prefix}_seg-{region_name}_tac.tsv')
        region_tac_file.to_tsv(filename=out_tac_path)


    def write_tacs(self, label_map_path: str=None):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and label map. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.
        """
        unique_segmentation_labels = np.unique(self.seg_img.numpy())

        if label_map_path is not None:
            label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
            regions_abrev = [self.str_to_camel_case(label) for label in label_map['abbreviation']]
            regions_map = label_map['mapping']
        else:
            regions_map = [int(label) for label in unique_segmentation_labels]
            regions_abrev = [str(label) for label in regions_map]

        for i, _label in enumerate(unique_segmentation_labels):
            self.extract_tac_and_write(regions_map[i],
                                       regions_abrev[i])


    def __call__(self, *args, **kwargs):
        self.write_tacs(*args, **kwargs)
