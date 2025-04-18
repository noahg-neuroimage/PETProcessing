"""
Regional TAC extraction
"""
import re
import os
import pathlib
import nibabel
import numpy as np
import ants

from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo
from ..utils.useful_functions import check_physical_space_for_ants_image_pair
from ..utils.time_activity_curve import TimeActivityCurve


def extract_mean_roi_tac_from_nifti_using_segmentation(input_image_4d_numpy: np.ndarray,
                                                       segmentation_image_numpy: np.ndarray,
                                                       region: int) -> np.ndarray:
    """
    Creates a time-activity curve (TAC) by computing the average value within a region, for each 
    frame in a 4D PET image series. Takes as input a PET image, which has been registered to
    anatomical space, a segmentation image, with the same sampling as the PET, and a list of values
    corresponding to regions in the segmentation image that are used to compute the average
    regional values. Currently, only the mean over a single region value is implemented.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions. Must have same sampling as PET
            input.
        region (int): Value in the segmentation image corresponding to a region
            over which the TAC is computed.
        verbose (bool): Set to ``True`` to output processing information.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        ValueError: If the segmentation image and PET image have different
            sampling.
    """

    pet_image_4d = input_image_4d_numpy
    if len(pet_image_4d.shape)==4:
        num_frames = pet_image_4d.shape[3]
    else:
        num_frames = 1
    seg_image = segmentation_image_numpy

    if seg_image.shape[:3]!=pet_image_4d.shape[:3]:
        raise ValueError('Mis-match in image shape of segmentation image '
                         f'({seg_image.shape}) and PET image '
                         f'({pet_image_4d.shape[:3]}). Consider resampling '
                         'segmentation to PET or vice versa.')

    masked_voxels = (seg_image > region - 0.1) & (seg_image < region + 0.1)
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)
    uncertainty = np.std(masked_image, axis=0)
    return tac_out, uncertainty


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               time_frame_keyword: str = 'FrameReferenceTime',
               out_tac_prefix: str = '', ):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and label map. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.load_metadata_for_nifti_with_same_filename(input_image_path)
    label_map = image_io.ImageIO.read_label_map_tsv(label_map_file=label_map_path)
    regions_abrev = label_map['abbreviation']
    regions_map = label_map['mapping']

    tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
    pet_numpy = nibabel.load(input_image_path).get_fdata()
    seg_numpy = nibabel.load(segmentation_image_path).get_fdata()

    for i, _maps in enumerate(label_map['mapping']):
        extracted_tac, uncertainty = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(regions_map[i]))
        region_tac_file = TimeActivityCurve(times=pet_meta[time_frame_keyword],
                                            activity=extracted_tac,
                                            uncertainty=uncertainty)
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir, f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        region_tac_file.to_tsv(filename=out_tac_path)
        np.savetxt(out_tac_path,region_tac_file.tac_werr,delimiter='\t',header=header_text,comments='')


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: int,
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
    tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
    pet_numpy = nibabel.load(input_image_4d_path).get_fdata()
    seg_numpy = nibabel.load(roi_image_path).get_fdata()


    extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                        segmentation_image_numpy=seg_numpy,
                                        region=region)
    region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
    header_text = 'mean_activity'
    np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


def extract_roi_voxel_tacs_from_image_using_mask(input_image: ants.core.ANTsImage,
                                                 mask_image: ants.core.ANTsImage,
                                                 verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an image using a mask image.

    This function returns all the voxel TACs, and unlike :func:`extract_mean_roi_tac_from_nifti_using_segmentation`,
    does not calculate the mean over all the voxels.

    Args:
        input_image (ants.core.ANTsImage): Input 4D-image from which to extract ROI voxel tacs.
        mask_image (ants.core.ANTsImage): Mask image which determines which voxels to extract.
        verbose (bool, optional): If True, prints information about the shape of extracted voxel tacs.

    Returns:
        out_voxels (np.ndarray): Array of voxel TACs of shape (num_voxels, num_frames)

    Raises:
         AssertionError: If input image is not 4D-image.
         AssertionError: If mask image is not in the same physical space as the input image.

    """
    assert len(input_image.shape) == 4, "Input image must be 4D."
    assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
        "Images must have the same physical dimensions.")

    x_inds, y_inds, z_inds = mask_image.nonzero()
    out_voxels = input_image.numpy()[x_inds, y_inds, z_inds, :]
    if verbose:
        print(f"(ImageOps): Output TACs have shape {out_voxels.shape}")
    return out_voxels


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
                 out_tac_dir: str | pathlib.Path,
                 tac_extraction_func: callable=None,):
        self.pet_img = ants.image_read(filename=input_image_path)
        self.seg_img = ants.image_read(filename=segmentation_path)
        if tac_extraction_func is None:
            self.tac_extraction_func = extract_mean_roi_tac_from_nifti_using_segmentation
        self.out_tac_prefix = out_tac_prefix
        self.out_tac_dir = out_tac_dir
        self.scan_timing = ScanTimingInfo.from_nifti(input_image_path)


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

        Special characters (? * - _) are removed and treated as word separaters. Different words are
        then capitalized at the first character, leaving other alphanumeric characters unchanged.
        """
        split_str = re.split(r'[-_?*]', input_str)
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
        extracted_tac, uncertainty = self.tac_extraction_func(input_image_4d_numpy=self.pet_img.numpy(),
                                            segmentation_image_numpy=self.seg_img.numpy(),
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
