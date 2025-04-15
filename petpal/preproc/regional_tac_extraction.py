"""
Regional TAC extraction
"""
import os
import nibabel
import numpy as np
import ants

from ..utils import image_io
from ..utils.useful_functions import check_physical_space_for_ants_image_pair


def extract_mean_roi_tac_from_nifti_using_segmentation(input_image_4d_numpy: np.ndarray,
                                                       segmentation_image_numpy: np.ndarray,
                                                       region: int,
                                                       verbose: bool) -> np.ndarray:
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

    if verbose:
        print(f'Running TAC for region index {region}')
    masked_voxels = (seg_image > region - 0.1) & (seg_image < region + 0.1)
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)
    return tac_out


def write_tacs(input_image_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
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
        extracted_tac = tac_extraction_func(input_image_4d_numpy=pet_numpy,
                                            segmentation_image_numpy=seg_numpy,
                                            region=int(regions_map[i]),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{regions_abrev[i]}_mean_activity'
        if out_tac_prefix:
            out_tac_path = os.path.join(out_tac_dir, f'{out_tac_prefix}_seg-{regions_abrev[i]}_tac.tsv')
        else:
            out_tac_path = os.path.join(out_tac_dir, f'seg-{regions_abrev[i]}_tac.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


def roi_tac(input_image_4d_path: str,
            roi_image_path: str,
            region: int,
            out_tac_path: str,
            verbose: bool,
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
                                        region=region,
                                        verbose=verbose)
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
