"""
Extracting TACs from masks or regions, getting statistics, and writing to file.
"""
import ants
import numpy as np

from ..utils.useful_functions import check_physical_space_for_ants_image_pair


def apply_mask_4d(input_image: ants.core.ANTsImage | np.ndarray,
                  mask_image: ants.core.ANTsImage | np.ndarray,
                  verbose: bool = False) -> np.ndarray:
    """
    Function to extract ROI voxel tacs from an image using a mask image.

    This function applies a 3D mask to a 4D image, returning the time series for each voxel in a
    single flattened numpy array.

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
    if isinstance(input_image, ants.ANTsImage) and isinstance(mask_image, ants.ANTsImage):
        assert check_physical_space_for_ants_image_pair(input_image, mask_image), (
            "Images must have the same physical dimensions.")
        input_arr = input_image.numpy()
        mask_arr = mask_image.numpy()
    else:
        assert input_image.shape[:3] == mask_image.shape, (
            "Images must have the same physical dimensions.")
        input_arr = input_image.copy()
        mask_arr = mask_image.copy()

    x_inds, y_inds, z_inds = mask_arr.nonzero()
    out_voxels = input_arr[x_inds, y_inds, z_inds, :]
    if verbose:
        print(f"(ImageOps): Output TACs have shape {out_voxels.shape}")
    return out_voxels
