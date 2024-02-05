"""
A module for performing basic motion correction on 4D PET scans.

Example:
    python pet_moco.py -i pet.nii -o pet_mc.nii --cost_func normcorr

Attributes:

Todo:
    * Write motion correction module
    * Test motion correction module
"""
import numpy as np
import nibabel
import ants


def calculate_pet_motion_correction(pet_image_series: np.ndarray,
                                    ) -> np.ndarray:
    """
    Takes an input 4D PET reconstructed time series image, 
    calculates the motion correction, and returns the motion
    corrected image.

    Args:
        pet_image_series (np.ndarray): Input 4D PET image series to be motion corrected.
        
    Returns:
        pet_image_moco (np.ndarray): The motion corrected 4D PET image.
    
    """
    pet_image_ants = ants.from_numpy(pet_image_series)
    pet_image_moco, moco_params, moco_fd = ants.motion_correction(pet_image_ants)
    return pet_image_moco, moco_params, moco_fd


def write_pet_motion_correction(pet_file: str, moco_output_file: str, **kwargs) -> int:
    """
    Function to call ``calculate_pet_motion_correction`` PET motion
    correction function, and write result to file.

    Args:
        pet_file (str): Path to input 4D PET image file
        moco_output_file (str): Path to output motion corrected PET file
    """
    return 0
