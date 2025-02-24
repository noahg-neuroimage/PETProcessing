"""
Library for math functions for use elsewhere.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import ants

def weighted_sum_computation(pet_series: ants.core.ANTsImage | np.ndarray,
                             frame_duration: np.ndarray,
                             half_life: float,
                             frame_start: np.ndarray,
                             decay_correction: np.ndarray):
    """
    Weighted sum of a PET image based on time and re-corrected for decay correction.

    Args:
        image_frame_duration (np.ndarray): Duration of each frame in pet series
        half_life (float): Half life of tracer radioisotope in seconds.
        pet_series (np.ndarray): 4D PET image series, as a data array.
        image_frame_start (np.ndarray): Start time of each frame in pet series,
            measured with respect to scan TimeZero.
        image_decay_correction (np.ndarray): Decay correction factor that scales
            each frame in the pet series. 

    Returns:
        image_weighted_sum (np.ndarray): 3D PET image computed by reversing decay correction
            on the PET image series, scaling each frame by the frame duration, then re-applying
            decay correction and scaling the image to the full duration.

    See Also:
        * :meth:`petpal.image_operations_4d.weighted_series_sum`: Function where this is implemented.

    """
    decay_constant = np.log(2.0) / half_life
    image_total_duration = np.sum(frame_duration)
    total_decay = decay_constant * image_total_duration
    total_decay /= 1.0 - np.exp(-1.0 * decay_constant * image_total_duration)
    total_decay /= np.exp(-1 * decay_constant * frame_start[0])
    
    pet_series_scaled = pet_series[:, :, :] * frame_duration / decay_correction
    pet_series_sum_scaled = pet_series_scaled.sum(axis=3)
    image_weighted_sum = pet_series_sum_scaled * total_decay / image_total_duration
    return image_weighted_sum


def weighted_sum_computation_over_index_window(pet_series: ants.core.ANTsImage | np.ndarray,
                                               window_start_id: int,
                                               window_end_id: int,
                                               half_life: float,
                                               frame_durations: np.ndarray,
                                               frame_starts: np.ndarray,
                                               decay_factors: np.ndarray):
    """
    Computes a weighted sum over a specified index window for a 4D PET image series. See
    :func:`weighted_sum_computation` for the computation.

    Args:
        pet_series (ants.core.ANTsImage | np.ndarray): PET image series as an ANTs image or a NumPy array.
        window_start_id (int): Start index of the window.
        window_end_id (int): End index of the window.
        half_life (float): Radioactive tracer's half-life in seconds.
        frame_durations (np.ndarray): Frame durations in seconds corresponding to the PET image series.
        frame_starts (np.ndarray): Frame start times in seconds corresponding to the PET image series.
        decay_factors (np.ndarray): Decay correction factors for the PET image series.

    Returns:
        np.ndarray: Weighted sum for the image series over the specified window.
    """
    window_image_series = pet_series[:, :, :, window_start_id:window_end_id]

    sub_frm_dur = frame_durations[window_start_id:window_end_id]
    sub_frm_st = frame_starts[window_start_id:window_end_id]
    sub_frm_decay = decay_factors[window_start_id:window_end_id]

    window_wss = weighted_sum_computation(frame_duration=sub_frm_dur,
                                          half_life=half_life,
                                          pet_series=window_image_series,
                                          frame_start=sub_frm_st,
                                          decay_correction=sub_frm_decay)

    return window_wss


def gauss_blur_computation(input_image: np.ndarray,
                           blur_size_mm: float,
                           input_zooms: list,
                           use_fwhm: bool):
    """
    Applies a Gaussian blur to an array image. Function intended to be a
    wrapper to be applied by other methods.
    """
    if use_fwhm:
        blur_size = blur_size_mm / (2*np.sqrt(2*np.log(2)))
    else:
        blur_size = blur_size_mm

    sigma_x = blur_size / input_zooms[0]
    sigma_y = blur_size / input_zooms[1]
    sigma_z = blur_size / input_zooms[2]

    blur_image = gaussian_filter(input=input_image,
                                 sigma=(sigma_x,sigma_y,sigma_z),
                                 axes=(0,1,2))
    return blur_image
