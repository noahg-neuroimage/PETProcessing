"""
Provides methods to motion correct 4D PET data. Includes method
:meth:`determine_motion_target`, which produces a flexible target based on the
4D input data to optimize contrast when computing motion correction or
registration.
"""
import ants
import numpy as np


from .image_operations_4d import determine_motion_target
from ..utils import image_io
from ..utils.scan_timing import ScanTimingInfo, get_window_index_pairs_for_image
from ..utils.useful_functions import weighted_series_sum_over_window_indecies
from ..utils.image_io import get_half_life_from_nifti


def motion_corr(input_image_4d_path: str,
                motion_target_option: str | tuple,
                out_image_path: str,
                verbose: bool,
                type_of_transform: str = 'DenseRigid',
                half_life: float = None,
                **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion
    correction module from Advanced Normalisation Tools (ANTs) with default
    inputs.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the
            motion corrected PET series is written.
        verbose (bool): Set to ``True`` to output processing information.
        type_of_transform (str): Type of transform to perform on the PET image,
            must be one of antspy's transformation types, i.e. 'DenseRigid' or
            'Translation'. Any transformation type that uses >6 degrees of
            freedom is not recommended, use with caution. See
            :py:func:`ants.registration`.
        half_life (float): Half life of the PET radioisotope in seconds. Used
            for certain settings of ``motion_target_option``.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy
            array.
        pet_moco_params (list[str]): List of ANTS registration files applied to
            each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure
            corresponding to each frame transform.
    """
    pet_ants = ants.image_read(input_image_4d_path)
    motion_target_image_path = determine_motion_target(motion_target_option=motion_target_option,
                                                       input_image_4d_path=input_image_4d_path,
                                                       half_life=half_life)

    motion_target_image = ants.image_read(motion_target_image_path)
    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=motion_target_image,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    ants.image_write(image=pet_moco_ants,filename=out_image_path)
    image_io.safe_copy_meta(input_image_path=input_image_4d_path, out_image_path=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def motion_corr_frame_list(input_image_4d_path: str,
                           motion_target_option: str | tuple,
                           out_image_path: str,
                           verbose: bool,
                           frames_list: list = None,
                           type_of_transform: str = 'Affine',
                           transform_metric: str = 'mattes',
                           half_life: float = None,
                           **kwargs):
    r"""
    Perform per-frame motion correction on a 4D PET image.

    This function applies motion correction to each frame of a 4D PET image based on a specified
    motion target. Only the frames in ``frames_list`` are motion corrected, all else are kept as is.

    Args:
        input_image_4d_path (str): Path to the input 4D PET image file.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        frames_list (list, optional): List of frame indices to correct. If None, corrects all
            frames. Default is None.
        type_of_transform (str, optional): Type of transformation to use for registration. Default
            is 'Affine'.
        transform_metric (str, optional): Metric to use for the transformation. Default is
            'mattes'.
        half_life (float, optional): Half-life value used by `determine_motion_target` if
            applicable. Default is None.
        **kwargs: Additional arguments passed to the `ants.registration` method.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frame_list

            motion_corr_frame_list(input_image_4d_path='/path/to/image.nii.gz',
                                  motion_target_option='/path/to/target_image.nii.gz',
                                  out_image_path='/path/to/output_motion_corrected.nii.gz',
                                  verbose=True)

    Notes:
        - The :func:`determine_motion_target` function is used to derive the motion target image
            based on the specified option.
        - If `frames_list` is not provided, all frames of the 4D image will be corrected.
        - Motion correction is performed using the :py:func:`ants.registration` method from the
            ANTsPy library.
        - The corrected frames are reassembled into a 4D image and saved to the specified output
            path.

    """
    input_image = ants.image_read(input_image_4d_path)

    motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                 input_image_4d_path=input_image_4d_path,
                                                 half_life=half_life)
    motion_target = ants.image_read(motion_target_path)

    frames_to_correct = np.zeros(input_image.shape[-1], dtype=bool)

    if frames_list is None:
        _correct_these_frames = np.ones(input_image.shape[-1], dtype=int)
        frames_to_correct[list(_correct_these_frames)] = True
    else:
        assert max(frames_list) < input_image.shape[-1]
        frames_to_correct[list(frames_list)] = True

    out_image = []
    input_image_list = input_image.ndimage_to_list()

    if verbose:
        print("(Info): On frame:", end=' ')

    for frame_id, moco_this_frame in enumerate(frames_to_correct):
        if verbose:
            print(f"{frame_id:>02}", end=' ')
        this_frame = input_image_list[frame_id]
        if moco_this_frame:
            tmp_reg = ants.registration(fixed=motion_target,
                                        moving=this_frame,
                                        type_of_transform=type_of_transform,
                                        aff_metric=transform_metric,
                                        interpolator='linear',
                                        reg_iterations=(),
                                        **kwargs)
            out_image.append(tmp_reg['warpedmovout'])
        else:
            out_image.append(this_frame)

    if verbose:
        print("... done!\n")
    tmp_image = gen_nd_image_based_on_image_list(out_image)
    out_image = ants.list_to_ndimage(tmp_image, out_image)
    ants.image_write(image=out_image,filename=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")


def motion_corr_frame_list_to_t1(input_image_4d_path: str,
                                 t1_image_path: str,
                                 motion_target_option: str | tuple,
                                 out_image_path: str,
                                 verbose: bool,
                                 frames_list: list = None,
                                 type_of_transform: str = 'AffineFast',
                                 transform_metric: str = "mattes",
                                 half_life: float = None):
    r"""
    Perform motion correction of a 4D PET image to a T1 anatomical image.

    This function corrects motion in a 4D PET image by registering it to a T1 anatomical
    image. The method uses a two-step process: first registering an intermediate motion
    target to the T1 image (either the time-averaged image or a weighted-series-sum), and
    then using the calculated transform to correct motion in individual frames of the PET series.
    The motion-target-option is registered to the T1 anatomical image. Then, given the frames in
    the frame list, the frames are registered to the T1 image, and all other frames are simply
    transformed to the motion-target in T1-space.

    Args:
        input_image_4d_path (str): Path to the 4D PET image to be corrected.
        t1_image_path (str): Path to the 3D T1 anatomical image.
        motion_target_option (str | tuple): Option for selecting the motion target image.
            Can be a path to a file or a tuple range. If None, the average of the PET timeseries
            is used.
        out_image_path (str): Path to save the motion-corrected 4D image.
        verbose (bool): Set to True to print verbose output during processing.
        frames_list (list, optional): List of frame indices to correct. If None, all frames
            are corrected & registered.
        type_of_transform (str): Type of transformation used in registration. Default is
            'AffineFast'.
        transform_metric (str): Metric for transformation optimization. Default is 'mattes'.
        half_life (float, optional): Half-life of the PET radioisotope. Used if a calculation
            is required for the motion target.

    Returns:
        None

    Raises:
        AssertionError: If maximum frame index in `frames_list` exceeds the number of frames in the
            PET image.

    Example:

        .. code-block:: python


            motion_corr_frame_list_to_t1(input_image_4d_path='pet_timeseries.nii.gz',
                              t1_image_path='t1_image.nii.gz',
                              motion_target_option='mean_image',
                              out_image_path='pet_corrected.nii.gz',
                              verbose=True)

    """

    input_image = ants.image_read(input_image_4d_path)
    t1_image = ants.image_read(t1_image_path)

    motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                 input_image_4d_path=input_image_4d_path,
                                                 half_life=half_life)
    motion_target = ants.image_read(motion_target_path)

    motion_target_to_mpr_reg = ants.registration(fixed=t1_image,
                                                 moving=motion_target,
                                                 type_of_transform=type_of_transform,
                                                 aff_metric=transform_metric, )

    motion_target_in_t1 = motion_target_to_mpr_reg['warpedmovout']
    motion_transform_matrix = motion_target_to_mpr_reg['fwdtransforms']

    frames_to_correct = np.zeros(input_image.shape[-1], dtype=bool)

    if frames_list is None:
        _correct_these_frames = np.ones(input_image.shape[-1], dtype=int)
        frames_to_correct[list(_correct_these_frames)] = True
    else:
        assert max(frames_list) < input_image.shape[-1]
        frames_to_correct[list(frames_list)] = True

    out_image = []
    input_image_list = input_image.ndimage_to_list()

    if verbose:
        print("(Info): On frame:", end=' ')

    for frame_id, moco_this_frame in enumerate(frames_to_correct):
        if verbose:
            print(f"{frame_id:>02}", end=' ')
        this_frame = input_image_list[frame_id]
        if moco_this_frame:
            tmp_reg = ants.registration(fixed=motion_target_in_t1,
                                        moving=this_frame,
                                        type_of_transform=type_of_transform,
                                        aff_metric=transform_metric,
                                        interpolator='linear')
            out_image.append(tmp_reg['warpedmovout'])
        else:
            tmp_transform = ants.apply_transforms(fixed=motion_target_in_t1,
                                                  moving=this_frame,
                                                  transformlist=motion_transform_matrix,
                                                  interpolator='linear')
            out_image.append(tmp_transform)

    if verbose:
        print("... done!\n")
    tmp_image = gen_nd_image_based_on_image_list(out_image)
    out_image = ants.list_to_ndimage(tmp_image, out_image)
    ants.image_write(image=out_image,filename=out_image_path)

    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")


def motion_corr_frames_above_mean_value(input_image_4d_path: str,
                                        out_image_path: str,
                                        motion_target_option: str | tuple,
                                        verbose: bool,
                                        type_of_transform: str = 'Affine',
                                        transform_metric: str = 'mattes',
                                        half_life: float = None,
                                        scale_factor=1.0,
                                        **kwargs):
    r"""
    Perform motion correction on frames with mean values above the mean of a 4D PET image.

    This function applies motion correction only to the frames in a 4D PET image whose mean voxel
    values are greater than the overall mean voxel value of the entire image. It internally
    utilizes the :func:`motion_corr_frame_list` function to perform the motion correction.

    Args:
        input_image_4d_path (str): Path to the input 4D PET image file.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        type_of_transform (str, optional): Type of transformation to use for registration.
            Default is 'Affine'.
        transform_metric (str, optional): Metric to use for the transformation. Default is
            'mattes'.
        half_life (float, optional): Half-life value used by `determine_motion_target`, if
            applicable. Default is None.
        scale_factor (float, optional): Scale factor to apply to frame mean values before
            comparison. Default is 1.0.
        **kwargs: Additional arguments passed to the `ants.registration` method.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frames_above_mean_value

            motion_corr_frames_above_mean_value(input_image_4d_path='/path/to/image.nii.gz',
                                                motion_target_option='/path/to/target_image.nii.gz',
                                                out_image_path='/path/to/output_motion_corrected.nii.gz',
                                                verbose=True,
                                                type_of_transform='Affine',
                                                transform_metric='mattes',
                                                scale_factor=1.2)

    Notes:
        - Uses :func:`motion_corr_frame_list` for the actual motion correction of specified frames.
        - Frames with mean voxel values greater than the total mean voxel value (optionally scaled
            by `scale_factor`) are selected for motion correction.
        - The :func:`_get_list_of_frames_above_total_mean` function is used to
            identify the frames to be motion corrected based on their mean voxel values.

    """

    frames_list = _get_list_of_frames_above_total_mean(image_4d_path=input_image_4d_path,
                                                       scale_factor=scale_factor)

    motion_corr_frame_list(input_image_4d_path=input_image_4d_path,
                           motion_target_option=motion_target_option,
                           out_image_path=out_image_path,
                           verbose=verbose,
                           frames_list=frames_list,
                           type_of_transform=type_of_transform,
                           transform_metric=transform_metric,
                           half_life=half_life,
                           **kwargs)


def motion_corr_frames_above_mean_value_to_t1(input_image_4d_path: str,
                                              t1_image_path: str,
                                              motion_target_option: str | tuple,
                                              out_image_path: str,
                                              verbose: bool,
                                              type_of_transform: str = 'AffineFast',
                                              transform_metric: str = "mattes",
                                              half_life: float = None,
                                              scale_factor: float = 1.0):
    """
    Perform motion correction on frames with mean values above the mean of a 4D PET image to a T1
    anatomical image.

    This function applies motion correction only to the frames in a 4D PET image whose mean voxel
    values are greater than the overall mean voxel value of the entire image. It corrects these
    frames by registering them to a T1 anatomical image, using the `motion_corr_frame_list_to_t1`
    function.

    Args:
        input_image_4d_path (str): Path to the input 4D PET image file.
        t1_image_path (str): Path to the 3D T1 anatomical image.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        out_image_path (str): Path to save the motion-corrected output image.
        verbose (bool): Whether to print verbose output during processing.
        type_of_transform (str, optional): Type of transformation to use for registration. Default
            is 'AffineFast'.
        transform_metric (str, optional): Metric to use for the transformation. Default is 'mattes'.
        half_life (float, optional): Half-life value used by `determine_motion_target`, if
            applicable. Default is None.
        scale_factor (float, optional): Scale factor applied to the mean voxel value of the entire
            image for comparison. Must be greater than 0. Default is 1.0.

    Returns:
        None

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import motion_corr_frames_above_mean_value_to_t1

            motion_corr_frames_above_mean_value_to_t1(input_image_4d_path='/path/to/image.nii.gz',
                                                      t1_image_path='/path/to/t1_image.nii.gz',
                                                      motion_target_option='/path/to/target_image.nii.gz',
                                                      out_image_path='/path/to/output_motion_corrected.nii.gz',
                                                      verbose=True,
                                                      type_of_transform='AffineFast',
                                                      transform_metric='mattes',
                                                      scale_factor=1.2)

    Notes:
        - This function internally uses :func:`motion_corr_frame_list_to_t1` for the actual motion
            correction of specified frames.
        - Frames with mean voxel values greater than the total mean voxel value (optionally scaled
            by `scale_factor`) are selected for motion correction.
        - The :func:`_get_list_of_frames_above_total_mean` function identifies
            the frames to be motion corrected based on their mean voxel values.
    """
    frames_list = _get_list_of_frames_above_total_mean(image_4d_path=input_image_4d_path,
                                                       scale_factor=scale_factor)

    motion_corr_frame_list_to_t1(input_image_4d_path=input_image_4d_path,
                                 t1_image_path=t1_image_path,
                                 motion_target_option=motion_target_option,
                                 out_image_path=out_image_path,
                                 verbose=verbose,
                                 frames_list=frames_list,
                                 type_of_transform=type_of_transform,
                                 transform_metric=transform_metric,
                                 half_life=half_life)


def windowed_motion_corr_to_target(input_image_path: str,
                                   out_image_path: str | None,
                                   motion_target_option: str | tuple,
                                   w_size: float,
                                   type_of_transform: str = 'QuickRigid',
                                   interpolator: str = 'linear',
                                   copy_metadata: bool = True,
                                   **kwargs):
    """
    Performs windowed motion correction (MoCo) to align frames of a 4D PET image to a given target image.
    We compute a combined image over the frames in a window, which is registered to the target image.
    Then, for each frame within the window, the same transformation is applied. This can be useful for
    initial frames with low counts. By setting a small-enough window size, later frames can still be
    individually registered to the target image.

    .. important::
        The motion-target will determine the space of the output image. If we provide a T1 image
        as the `motion_target_option`, the output image will be in T1-space.

    Args:
        input_image_path (str): Path to the input 4D PET image file.
        out_image_path (str | None): Path to save the resulting motion-corrected image. If
            None, don't save image to disk.
        motion_target_option (str | tuple): Option to determine the motion target. This can
            be a path to a specific image file, a tuple of frame indices to generate a target, or
            specific options recognized by :func:`determine_motion_target`.
        w_size (float): Window size in seconds for dividing the image into time sections.
        type_of_transform (str): Type of transformation to use in registration (default: 'QuickRigid').
        interpolator (str): Interpolation method for the transformation (default: 'linear').
        **kwargs: Additional arguments passed to :func:`ants.registration`.

    Returns:
        ants.core.ANTsImage: Motion-corrected 4D image.

    Workflow:
        1. Reads the input 4D image and splits it into individual frames.
        2. Computes index windows based on the specified window size (`w_size`).
        3. Extracts necessary frame timing information and the tracer's half-life.
        4. For each window:
            - Calculates a weighted sum image for the window.
              See :func:`petpal.utils.useful_functions.weighted_series_sum_over_window_indecies`.
            - Performs registration of the weighted sum image to the target image.
            - Applies the obtained transformations to each frame within the window.
        5. Combines the transformed frames into a corrected 4D image.
        6. Saves the output image to the specified path, if provided.

    Note:
        If `out_image_path` is provided, the corrected 4D image will be saved to the specified path.
    """
    input_image = ants.image_read(filename=input_image_path)
    input_image_list = ants.ndimage_to_list(input_image)
    window_idx_pairs = get_window_index_pairs_for_image(image_path=input_image_path, w_size=w_size)
    half_life = get_half_life_from_nifti(image_path=input_image_path)
    frame_timing_info = ScanTimingInfo.from_nifti(image_path=input_image_path)

    target_image = determine_motion_target(motion_target_option=motion_target_option,
                                           input_image_4d_path=input_image_path,
                                           half_life=half_life)
    target_image = ants.image_read(target_image)

    reg_kwargs_default = {'aff_metric'               : 'mattes',
                          'write_composite_transform': True}
    reg_kwargs = {**reg_kwargs_default, **kwargs}

    out_image = []
    for win_id, (st_id, end_id) in enumerate(zip(*window_idx_pairs)):
        window_tgt_image = weighted_series_sum_over_window_indecies(input_image_4d=input_image,
                                                                    output_image_path=None,
                                                                    window_start_id=st_id,
                                                                    window_end_id=end_id,
                                                                    half_life=half_life,
                                                                    image_frame_info=frame_timing_info)
        window_registration = ants.registration(fixed=target_image,
                                                moving=window_tgt_image,
                                                type_of_transform=type_of_transform,
                                                interpolator=interpolator,
                                                **reg_kwargs)
        for frm_id in range(st_id, end_id):
            out_image.append(ants.apply_transforms(fixed=target_image,
                                                   moving=input_image_list[frm_id],
                                                   transformlist=window_registration['fwdtransforms']))

    out_image = gen_timeseries_from_image_list(out_image)

    if out_image_path is not None:
        ants.image_write(image=out_image, filename=out_image_path)

    if copy_metadata:
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=out_image_path)
    return out_image

def gen_timeseries_from_image_list(image_list: list[ants.core.ANTsImage]) -> ants.core.ANTsImage:
    r"""
    Takes a list of ANTs ndimages, and generates a 4D ndimage. Undoes :func:`ants.ndimage_to_list`
    so that we take a list of 3D images and generates a 4D image.

    Args:
        image_list (list[ants.core.ANTsImage]): A list of ndimages.

    Returns:
        ants.core.ANTsImage: 4D ndimage.
    """
    tmp_image = gen_nd_image_based_on_image_list(image_list)
    return ants.list_to_ndimage(tmp_image, image_list)


def gen_nd_image_based_on_image_list(image_list: list[ants.core.ants_image.ANTsImage]):
    r"""
    Generate a 4D ANTsImage based on a list of 3D ANTsImages.

    This function takes a list of 3D ANTsImages and constructs a new 4D ANTsImage,
    where the additional dimension represents the number of frames (3D images) in the list.
    The 4D image retains the spacing, origin, direction, and shape properties of the 3D images,
    with appropriate modifications for the additional dimension.

    Args:
        image_list (list[ants.core.ants_image.ANTsImage]):
            List of 3D ANTsImage objects to be combined into a 4D image.
            The list must contain at least one image, and all images must have the same
            dimensions and properties.

    Returns:
        ants.core.ants_image.ANTsImage:
            A 4D ANTsImage constructed from the input list of 3D images. The additional
            dimension corresponds to the number of frames (length of the image list).

    Raises:
        AssertionError: If the `image_list` is empty or if the images in the list are not 3D.

    See Also
        * :func:`petpal.preproc.motion_corr.motion_corr_frame_list_to_t1`

    Example:

        .. code-block:: python


            import ants
            image1 = ants.image_read('frame1.nii.gz')
            image2 = ants.image_read('frame2.nii.gz')
            image_list = [image1, image2]
            result = _gen_nd_image_based_on_image_list(image_list)
            print(result.dimension)  # 4
            image4d = ants.list_to_ndimage(result, image_list)

    """
    assert len(image_list) > 0
    assert image_list[0].dimension == 3

    num_frames = len(image_list)
    spacing_3d = image_list[0].spacing
    origin_3d = image_list[0].origin
    shape_3d = image_list[0].shape
    direction_3d = image_list[0].direction

    direction_4d = np.eye(4)
    direction_4d[:3, :3] = direction_3d
    spacing_4d = (*spacing_3d, 1.0)
    origin_4d = (*origin_3d, 0.0)
    shape_4d = (*shape_3d, num_frames)

    tmp_image = ants.make_image(imagesize=shape_4d,
                                spacing=spacing_4d,
                                origin=origin_4d,
                                direction=direction_4d)
    return tmp_image


def _get_list_of_frames_above_total_mean(image_4d_path: str,
                                         scale_factor: float = 1.0):
    """
    Get the frame indices where the frame mean is higher than the total mean of a 4D image.

    This function calculates the mean voxel value of each frame in a 4D image and returns the
        indices of the frames whose mean voxel value is greater than or equal to the mean voxel
        value of the entire image, optionally scaled by a provided factor.

    Args:
        image_4d_path (str): Path to the input 4D PET image file.
        scale_factor (float, optional): Scale factor applied to the mean voxel value of the entire
            image for comparison. Must be greater than 0. Default is 1.0.

    Returns:
        list: A list of frame indices where the frame mean voxel value is greater than or equal to
            the scaled total mean voxel value.

    Example:

        .. code-block:: python

            from petpal.preproc.motion_corr import _get_list_of_frames_above_total_mean

            frame_ids = _get_list_of_frames_above_total_mean(image_4d_path='/path/to/image.nii.gz',
                                                                                  scale_factor=1.2)

            print(frame_ids)  # Output: [0, 3, 5, ...]

    Notes:
        - The :func:`ants.image_read` from ANTsPy is used to read the 4D image into memory.
        - The mean voxel value of the entire image is scaled by `scale_factor` for comparison with
            individual frame means.
        - The function uses the :func:`ants.ndimage_to_list` method from ANTsPy to convert the 4D
            image into a list of 3D frames.

    """
    assert scale_factor > 0
    image = ants.image_read(image_4d_path)
    total_mean = scale_factor * image.mean()

    frames_list = []
    for frame_id, a_frame in enumerate(image.ndimage_to_list()):
        if a_frame.mean() >= total_mean:
            frames_list.append(frame_id)

    return frames_list
