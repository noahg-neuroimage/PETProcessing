import functools
import ants

from ..preproc.segmentation_tools import calc_vesselness_mask_from_quantiled_vesselness, calc_vesselness_measure_image


def ANTsImageToANTsImage(func):
    """
    A decorator for functions that process an ANTs image and output another ANTs image.
    Assumes that the argument of the passed in function is an ANTs image.

    This decorator is designed to extend functions that take an ANTs image as input
    and output another ANTs image. It supports seamless handling of input images
    provided as either file paths (str) or `ants.core.ANTsImage` objects. The resulting
    processed image can optionally be saved to a specified file path.

    Args:
        func (Callable): The function to be decorated. It should accept an ANTs image as
            the first argument and return a processed ANTs image.

    Returns:
        Callable: A wrapper function that:
            - Reads the input image if a file path (str) is provided.
            - Passes an `ants.core.ANTsImage` object to the decorated function.
            - Saves the output image to the specified file path if `out_path` is provided.

    Wrapper Parameters:
        in_img (ants.core.ANTsImage | str): Input image, either an ANTsImage object or
            the file path to a NIfTI image.
        out_path (str): File path to save the output image. If `None`, the output image
            is not saved.
        *args: Additional positional arguments for the decorated function.
        **kwargs: Additional keyword arguments for the decorated function.

    Raises:
        TypeError: If `in_img` is not a string or `ants.core.ANTsImage`.

    Notes:
        - If `in_img` is provided as a file path, the image is read using `ants.image_read`.
        - The output image is written to the desired path using `ants.image_write` if
          `out_path` is specified.
    """

    @functools.wraps(func)
    def wrapper(in_img:ants.core.ANTsImage | str,
                out_path: str,
                *args, **kwargs):
        if isinstance(in_img, str):
            in_image = ants.image_read(in_img)
        elif isinstance(in_img, ants.core.ANTsImage):
            in_image = in_img
        else:
            raise TypeError('in_img must be str or ants.core.ANTsImage')
        out_img = func(in_image, *args, **kwargs)
        if out_path is not None:
            ants.image_write(out_img, out_path)
        return out_img
    return wrapper
