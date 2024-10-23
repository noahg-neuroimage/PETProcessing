"""
This script provides a command-line interface (CLI) to calculate the image-derived input function (IDIF) using necktangle methods.
Users can choose between the single threshold method and the double threshold method.
The script supports the following functionalities:
- Single threshold method: Calculates the time-activity curve (TAC) from PET data using a single percentile threshold.
- Double threshold method: Computes the IDIF from a 4D PET necktangle matrix using a double threshold method.
Requires:
    The script relies on the :doc:`argparse <argparse:index>`, :doc:`numpy <numpy:index>`, and :doc:`nibabel <nibabel:index>` modules.
Example usage:
    Using the single threshold method:
        .. code-block:: bash
            python idif_necktangle_cli.py single --pet-4d-filepath /path/to/pet_4d.nii --carotid-necktangle-mask-3d-filepath /path/to/mask.nii --percentile 90
    Using the double threshold method:
        .. code-block:: bash
            python idif_necktangle_cli.py double --necktangle-filepath /path/to/necktangle.npy --percentile 90 --frame-start-times-filepath /path/to/start_times.npy --frame-duration-times-filepath /path/to/duration_times.npy
See Also:
    :mod:`pet_cli.idif_necktangle` - Module for calculating IDIF using necktangle methods.
"""

import argparse
import numpy as np
import nibabel as nib
from ..input_function.idif_necktangle import (
    single_threshold_idif_from_4d_pet_with_necktangle,
    double_threshold_idif_from_4d_pet_necktangle,
    get_frame_time_midpoints
)


def main():
    """
    Main function to handle command-line arguments and execute the appropriate IDIF calculation method.
    The script supports two methods:
    - Single threshold method
    - Double threshold method
    The user must provide the required file paths and parameters as command-line arguments.
    Raises:
        ValueError: If any input file paths are invalid or if required arguments are missing.
    """
    parser = argparse.ArgumentParser(description="Calculate IDIF using necktangle methods.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Single threshold sub-command
    single_parser = subparsers.add_parser("single", help="Use single threshold method")
    single_parser.add_argument("--pet-4d-filepath", type=str, required=True,
                               help="Path to the input 4D PET image file.")
    single_parser.add_argument("--carotid-necktangle-mask-3d-filepath", type=str, required=True,
                               help="Path to the carotid mask file.")
    single_parser.add_argument("--percentile", type=float, required=True, help="Percentile for threshold value.")
    single_parser.add_argument("--bolus_start_frame", type=int, default=3, help="Bolus start frame (default: 3).")
    single_parser.add_argument("--bolus_end_frame", type=int, default=7, help="Bolus end frame (default: 7).")

    # Double threshold sub-command
    double_parser = subparsers.add_parser("double", help="Use double threshold method")
    double_parser.add_argument("--necktangle-filepath", type=str, required=True,
                               help="Path to the 4D PET necktangle matrix file.")
    double_parser.add_argument("--percentile", type=float, required=True, help="Percentile for manual threshold value.")
    double_parser.add_argument("--frame-start-times-filepath", type=str, required=True,
                               help="Path to the frame start times file.")
    double_parser.add_argument("--frame-duration-times-filepath", type=str, required=True,
                               help="Path to the frame duration times file.")

    args = parser.parse_args()

    if args.command == "single":
        pet_4d_data = nib.load(args.pet_4d_filepath).get_fdata()
        carotid_necktangle_mask_3d_data = nib.load(args.carotid_necktangle_mask_3d_filepath).get_fdata()
        tac = single_threshold_idif_from_4d_pet_with_necktangle(
            pet_4d_data=pet_4d_data,
            carotid_necktangle_mask_3d_data=carotid_necktangle_mask_3d_data,
            percentile=args.percentile,
            bolus_start_frame=args.bolus_start_frame,
            bolus_end_frame=args.bolus_end_frame
        )
        np.savetxt("single_threshold_tac.csv", tac, delimiter=",")
        print("Single threshold IDIF calculation complete. TAC saved to single_threshold_tac.csv")

    elif args.command == "double":
        necktangle_matrix = np.load(args.necktangle_filepath)
        frame_start_times = np.load(args.frame_start_times_filepath)
        frame_duration_times = np.load(args.frame_duration_times_filepath)
        frame_midpoint_times = get_frame_time_midpoints(frame_start_times, frame_duration_times)
        tac = double_threshold_idif_from_4d_pet_necktangle(
            necktangle_matrix=necktangle_matrix,
            percentile=args.percentile,
            frame_midpoint_times=frame_midpoint_times
        )
        np.savetxt("double_threshold_tac.csv", tac, delimiter=",")
        print("Double threshold IDIF calculation complete. TAC saved to double_threshold_tac.csv")


if __name__ == "__main__":
    main()
